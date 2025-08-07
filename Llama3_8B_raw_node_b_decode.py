from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch
import json
import socket
import pickle
import time

# 配置 - 节点B的本地模型路径
local_model_path = r"/home/vemu4/yhbian/models/Llama3_8B"
device = "cpu"
HOST = "0.0.0.0"
PORT = 12347
BANDWIDTH_LIMIT = 0.5 * 1024 * 1024  # 带宽限制 0.5MB/s
NETWORK_DELAY = 0.04  # 网络延迟 40ms

def rms_norm(tensor, norm_weights, norm_eps):
    """RMS归一化函数"""
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) * norm_weights

def simulate_network_delay():
    """模拟网络延迟"""
    time.sleep(NETWORK_DELAY)

def calculate_transmission_time(data_size):
    """计算传输时间，基于带宽限制"""
    return data_size / BANDWIDTH_LIMIT

def receive_data_with_bandwidth_control(sock, size):
    """带带宽控制的数据接收函数"""
    transmission_time = calculate_transmission_time(size)
    
    # 模拟网络延迟
    simulate_network_delay()
    
    # 接收数据
    data = b''
    while len(data) < size:
        chunk = sock.recv(min(4096, size - len(data)))
        if not chunk:
            break
        data += chunk
    
    # 模拟传输时间（带宽限制）
    time.sleep(transmission_time)
    
    print(f"节点B：接收数据 {size} 字节，网络延迟: {NETWORK_DELAY*1000:.0f}ms，传输时间: {transmission_time:.3f}秒")
    return data

def send_data_with_bandwidth_control(sock, data):
    """带带宽控制的数据发送函数"""
    data_size = len(data)
    transmission_time = calculate_transmission_time(data_size)
    
    # 模拟网络延迟
    simulate_network_delay()
    
    # 发送数据
    sock.sendall(data)
    
    # 模拟传输时间（带宽限制）
    time.sleep(transmission_time)
    
    print(f"节点B：发送数据 {data_size} 字节，网络延迟: {NETWORK_DELAY*1000:.0f}ms，传输时间: {transmission_time:.3f}秒")

def main():
    print("节点B：启动Llama-3-8B Raw Decode服务器...")
    print(f"节点B：带宽限制: {BANDWIDTH_LIMIT/1024/1024:.1f}MB/s, 网络延迟: {NETWORK_DELAY*1000:.0f}ms")
    
    # 预先加载本地模型参数
    print(f"节点B：从本地路径加载模型参数: {local_model_path}")
    
    # 加载配置
    with open(f"{local_model_path}/params.json", "r") as f:
        config = json.load(f)
    
    dim = config["dim"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    vocab_size = config["vocab_size"]
    norm_eps = config["norm_eps"]
    rope_theta = torch.tensor(config["rope_theta"])
    
    print(f"节点B：配置加载完成 - 层数: {n_layers}, 头数: {n_heads}")
    
    # 加载模型权重
    print("节点B：加载模型权重...")
    model = torch.load(f"{local_model_path}/consolidated.00.pth")
    
    # 加载tokenizer
    tokenizer_path = f"{local_model_path}/tokenizer.model"
    special_tokens = [
        "<|begin_of_text|>", "<|end_of_text|>", "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>", "<|reserved_special_token_2|>", "<|reserved_special_token_3|>",
        "<|start_header_id|>", "<|end_header_id|>", "<|reserved_special_token_4|>", "<|eot_id|>",
    ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
    
    mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
    tokenizer = tiktoken.Encoding(
        name=Path(tokenizer_path).name,
        pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        mergeable_ranks=mergeable_ranks,
        special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
    )
    
    print("节点B：模型和tokenizer预加载完成，等待连接...")
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"节点B：监听端口 {PORT}...")
        
        conn, addr = s.accept()
        with conn:
            print(f"节点B：接收到来自 {addr} 的连接")
            
            # 使用带宽控制接收数据大小
            data_size_bytes = receive_data_with_bandwidth_control(conn, 4)
            data_size = int.from_bytes(data_size_bytes, byteorder='big')
            print(f"节点B：准备接收 {data_size:,} 字节的数据 ({data_size/1024/1024:.2f} MB)...")
            
            # 使用带宽控制接收实际数据
            received_data = receive_data_with_bandwidth_control(conn, data_size)
            print("节点B：数据接收完成，开始反序列化...")
            
            # 反序列化数据
            data = pickle.loads(received_data)
            print("节点B：数据反序列化完成，开始Decode阶段...")
            
            # 获取推理状态数据（不再包含模型参数）
            next_token = data['next_token']
            kv_cache = data['kv_cache']
            max_new_tokens = data['max_new_tokens']
            seq_len = data['seq_len']
            prompt = data.get('prompt', 'unknown')  # 用于验证
            
            print(f"节点B：接收到推理状态 - 序列长度: {seq_len}, 目标生成: {max_new_tokens} tokens")
            print(f"节点B：原始prompt: {prompt[:50]}...")
            
            generated_tokens = [next_token]
            current_seq_len = seq_len
            
            # 获取结束token的ID
            eot_token_id = None
            for token, token_id in tokenizer._special_tokens.items():
                if token == "<|eot_id|>":
                    eot_token_id = token_id
                    break
            if eot_token_id is None:
                eot_token_id = 128009  # 默认值
            
            print(f"节点B：结束token ID: {eot_token_id}")
            
            # =========================  Decode  ==========================
            for step in range(max_new_tokens - 1):
                # 创建当前token的embedding
                current_token = torch.tensor([generated_tokens[-1]])
                embedding_layer = torch.nn.Embedding(vocab_size, dim)
                embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])
                token_embedding = embedding_layer(current_token).to(torch.bfloat16)
                
                # RoPE频率计算（针对当前位置）
                zero_to_one_split_into_64_parts = torch.tensor(range(64))/64
                freqs = 1.0 / (rope_theta ** zero_to_one_split_into_64_parts)
                current_pos = current_seq_len  # 当前位置
                freqs_for_current_token = current_pos * freqs
                freqs_cis_current = torch.polar(torch.ones_like(freqs_for_current_token), freqs_for_current_token)
                
                final_embedding = token_embedding
                
                # 逐层处理
                for layer in range(n_layers):
                    qkv_attention_store = []
                    layer_embedding_norm = rms_norm(final_embedding, model[f"layers.{layer}.attention_norm.weight"], norm_eps)
                    
                    # 获取当前层的权重
                    q_layer = model[f"layers.{layer}.attention.wq.weight"].view(n_heads, -1, dim)
                    w_layer = model[f"layers.{layer}.attention.wo.weight"]
                    
                    # 获取缓存的K, V
                    cached_k = kv_cache[layer]['k_cache']  # [n_kv_heads, prev_seq_len, head_dim]
                    cached_v = kv_cache[layer]['v_cache']  # [n_kv_heads, prev_seq_len, head_dim]
                    
                    # 计算当前token的K, V
                    k_layer = model[f"layers.{layer}.attention.wk.weight"].view(n_kv_heads, -1, dim)
                    v_layer = model[f"layers.{layer}.attention.wv.weight"].view(n_kv_heads, -1, dim)
                    
                    new_k_cache = []
                    new_v_cache = []
                    
                    for kv_head in range(n_kv_heads):
                        k_layer_head = k_layer[kv_head]
                        v_layer_head = v_layer[kv_head]
                        
                        # 计算当前token的k, v
                        k_current = torch.matmul(layer_embedding_norm, k_layer_head.T)  # [1, head_dim]
                        v_current = torch.matmul(layer_embedding_norm, v_layer_head.T)  # [1, head_dim]
                        
                        # 应用RoPE到k
                        k_current_split = k_current.float().view(1, -1, 2)
                        k_current_complex = torch.view_as_complex(k_current_split)
                        k_current_rotated_complex = k_current_complex * freqs_cis_current
                        k_current_rotated = torch.view_as_real(k_current_rotated_complex).view(k_current.shape)
                        
                        # 更新K, V缓存
                        updated_k = torch.cat([cached_k[kv_head], k_current_rotated], dim=0)  # [seq_len+1, head_dim]
                        updated_v = torch.cat([cached_v[kv_head], v_current], dim=0)  # [seq_len+1, head_dim]
                        
                        new_k_cache.append(updated_k)
                        new_v_cache.append(updated_v)
                    
                    # 更新缓存
                    kv_cache[layer]['k_cache'] = torch.stack(new_k_cache)
                    kv_cache[layer]['v_cache'] = torch.stack(new_v_cache)
                    
                    # 计算所有头的注意力
                    for head in range(n_heads):
                        q_layer_head = q_layer[head]
                        
                        # 计算当前token的q
                        q_current = torch.matmul(layer_embedding_norm, q_layer_head.T)  # [1, head_dim]
                        
                        # 应用RoPE到q
                        q_current_split = q_current.float().view(1, -1, 2)
                        q_current_complex = torch.view_as_complex(q_current_split)
                        q_current_rotated_complex = q_current_complex * freqs_cis_current
                        q_current_rotated = torch.view_as_real(q_current_rotated_complex).view(q_current.shape)
                        
                        # 获取对应的K, V (每4个头共享)
                        kv_head_idx = head // 4
                        k_all = kv_cache[layer]['k_cache'][kv_head_idx]  # [seq_len+1, head_dim]
                        v_all = kv_cache[layer]['v_cache'][kv_head_idx]  # [seq_len+1, head_dim]
                        
                        # 计算注意力分数
                        qk_scores = torch.matmul(q_current_rotated, k_all.T) / (128**0.5)  # [1, seq_len+1]
                        
                        # Causal mask不需要，因为我们只关注当前token
                        attention_weights = torch.nn.functional.softmax(qk_scores, dim=-1).to(torch.bfloat16)
                        
                        # 计算注意力输出
                        attention_output = torch.matmul(attention_weights, v_all)  # [1, head_dim]
                        qkv_attention_store.append(attention_output)
                    
                    # 合并所有头的输出
                    stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
                    embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T)
                    embedding_after_edit = final_embedding + embedding_delta
                    embedding_after_edit_normalized = rms_norm(embedding_after_edit, model[f"layers.{layer}.ffn_norm.weight"], norm_eps)
                    
                    # FFN
                    w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
                    w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
                    w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
                    output_after_feedforward = torch.matmul(
                        torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * 
                        torch.matmul(embedding_after_edit_normalized, w3.T), w2.T
                    )
                    final_embedding = embedding_after_edit + output_after_feedforward
                
                # 最终归一化和输出
                final_embedding = rms_norm(final_embedding, model["norm.weight"], norm_eps)
                logits = torch.matmul(final_embedding[0], model["output.weight"].T)
                next_token = torch.argmax(logits, dim=-1)
                
                generated_tokens.append(next_token.item())
                current_seq_len += 1
                
                # 检查是否遇到结束token
                if next_token.item() == eot_token_id:
                    print(f"节点B：遇到结束token，提前结束生成 (已生成 {step+1} tokens)")
                    break
                
                if (step + 1) % 10 == 0:
                    print(f"节点B：已生成 {step+1}/{max_new_tokens-1} tokens")
            
            # 解码生成的文本
            try:
                response = tokenizer.decode(generated_tokens)
            except Exception as e:
                print(f"节点B：解码出错: {e}")
                # 备用解码方式
                response = "".join([tokenizer.decode([token]) for token in generated_tokens if token < len(tokenizer._mergeable_ranks)])
            
            print(f"节点B：Decode完成，生成了 {len(generated_tokens)} 个tokens")
            print(f"节点B：生成内容预览: {response[:100]}...")
            print("节点B：返回结果给节点A...")
            
            # 使用带宽控制发送结果回节点A
            result_data = pickle.dumps(response)
            result_size = len(result_data)
            
            send_data_with_bandwidth_control(conn, result_size.to_bytes(4, byteorder='big'))
            send_data_with_bandwidth_control(conn, result_data)
            
            print("节点B：结果发送完成")

if __name__ == "__main__":
    main()
