from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch
import json
import socket
import pickle
import time
import random

# 配置
model_path = r"C:\Users\yhbia\Desktop\边彦晖\Proj\Meta_llama\Meta-Llama-3-8B\original"
device = "cpu"
max_new_tokens = 20

# 节点B的地址配置
NODE_B_HOST = "192.168.1.104"  # 修改为节点B的实际IP
NODE_B_PORT = 12347  # 使用不同端口

# 网络模拟配置
PACKET_LOSS_RATE = 10e-2  # 丢包率 (0.0-1.0)，0.1表示10%丢包率
MAX_RETRIES = 15  # 最大重传次数
BANDWIDTH_LIMIT = 0.5 * 1024 * 1024  # 带宽限制 0.5MB/s
NETWORK_DELAY = 0.04  # 网络延迟 40ms
MTU_SIZE = 1500  # 最大传输单元，字节
HEADER_SIZE = 20  # IP头部大小（简化）
PAYLOAD_SIZE = MTU_SIZE - HEADER_SIZE  # 实际有效载荷大小

def simulate_packet_loss():
    """模拟丢包，返回True表示丢包"""
    return random.random() < PACKET_LOSS_RATE

def simulate_network_delay():
    """模拟网络延迟"""
    time.sleep(NETWORK_DELAY)

def calculate_transmission_time(data_size):
    """计算传输时间，基于带宽限制"""
    return data_size / BANDWIDTH_LIMIT

def split_data_into_packets(data):
    """将数据分割为数据包"""
    packets = []
    for i in range(0, len(data), PAYLOAD_SIZE):
        packet = data[i:i + PAYLOAD_SIZE]
        packets.append(packet)
    return packets

def send_packet_with_retransmission(sock, packet, packet_id):
    """发送单个数据包，支持重传"""
    packet_size = len(packet)
    total_attempts = 0
    transmission_time = 0
    
    for attempt in range(MAX_RETRIES + 1):
        total_attempts += 1
        
        # 模拟网络延迟
        simulate_network_delay()
        delay_time = NETWORK_DELAY
        
        # 计算传输时间（包括头部）
        packet_transmission_time = calculate_transmission_time(packet_size + HEADER_SIZE)
        
        if simulate_packet_loss() and attempt < MAX_RETRIES:
            print(f"节点A：数据包{packet_id}丢失，第{attempt + 1}次尝试失败，准备重传...")
            transmission_time += delay_time + packet_transmission_time
            time.sleep(0.001)  # 短暂等待重传
            continue
        else:
            # 发送成功
            sock.sendall(packet)
            transmission_time += delay_time + packet_transmission_time
            time.sleep(packet_transmission_time)
            
            if attempt > 0:
                print(f"节点A：数据包{packet_id}重传成功，共尝试{attempt + 1}次")
            
            return True, transmission_time, total_attempts
    
    print(f"节点A：数据包{packet_id}发送失败，已达到最大重传次数{MAX_RETRIES}")
    return False, transmission_time, total_attempts

def receive_packet_with_retransmission(sock, packet_size, packet_id):
    """接收单个数据包，支持重传"""
    total_attempts = 0
    transmission_time = 0
    
    for attempt in range(MAX_RETRIES + 1):
        total_attempts += 1
        
        # 模拟网络延迟
        simulate_network_delay()
        delay_time = NETWORK_DELAY
        
        # 计算传输时间（包括头部）
        packet_transmission_time = calculate_transmission_time(packet_size + HEADER_SIZE)
        
        if simulate_packet_loss() and attempt < MAX_RETRIES:
            print(f"节点A：数据包{packet_id}接收丢失，第{attempt + 1}次尝试失败...")
            transmission_time += delay_time + packet_transmission_time
            time.sleep(0.001)  # 短暂等待重传
            continue
        else:
            # 接收成功
            data = b''
            while len(data) < packet_size:
                data += sock.recv(packet_size - len(data))
            
            transmission_time += delay_time + packet_transmission_time
            time.sleep(packet_transmission_time)
            
            if attempt > 0:
                print(f"节点A：数据包{packet_id}接收重传成功，共尝试{attempt + 1}次")
            
            return data, transmission_time, total_attempts
    
    print(f"节点A：数据包{packet_id}接收失败，已达到最大重传次数{MAX_RETRIES}")
    return None, transmission_time, total_attempts

def send_data_with_packet_simulation(sock, data):
    """使用数据包级别模拟发送数据"""
    print(f"节点A：开始发送数据，总大小: {len(data):,} 字节")
    
    # 分割数据为数据包
    packets = split_data_into_packets(data)
    total_packets = len(packets)
    
    print(f"节点A：数据分割为 {total_packets} 个数据包 (MTU: {MTU_SIZE}, 有效载荷: {PAYLOAD_SIZE})")
    
    total_transmission_time = 0
    total_attempts = 0
    failed_packets = 0
    
    for i, packet in enumerate(packets):
        success, transmission_time, attempts = send_packet_with_retransmission(sock, packet, i + 1)
        total_transmission_time += transmission_time
        total_attempts += attempts
        
        if not success:
            failed_packets += 1
        
        # 显示进度
        if (i + 1) % max(1, total_packets // 10) == 0 or i == total_packets - 1:
            progress = ((i + 1) / total_packets) * 100
            print(f"节点A：发送进度 {progress:.1f}% ({i + 1}/{total_packets})")
    
    if failed_packets > 0:
        print(f"节点A：发送完成，但有 {failed_packets} 个数据包发送失败")
        return False, total_transmission_time, total_attempts
    
    print(f"节点A：数据发送完成")
    print(f"  - 总传输时间: {total_transmission_time:.3f}秒")
    print(f"  - 总重传次数: {total_attempts - total_packets}")
    print(f"  - 平均每包重传次数: {(total_attempts - total_packets) / total_packets:.2f}")
    
    return True, total_transmission_time, total_attempts

def receive_data_with_packet_simulation(sock, total_size):
    """使用数据包级别模拟接收数据"""
    print(f"节点A：开始接收数据，总大小: {total_size:,} 字节")
    
    # 计算需要接收的数据包数量
    total_packets = (total_size + PAYLOAD_SIZE - 1) // PAYLOAD_SIZE
    print(f"节点A：预期接收 {total_packets} 个数据包")
    
    received_data = b''
    total_transmission_time = 0
    total_attempts = 0
    failed_packets = 0
    
    for i in range(total_packets):
        # 计算当前包的大小
        remaining_size = total_size - len(received_data)
        packet_size = min(PAYLOAD_SIZE, remaining_size)
        
        packet_data, transmission_time, attempts = receive_packet_with_retransmission(sock, packet_size, i + 1)
        total_transmission_time += transmission_time
        total_attempts += attempts
        
        if packet_data is None:
            failed_packets += 1
            print(f"节点A：数据包{i + 1}接收失败，可能导致数据不完整")
            break
        
        received_data += packet_data
        
        # 显示进度
        if (i + 1) % max(1, total_packets // 10) == 0 or i == total_packets - 1:
            progress = ((i + 1) / total_packets) * 100
            print(f"节点A：接收进度 {progress:.1f}% ({i + 1}/{total_packets})")
    
    if failed_packets > 0:
        print(f"节点A：接收完成，但有 {failed_packets} 个数据包接收失败")
        return None, total_transmission_time, total_attempts
    
    print(f"节点A：数据接收完成")
    print(f"  - 总传输时间: {total_transmission_time:.3f}秒")
    print(f"  - 总重传次数: {total_attempts - total_packets}")
    print(f"  - 平均每包重传次数: {(total_attempts - total_packets) / total_packets:.2f}")
    
    return received_data, total_transmission_time, total_attempts

# 替换原有的发送和接收函数
def send_data_with_bandwidth_control(sock, data):
    """保持兼容性的包装函数"""
    success, _, _ = send_data_with_packet_simulation(sock, data)
    return success

def receive_data_with_bandwidth_control(sock, size):
    """保持兼容性的包装函数"""
    data, _, _ = receive_data_with_packet_simulation(sock, size)
    return data

def send_data_with_loss_simulation(sock, data):
    """使用新的数据包级别模拟"""
    success, transmission_time, attempts = send_data_with_packet_simulation(sock, data)
    return success

def receive_data_with_loss_simulation(sock, size):
    """使用新的数据包级别模拟"""
    data, transmission_time, attempts = receive_data_with_packet_simulation(sock, size)
    return data

def rms_norm(tensor, norm_weights):
    """RMS归一化函数"""
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) * norm_weights

def analyze_data_size(data_to_send):
    """分析和显示数据传输量"""
    serialized_data = pickle.dumps(data_to_send)
    total_size = len(serialized_data)
    
    print(f"\n{'='*60}")
    print("数据传输量分析 (Llama-3-8B Raw):")
    print(f"序列化后总数据大小: {total_size:,} 字节 ({total_size/1024/1024:.2f} MB)")
    
    # 分析各组件大小
    next_token_size = len(pickle.dumps(data_to_send['next_token']))
    kv_cache_size = len(pickle.dumps(data_to_send['kv_cache']))
    other_size = total_size - next_token_size - kv_cache_size
    
    print(f"  - Next Token: {next_token_size:,} 字节 ({next_token_size/1024:.2f} KB)")
    print(f"  - KV Cache: {kv_cache_size:,} 字节 ({kv_cache_size/1024/1024:.2f} MB)")
    print(f"  - 其他数据: {other_size:,} 字节 ({other_size/1024:.2f} KB)")
    print(f"{'='*60}\n")
    
    return serialized_data

def main():
    global norm_eps
    
    print("节点A：启动Llama-3-8B Raw Prefill节点...")
    print(f"节点A：网络模拟配置 - 丢包率: {PACKET_LOSS_RATE*100:.9f}%, 最大重传次数: {MAX_RETRIES}")
    print(f"节点A：带宽限制: {BANDWIDTH_LIMIT/1024/1024:.1f}MB/s, 网络延迟: {NETWORK_DELAY*1000:.0f}ms")
    
    # 加载tokenizer
    print("节点A：加载tokenizer...")
    tokenizer_path = f"{model_path}\\tokenizer.model"
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

    # 加载模型权重
    print("节点A：加载模型权重...")
    model = torch.load(f"{model_path}\\consolidated.00.pth")

    # 加载配置
    with open(f"{model_path}\\params.json", "r") as f:
        config = json.load(f)

    dim = config["dim"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    vocab_size = config["vocab_size"]
    norm_eps = config["norm_eps"]
    rope_theta = torch.tensor(config["rope_theta"])

    print("节点A：模型加载完成")
    print(f"节点A：模型配置 - 层数: {n_layers}, 头数: {n_heads}, 维度: {dim}")

    # 准备输入
    prompt = "you are a person with a good heart, and as your neighbor, I "
    # text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    # 修复tiktoken编码问题 - 允许所有特殊token
    tokens = [128000] + tokenizer.encode(prompt)
    tokens = torch.tensor(tokens)
    print(f"节点A：输入tokens长度: {len(tokens)}")

    # 记录推理开始时间
    start_time = time.time()
    print("节点A：开始Prefill阶段...")

    # =========================  Prefill  =========================
    # 初始化embedding
    embedding_layer = torch.nn.Embedding(vocab_size, dim)
    embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])
    token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16)

    # RoPE频率计算
    zero_to_one_split_into_64_parts = torch.tensor(range(64))/64
    freqs = 1.0 / (rope_theta ** zero_to_one_split_into_64_parts)
    freqs_for_each_token = torch.outer(torch.arange(len(tokens)), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)

    # 存储KV cache用于传输
    kv_cache = []
    final_embedding = token_embeddings_unnormalized

    # 只运行Prefill阶段（处理所有输入tokens）
    for layer in range(n_layers):
        qkv_attention_store = []
        layer_embedding_norm = rms_norm(final_embedding, model[f"layers.{layer}.attention_norm.weight"])
        
        # 获取权重
        q_layer = model[f"layers.{layer}.attention.wq.weight"].view(n_heads, -1, dim)
        k_layer = model[f"layers.{layer}.attention.wk.weight"].view(n_kv_heads, -1, dim)
        v_layer = model[f"layers.{layer}.attention.wv.weight"].view(n_kv_heads, -1, dim)
        w_layer = model[f"layers.{layer}.attention.wo.weight"]

        # 计算所有头的注意力
        layer_k_cache = []
        layer_v_cache = []
        
        for head in range(n_heads):
            q_layer_head = q_layer[head]
            k_layer_head = k_layer[head//4]
            v_layer_head = v_layer[head//4]
            
            q_per_token = torch.matmul(layer_embedding_norm, q_layer_head.T)
            k_per_token = torch.matmul(layer_embedding_norm, k_layer_head.T)
            v_per_token = torch.matmul(layer_embedding_norm, v_layer_head.T)
            
            # 保存K,V用于后续Decode
            if head % 4 == 0:  # 每4个头共享一个KV
                layer_k_cache.append(k_per_token)
                layer_v_cache.append(v_per_token)
            
            # RoPE旋转
            q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
            q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
            q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis)
            q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)
            
            k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
            k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
            k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis)
            k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)
            
            # 注意力计算
            qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(128)**0.5
            mask = torch.full((len(tokens), len(tokens)), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            qk_per_token_after_masking = qk_per_token + mask
            qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
            qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
            qkv_attention_store.append(qkv_attention)

        # 保存该层的KV cache
        kv_cache.append({
            'k_cache': torch.stack(layer_k_cache),  # [n_kv_heads, seq_len, head_dim]
            'v_cache': torch.stack(layer_v_cache)   # [n_kv_heads, seq_len, head_dim]
        })

        # 完成该层的前向传播
        stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
        embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T)
        embedding_after_edit = final_embedding + embedding_delta
        embedding_after_edit_normalized = rms_norm(embedding_after_edit, model[f"layers.{layer}.ffn_norm.weight"])
        
        # FFN
        w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
        w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
        w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
        output_after_feedforward = torch.matmul(torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T)
        final_embedding = embedding_after_edit + output_after_feedforward

    # 最终归一化和输出
    final_embedding = rms_norm(final_embedding, model["norm.weight"])
    logits = torch.matmul(final_embedding[-1], model["output.weight"].T)
    next_token = torch.argmax(logits, dim=-1)

    print("节点A：Prefill完成，准备发送数据给节点B...")

    # 准备发送的数据（移除模型参数，只发送推理状态）
    data_to_send = {
        'next_token': next_token.item(),
        'kv_cache': kv_cache,
        'max_new_tokens': max_new_tokens,
        'seq_len': len(tokens),
        'prompt': prompt  # 发送原始prompt用于验证
    }

    # 分析数据传输量
    serialized_data = analyze_data_size(data_to_send)

    # 连接节点B并发送数据
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((NODE_B_HOST, NODE_B_PORT))
        
        data_size = len(serialized_data)
        
        # 使用带丢包模拟的发送函数
        print("节点A：发送数据大小...")
        if not send_data_with_loss_simulation(s, data_size.to_bytes(4, byteorder='big')):
            print("节点A：发送数据大小失败，终止程序")
            return
        
        print("节点A：发送实际数据...")
        if not send_data_with_loss_simulation(s, serialized_data):
            print("节点A：发送实际数据失败，终止程序")
            return
        
        print("节点A：数据发送完成，等待节点B返回结果...")
        
        # 接收返回结果
        result_size_data = receive_data_with_loss_simulation(s, 4)
        if result_size_data is None:
            print("节点A：接收结果大小失败，终止程序")
            return
        
        result_size = int.from_bytes(result_size_data, byteorder='big')
        
        result_data = receive_data_with_loss_simulation(s, result_size)
        if result_data is None:
            print("节点A：接收结果数据失败，终止程序")
            return
        
        response = pickle.loads(result_data)
        
        # 记录推理结束时间
        end_time = time.time()
        
        # 计算统计数据
        total_latency = end_time - start_time
        
        # 修复token计数
        try:
            generated_tokens = len(tokenizer.encode(response, allowed_special="all"))
        except Exception as e:
            print(f"节点A：token计数出错: {e}")
            # 备用计数方式
            generated_tokens = len(response.split()) if isinstance(response, str) else 0
        
        throughput = generated_tokens / total_latency if total_latency > 0 else 0
        
        print(f"节点A：收到最终结果：\n{response}")
        print("\n" + "="*50)
        print("推理性能统计 (Llama-3-8B Raw):")
        print(f"总时延: {total_latency:.3f} 秒")
        print(f"生成Token数量: {generated_tokens}")
        print(f"吞吐量: {throughput:.2f} Token/s")
        print("="*50)

if __name__ == "__main__":
    main()
