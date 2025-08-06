from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import socket
import pickle
import time

# 配置
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cpu"
max_new_tokens = 32

# 节点B的地址配置
NODE_B_HOST = "127.0.0.1"  # 修改为节点B的实际IP
NODE_B_PORT = 12345

def main():
    # 载入模型、tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 准备输入
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    input_ids = tokenizer([text], return_tensors="pt").input_ids.to(device)

    # 记录推理开始时间
    start_time = time.time()
    print("节点A：开始Prefill阶段...")
    # =========================  Prefill  =========================
    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

    print("节点A：Prefill完成，发送数据给节点B...")
    
    # 准备发送的数据
    data_to_send = {
        'next_token': next_token,
        'past_key_values': past_key_values,
        'max_new_tokens': max_new_tokens,
        'model_name': model_name
    }

    # 连接节点B并发送数据
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((NODE_B_HOST, NODE_B_PORT))
        
        # 序列化并发送数据
        serialized_data = pickle.dumps(data_to_send)
        data_size = len(serialized_data)
        
        # 先发送数据大小
        s.sendall(data_size.to_bytes(4, byteorder='big'))
        # 再发送实际数据
        s.sendall(serialized_data)
        
        print("节点A：数据发送完成，等待节点B返回结果...")
        
        # 接收返回结果
        result_size = int.from_bytes(s.recv(4), byteorder='big')
        result_data = b''
        while len(result_data) < result_size:
            result_data += s.recv(result_size - len(result_data))
        
        response = pickle.loads(result_data)
        
        # 记录推理结束时间
        end_time = time.time()
        
        # 计算统计数据
        total_latency = end_time - start_time
        
        # 计算生成的token数量（从response中解析生成的文本）
        generated_tokens = len(tokenizer.encode(response))
        
        # 计算吞吐量 (tokens/second)
        throughput = generated_tokens / total_latency if total_latency > 0 else 0
        
        print(f"节点A：收到最终结果：\n{response}")
        print("\n" + "="*50)
        print("推理性能统计:")
        print(f"总时延: {total_latency:.3f} 秒")
        print(f"生成Token数量: {generated_tokens}")
        print(f"吞吐量: {throughput:.2f} Token/s")
        print("="*50)

if __name__ == "__main__":
    main()
