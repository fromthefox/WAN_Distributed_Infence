from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import socket
import pickle
import time
import random

# 配置
model_name = r"D:\下载项目"
device = "cpu"
max_new_tokens = 256

# 节点B的地址配置
NODE_B_HOST = "192.168.1.104"  # 修改为节点B的实际IP
NODE_B_PORT = 12345

# 网络模拟配置
PACKET_LOSS_RATE = 0.1  # 丢包率 (0.0-1.0)，0.1表示10%丢包率
MAX_RETRIES = 15  # 最大重传次数
BANDWIDTH_LIMIT = 10 * 1024 * 1024  # 带宽限制 10MB/s
NETWORK_DELAY = 0.04  # 网络延迟 40ms

def simulate_packet_loss():
    """模拟丢包，返回True表示丢包"""
    return random.random() < PACKET_LOSS_RATE

def simulate_network_delay():
    """模拟网络延迟"""
    time.sleep(NETWORK_DELAY)

def calculate_transmission_time(data_size):
    """计算传输时间，基于带宽限制"""
    return data_size / BANDWIDTH_LIMIT

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
    
    print(f"节点A：发送数据 {data_size} 字节，传输时间: {transmission_time:.3f}秒")

def receive_data_with_bandwidth_control(sock, size):
    """带带宽控制的数据接收函数"""
    transmission_time = calculate_transmission_time(size)
    
    # 模拟网络延迟
    simulate_network_delay()
    
    # 接收数据
    data = b''
    while len(data) < size:
        data += sock.recv(size - len(data))
    
    # 模拟传输时间（带宽限制）
    time.sleep(transmission_time)
    
    print(f"节点A：接收数据 {size} 字节，传输时间: {transmission_time:.3f}秒")
    return data

def send_data_with_loss_simulation(sock, data):
    """带丢包模拟的数据发送函数"""
    for attempt in range(MAX_RETRIES + 1):
        if simulate_packet_loss() and attempt < MAX_RETRIES:
            print(f"节点A：模拟丢包，第{attempt + 1}次尝试失败，准备重传...")
            time.sleep(0.1)  # 模拟重传延迟
            continue
        else:
            send_data_with_bandwidth_control(sock, data)
            if attempt > 0:
                print(f"节点A：重传成功，共尝试{attempt + 1}次")
            return True
    
    print(f"节点A：发送失败，已达到最大重传次数{MAX_RETRIES}")
    return False

def receive_data_with_loss_simulation(sock, size):
    """带丢包模拟的数据接收函数"""
    for attempt in range(MAX_RETRIES + 1):
        if simulate_packet_loss() and attempt < MAX_RETRIES:
            print(f"节点A：模拟接收丢包，第{attempt + 1}次尝试失败...")
            time.sleep(0.1)  # 模拟重传延迟
            continue
        else:
            data = receive_data_with_bandwidth_control(sock, size)
            if attempt > 0:
                print(f"节点A：接收重传成功，共尝试{attempt + 1}次")
            return data
    
    print(f"节点A：接收失败，已达到最大重传次数{MAX_RETRIES}")
    return None

def main():
    print(f"节点A：网络模拟配置 - 丢包率: {PACKET_LOSS_RATE*100:.1f}%, 最大重传次数: {MAX_RETRIES}")
    print(f"节点A：带宽限制: {BANDWIDTH_LIMIT/1024/1024:.1f}MB/s, 网络延迟: {NETWORK_DELAY*1000:.0f}ms")
    
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
        
        # 使用带丢包模拟的接收函数
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
