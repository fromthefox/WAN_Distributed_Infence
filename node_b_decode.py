from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import socket
import pickle

# 配置
model_name = r"/yhbian_wan_d_inf"
device = "cpu"
HOST = "0.0.0.0"  # 监听所有接口
PORT = 12345

def main():
    print("节点B：启动Decode服务器...")
    
    # 预先加载模型和tokenizer
    print("节点B：加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("节点B：模型加载完成")
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"节点B：监听端口 {PORT}...")
        
        conn, addr = s.accept()
        with conn:
            print(f"节点B：接收到来自 {addr} 的连接")
            
            # 接收数据大小
            data_size = int.from_bytes(conn.recv(4), byteorder='big')
            print(f"节点B：准备接收 {data_size} 字节的数据...")
            
            # 接收实际数据
            received_data = b''
            while len(received_data) < data_size:
                received_data += conn.recv(data_size - len(received_data))
            
            # 反序列化数据
            data = pickle.loads(received_data)
            print("节点B：数据接收完成，开始Decode阶段...")
            
            # 获取Prefill的结果
            next_token = data['next_token'].to(device)
            past_key_values = data['past_key_values']
            max_new_tokens = data['max_new_tokens']
            
            generated_tokens = [next_token.item()]
            
            # =========================  Decode  ==========================
            for i in range(max_new_tokens - 1):
                with torch.no_grad():
                    outputs = model(
                        next_token,
                        past_key_values=past_key_values
                    )
                next_token_logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated_tokens.append(next_token.item())
                
                if i % 5 == 0:  # 每5个token打印一次进度
                    print(f"节点B：已生成 {i+1}/{max_new_tokens-1} tokens")
            
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print("节点B：Decode完成，返回结果给节点A...")
            
            # 发送结果回节点A
            result_data = pickle.dumps(response)
            result_size = len(result_data)
            
            conn.sendall(result_size.to_bytes(4, byteorder='big'))
            conn.sendall(result_data)
            
            print("节点B：结果发送完成")

if __name__ == "__main__":
    main()
