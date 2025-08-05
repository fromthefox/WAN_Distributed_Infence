from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
device      = "cpu"
max_new_tokens = 32

# 0. 载入模型、tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 1. 把对话模板化、tokenize，拿到 input_ids（Prefill 的输入）
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

# =========================  Prefill  =========================
with torch.no_grad():
    outputs = model(input_ids)                     # 第一次前向
    next_token_logits = outputs.logits[:, -1, :]   # 用最后一步的 logits
    past_key_values   = outputs.past_key_values    # 缓存的 KV
    next_token        = torch.argmax(next_token_logits, dim=-1, keepdim=True)

generated_tokens = [next_token.item()]

# =========================  Decode  ==========================
for _ in range(max_new_tokens - 1):
    with torch.no_grad():
        outputs = model(
            next_token,                      # 只送 1 个 token
            past_key_values=past_key_values  # 用缓存的 KV
        )
    next_token_logits = outputs.logits[:, -1, :]
    past_key_values   = outputs.past_key_values
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    generated_tokens.append(next_token.item())

response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(response)
