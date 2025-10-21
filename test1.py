from transformers import AutoTokenizer, AutoModel
import torch

# 1. 加载刚才下载的模型和分词器
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese")

# 2. 准备输入文本（试试中文！）
text = "今天天气真好，我想去公园散步。"

# 3. 文本预处理（Tokenization）
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
print("分词结果:", tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))

# 4. 模型推理
with torch.no_grad():
    outputs = model(**inputs)

# 5. 查看结果
embeddings = outputs.last_hidden_state
print(f"输入文本: {text}")
print(embeddings[:,0,:])# 查看第一个词向量
print(f"词向量形状: {embeddings.shape}")  # [batch_size, sequence_length, hidden_size]
print(f"最后一个词的向量: {embeddings[0, -1, :10]}...")  # 只看前10维
