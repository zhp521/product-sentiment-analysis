from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 加载本地模型（假设模型路径为 ./bert-base-chinese）
model_path = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_path)

# 如果要做分类，需要加载带分类头的模型（此处以二分类为例）
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

text = "手机质量非常好，价格也很实惠！"  # 待分析的评论

# 使用分词器处理文本
inputs = tokenizer(
    text,
    return_tensors="pt",           # 返回PyTorch张量
    truncation=True,               # 截断超长文本
    padding=True,                  # 填充到相同长度
    max_length=512                 # 最大长度
)
print(inputs)
# 设置为评估模式
model.eval()

with torch.no_grad():
    outputs = model(**inputs)
    print(outputs.logits)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(predictions)
    predicted_label = torch.argmax(predictions).item()

# 输出结果
label_map = {0: "负面", 1: "正面"}
print(f"评论: {text}")
print(f"情感倾向: {label_map[predicted_label]} (置信度: {predictions[0][predicted_label]:.3f})")
