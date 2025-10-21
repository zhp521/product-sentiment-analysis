from datasets import load_dataset

# 加载数据集（会自动识别本地文件）
dataset = load_dataset('./ChnSentiCorp/data')  # 当前目录就是数据集根目录
print(dataset)
# 查看训练集的前3条数据
print("=== 训练集前3条 ===")
train_data = dataset['train']
for i in range(3):
    print(train_data[i])
    print("---")

# 查看数据集信息
print("\n=== 数据集信息 ===")
print(f"训练集大小: {len(dataset['train'])}")
print(f"测试集大小: {len(dataset['test'])}") 
print(f"验证集大小: {len(dataset['validation'])}")

# 查看数据结构
print("\n=== 数据特征 ===")
print(dataset['train'].features)
