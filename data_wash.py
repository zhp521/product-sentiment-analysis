import re
import jieba

#基础文本清洗
def clean_text(text):
    """
    基础文本清洗函数
    """
    # 去除多余的标点重复（如"。。。。。。" -> "。"）
    text = re.sub(r'[。！？，；]{2,}', lambda x: x.group()[0], text)# 效果是：将连续重复的标点替换为单个标点
    
    # 去除特殊字符，但保留中文、英文、数字和基本标点
    text = re.sub(r'[^\w\s\u4e00-\u9fff。！？，；：（）《》]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()# 去除多余空格
    print(type(text))
    
    return text

# 批量清洗数据集中的文本
def clean_dataset(dataset):# dataset.map()对数据集中的每个样本应用某个函数，example是数据集中的一个样本，通常是 dict类型
    cleaned_dataset = dataset.map(lambda example: {
        **example,  # 展开所有原有字段
        'text': clean_text(example['text'])
    })
    return cleaned_dataset


#处理停用词
# 创建或加载停用词表
stopwords = set([
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '他', '她', '它'
])

def remove_stopwords(text):
    words = jieba.lcut(text)
    filtered_words = [word for word in words if word not in stopwords and len(word) > 1]
    return ' '.join(filtered_words)

# 对数据集进行停用词清洗
def process_stopwords(dataset):
    processed_dataset = dataset.map(lambda example: {
        'text': remove_stopwords(example['text'])
    })
    return processed_dataset

#去除重复数据

def remove_duplicates(dataset):
    """去除完全重复的样本"""
    texts_seen = set()          # 1. 初始化空集合记录已见文本
    unique_indices = []         # 2. 存储非重复样本的索引

    for i, example in enumerate(dataset):  # 3. 遍历数据集
        if example['text'] not in texts_seen:  # 4. 如果文本未见过
            texts_seen.add(example['text'])     # 5. 记录该文本
            unique_indices.append(i)            # 6. 保存索引

    return dataset.select(unique_indices)  # 7. 返回去重后的数据集
