from data_wash import clean_text,clean_dataset,remove_stopwords,process_stopwords,remove_duplicates
# from data_enhence import back_translation
from datasets import load_dataset
import pandas as pd

# 1. 加载数据
dataset = load_dataset('ChnSentiCorp')

# 2. 数据清洗流程
def data_cleaning_pipeline(dataset):
    print("开始数据清洗...")
    # 基础清洗
    cleaned_dataset = clean_dataset(dataset)
    print("基础清洗完成")
    
    # 去除重复
    cleaned_dataset['train'] = remove_duplicates(cleaned_dataset['train'])
    cleaned_dataset['test'] = remove_duplicates(cleaned_dataset['test'])
    cleaned_dataset['validation'] = remove_duplicates(cleaned_dataset['validation'])
    print("去重完成")
    
    # 停用词处理（可选，根据任务决定）
    cleaned_dataset = process_stopwords(cleaned_dataset)
    
    return cleaned_dataset

# # 3. 数据增强流程
# def data_augmentation_pipeline(dataset, augment_ratio=0.3):
#     """对部分数据进行增强"""
#     print("开始数据增强...")
    
#     augmented_examples = []
    
#     for split in ['train', 'validation']:
#         split_data = dataset[split]
#         num_to_augment = int(len(split_data) * augment_ratio)
        
#         # 随机选择要增强的样本
#         indices_to_augment = random.sample(range(len(split_data)), num_to_augment)
        
#         for idx in indices_to_augment:
#             original_example = split_data[idx]
            
#             # 选择增强方法
#             aug_text = synonym_replacement(original_example['text'], n=1)
#             # aug_text=original_example['text']
            
#             augmented_examples.append({
#                 'text': aug_text,
#                 'label': original_example['label']
#             })
    
#     # 将增强数据添加到原数据集
#     from datasets import concatenate_datasets, Dataset
    
#     if augmented_examples:
#         augmented_dataset = Dataset.from_list(augmented_examples)
#         for split in ['train', 'validation']:
#             dataset[split] = concatenate_datasets([dataset[split], augmented_dataset])
    
#     print(f"数据增强完成，新增 {len(augmented_examples)} 个样本")
#     return dataset

# 4. 执行完整流程
def main():
    # 加载原始数据
    dataset = load_dataset('ChnSentiCorp')
    print("原始数据统计:")
    for split in ['train', 'validation', 'test']:
        print(f"{split}: {len(dataset[split])} 条数据")
    
    # 数据清洗
    cleaned_dataset = data_cleaning_pipeline(dataset)
    
    # # 数据增强（只在训练集和验证集上进行）
    # final_dataset = data_augmentation_pipeline(cleaned_dataset, augment_ratio=0.3)
    
    # print("\n处理后数据统计:")
    # for split in ['train', 'validation', 'test']:
    #     print(f"{split}: {len(final_dataset[split])} 条数据")
    
    # 保存处理后的数据
    # final_dataset.save_to_disk('./ChnSentiCorp_cleaned')
    cleaned_dataset.save_to_disk('./ChnSentiCorp_cleaned')
    
    # return final_dataset
    return cleaned_dataset

# 运行
if __name__ == "__main__":
    final_dataset = main()