import torch
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments, EvalPrediction,
    DataCollatorWithPadding
)
from datasets import Dataset, load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import logging
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 设置日志
logging.basicConfig(level=logging.INFO)# 只有INFO级别及以上的日志消息会被记录（INFO, WARNING, ERROR, CRITICAL）
logger = logging.getLogger(__name__)# 创建一个日志记录器(logger)实例

class SentimentAnalyzer:
    def __init__(self, model_name="bert-base-chinese"):
        self.model_name = model_name # 初始化BERT模型和分词器
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # 商品评价领域关键词（用于增强模型关注度）
        self.domain_keywords = [
            "物流", "快递", "包装", "客服", "性价比", "退货", "发货", "质量", 
            "价格", "效果", "速度", "服务", "态度", "正品", "售后", "体验"
        ]

    # 用于计算模型评估指标​​，p: EvalPrediction​​一个包含模型预测结果和真实标签的对象，通常包含以下属性：
    def compute_metrics(self, p: EvalPrediction):
        """计算评估指标"""
        preds = np.argmax(p.predictions, axis=1)
        accuracy = accuracy_score(p.label_ids, preds)
        f1 = f1_score(p.label_ids, preds, average='binary')
        
        return {
            "accuracy": accuracy,
            "f1": f1,
        }

    # 通过 ​​领域关键词​ 增强模型对特定领域内容的关注。将文本转化为模型可输入的形式
    def tokenize_function(self, examples):
        """Tokenize文本，并添加领域关键词的特殊标记"""
        # 基础tokenization
        tokenized = self.tokenizer( # 调用分词器处理文本
            examples["text"],
            truncation=True,
            padding=False,
            max_length=512
        )
        
        # # 简单的领域关键词增强：在存在关键词的样本上添加特殊标记，实际生产中可采用更复杂的注意力掩码机制
        # for i, text in enumerate(examples["text"]):
        #     if any(keyword in text for keyword in self.domain_keywords):
        #         # 这里可以扩展为修改attention mask，增加关键词权重
        #         pass  # 占位符，高级实现可在此处修改
        
        return tokenized

    def train(self, train_dataset, eval_dataset, output_dir="./bert-sentiment-model"):
        """两阶段训练流程"""
        
        # 初始化模型
        model = BertForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=2  # 假设是二分类情感分析
        )
        model.to(self.device)

        # 数据预处理
        logger.info("Tokenizing datasets...")
        tokenized_train = train_dataset.map(self.tokenize_function, batched=True)
        tokenized_eval = eval_dataset.map(self.tokenize_function, batched=True)

        # 数据收集器，动态填充批次内样本到相同长度（优化显存使用）
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # 训练参数 - 第一阶段：通用情感微调（冻结底层）
        training_args_stage1 = TrainingArguments(
            output_dir=os.path.join(output_dir, "stage1"),# 第一阶段模型和日志的保存路径
            # overwrite_output_dir=True,# 如果输出目录已存在，则覆盖它（避免手动清理）
            num_train_epochs=2,# 训练总轮次（整个数据集遍历2次）
            per_device_train_batch_size=16,# 每个GPU/CPU​​ 的训练批次大小（16个样本/批次）
            per_device_eval_batch_size=32,# ​每个GPU/CPU​​ 的验证批次大小（验证时可更大以加速）
            learning_rate=3e-5,# 初始学习率
            warmup_steps=100, # 前100步逐步增加学习率（避免初始训练不稳定）
            weight_decay=0.01,# L2正则化系数（防止过拟合）
            logging_dir='./logs',# TensorBoard日志保存路径
            logging_steps=50,# 每50个训练步记录一次日志（如loss、学习率）
            # evaluation_strategy="steps",# 按步数（而非轮次）进行验证
            # eval_steps=200,# 每200个训练步验证一次模型性能
            save_steps=500,# 每500步保存一次模型检查点
            # load_best_model_at_end=True,# 训练结束时加载验证集上性能最好的模型
            # metric_for_best_model="f1",# 用F1分数（而非准确率）作为选择最佳模型的标准
        )

        # 第一阶段：冻结BERT底层参数1~6
        logger.info("Stage 1: Generic sentiment fine-tuning (freezing lower layers)")
        for name, param in model.named_parameters():# 遍历模型中所有可训练参数（包括权重和偏置），返回参数名和参数对象
            if 'bert.encoder.layer' in name and int(name.split('.')[3]) < 6:  # 冻结前6层,​​'筛选属于BERT encoder层的参数
                param.requires_grad = False # 关闭梯度计算，冻结该参数（训练时不再更新

        # 创建了一个 ​​Hugging Face 的 Trainer实例​​，用于管理和执行BERT模型的第一阶段训练
        trainer_stage1 = Trainer(
            model=model,# 传入要训练的模型实例（此处是冻结了前6层的BERT分类模型）
            args=training_args_stage1,# 绑定第一阶段的训练参数配置（如学习率、批次大小等）
            train_dataset=tokenized_train,# 传入​​分词后的训练集​​（通过tokenize_function预处理）
            eval_dataset=tokenized_eval,# 传入​​分词后的验证集​​（通过tokenize_function预处理）
            tokenizer=self.tokenizer,# 绑定分词器
            data_collator=data_collator,# 态将批次内样本填充到相同长度（优化显存）
            compute_metrics=self.compute_metrics,# 定义评估指标的计算方法（如准确率、F1）
        )

        # 开始第一阶段训练
        trainer_stage1.train()
        trainer_stage1.save_model()

        # 第二阶段：全参数商品领域微调
        logger.info("Stage 2: Domain-specific fine-tuning (unfreezing all layers)")
        
        # 解冻所有参数
        for param in model.parameters():
            param.requires_grad = True

        # 第二阶段训练参数（更小的学习率）
        training_args_stage2 = TrainingArguments(
            output_dir=os.path.join(output_dir, "stage2"),
            # overwrite_output_dir=True,
            num_train_epochs=3,  # 更多epochs专注于领域适应
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=5e-6,  # 更小的学习率
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir='./logs_stage2',
            logging_steps=50,
            # evaluation_strategy="steps",
            # eval_steps=200,
            save_steps=500,
            # load_best_model_at_end=True,
            # metric_for_best_model="f1",
        )

        trainer_stage2 = Trainer(
            model=model,
            args=training_args_stage2,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        # 开始第二阶段训练
        trainer_stage2.train()
        
        # 保存最终模型
        final_model_path = os.path.join(output_dir, "final_model")
        trainer_stage2.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)# 将分词器保存到与模型相同的目录
        
        logger.info(f"Training completed! Model saved to: {final_model_path}")
        
        return trainer_stage2

    def predict(self, text, model_path=None):
        """使用训练好的模型进行预测"""
        if model_path is None:
            model_path = "/home/zhp//bert/bert-sentiment-model/final_model"
            
        model = BertForSequenceClassification.from_pretrained(model_path,local_files_only=True)
        model.to(self.device)
        model.eval()
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}# 将输入数据（input_ids, attention_mask）移至与模型相同的设备。
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        label = torch.argmax(predictions, dim=1).item()
        confidence = predictions[0][label].item()# 将输入数据（input_ids, attention_mask）移至与模型相同的设备。
        
        sentiment = "正面" if label == 1 else "负面"
        
        return {
            "sentiment": sentiment,# # 情感标签（中文）
            "label": label,# # 数字标签（0/1）
            "confidence": confidence,# # 置信度（0~1）
            "predictions": predictions.cpu().numpy()# 原始概率分布
        }

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(test_dataset, analyzer, model_path="/home/zhp/bert/my-product-sentiment-model/final_model"):
    """
    使用训练好的模型在测试集上进行全面性能评价
    """
    logger.info("开始模型性能评估...")
    
    # 1. 准备测试数据
    def prepare_test_data(example):
        """准备测试样本"""
        return analyzer.tokenize_function(example)
    
    # 对测试集进行tokenization
    tokenized_test = test_dataset.map(prepare_test_data, batched=True)
    
    # 2. 加载训练好的模型
    logger.info("加载训练好的模型...")
    model = BertForSequenceClassification.from_pretrained(
        model_path, 
        local_files_only=True
    )
    model.to(analyzer.device)
    model.eval()
    
    # 3. 批量预测
    logger.info("在测试集上进行预测...")
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(len(tokenized_test)):
            # 获取单个样本
            sample = tokenized_test[i]
            input_ids = torch.tensor(sample['input_ids']).unsqueeze(0).to(analyzer.device)
            attention_mask = torch.tensor(sample['attention_mask']).unsqueeze(0).to(analyzer.device)
            label = sample['label']
            
            # 预测
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_label = torch.argmax(predictions, dim=1).item()
            
            all_predictions.append(predicted_label)
            all_labels.append(label)
            
            # 进度显示
            if (i + 1) % 100 == 0:
                logger.info(f"已处理 {i+1}/{len(tokenized_test)} 个样本")
    
    # 4. 计算评估指标
    logger.info("计算评估指标...")
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    
    # 基础指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # 详细分类报告
    class_report = classification_report(y_true, y_pred, target_names=['负面', '正面'])
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 5. 输出结果
    logger.info("🎯 模型性能评估结果")
    logger.info("=" * 60)
    logger.info(f"📊 数据集信息:")
    logger.info(f"   测试集样本数: {len(y_true)}")
    logger.info(f"   正面样本数: {np.sum(y_true == 1)}")
    logger.info(f"   负面样本数: {np.sum(y_true == 0)}")
    logger.info("")
    
    logger.info(f"📈 核心指标:")
    logger.info(f"   准确率 (Accuracy): {accuracy:.4f}")
    logger.info(f"   精确率 (Precision): {precision:.4f}")
    logger.info(f"   召回率 (Recall): {recall:.4f}")
    logger.info(f"   F1分数: {f1:.4f}")
    logger.info("")
    
    logger.info(f"📋 详细分类报告:")
    logger.info(class_report)
    logger.info("")
    
    logger.info(f"🎭 混淆矩阵:")
    logger.info(f"   TN: {cm[0,0]} | FP: {cm[0,1]}")
    logger.info(f"   FN: {cm[1,0]} | TP: {cm[1,1]}")
    logger.info("")
    
    # 6. 业务场景分析
    logger.info(f"💼 业务场景分析:")
    negative_precision = cm[0,0] / (cm[0,0] + cm[1,0]) if (cm[0,0] + cm[1,0]) > 0 else 0
    positive_recall = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
    
    logger.info(f"   负面评论精确率: {negative_precision:.4f} (减少误判负面为正面)")
    logger.info(f"   正面评论召回率: {positive_recall:.4f} (不错过任何正面评价)")
    logger.info("=" * 60)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_true': y_true,
        'y_pred': y_pred,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, title='混淆矩阵'):
    """可视化混淆矩阵"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['负面', '正面'], 
                yticklabels=['负面', '正面'])
    plt.title(title)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.show()

def benchmark_original_model(test_dataset):
    """
    测试初始 Bert-base-chinese 模型在测试集上的性能
    作为优化前的基准对比
    """
    logger.info("🧪 开始基准测试：原始 Bert-base-chinese 模型")
    
    # 1. 加载原始未微调的模型
    logger.info("加载原始 Bert-base-chinese 模型...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-chinese", 
        num_labels=2
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 2. 准备测试数据
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=False, max_length=512)
    
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    # 3. 预测
    logger.info("在测试集上进行预测...")
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(len(tokenized_test)):
            sample = tokenized_test[i]
            input_ids = torch.tensor(sample['input_ids']).unsqueeze(0).to(device)
            attention_mask = torch.tensor(sample['attention_mask']).unsqueeze(0).to(device)
            label = sample['label']
            
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_label = torch.argmax(predictions, dim=1).item()
            
            all_predictions.append(predicted_label)
            all_labels.append(label)
            
            if (i + 1) % 100 == 0:
                logger.info(f"已处理 {i+1}/{len(tokenized_test)} 个样本")
    
    # 4. 计算指标
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    cm = confusion_matrix(y_true, y_pred)
    
    # 5. 输出基准结果
    logger.info("📊 原始模型基准性能")
    logger.info("=" * 50)
    logger.info(f"准确率 (Accuracy): {accuracy:.4f}")
    logger.info(f"精确率 (Precision): {precision:.4f}")
    logger.info(f"召回率 (Recall): {recall:.4f}")
    logger.info(f"F1分数: {f1:.4f}")
    logger.info("")
    logger.info(f"混淆矩阵:")
    logger.info(f"TN: {cm[0,0]} | FP: {cm[0,1]}")
    logger.info(f"FN: {cm[1,0]} | TP: {cm[1,1]}")
    logger.info("=" * 50)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred
    }

def compare_models(original_results, optimized_results):
    """
    对比优化前后的模型性能
    """
    logger.info("🔄 模型优化效果对比")
    logger.info("=" * 60)
    
    # 性能提升计算
    acc_improvement = optimized_results['accuracy'] - original_results['accuracy']
    f1_improvement = optimized_results['f1'] - original_results['f1']
    
    # 性能提升百分比
    acc_improvement_pct = (acc_improvement / original_results['accuracy']) * 100
    f1_improvement_pct = (f1_improvement / original_results['f1']) * 100
    
    logger.info("📈 性能对比表:")
    logger.info("指标         | 原始模型 | 优化模型 | 提升值   | 提升百分比")
    logger.info("-" * 60)
    logger.info(f"准确率      | {original_results['accuracy']:.4f} | {optimized_results['accuracy']:.4f} | +{acc_improvement:.4f} | +{acc_improvement_pct:.1f}%")
    logger.info(f"F1分数      | {original_results['f1']:.4f} | {optimized_results['f1']:.4f} | +{f1_improvement:.4f} | +{f1_improvement_pct:.1f}%")
    logger.info(f"精确率      | {original_results['precision']:.4f} | {optimized_results['precision']:.4f}")
    logger.info(f"召回率      | {original_results['recall']:.4f} | {optimized_results['recall']:.4f}")
    logger.info("")
    
    # 业务指标对比
    orig_cm = original_results['confusion_matrix']
    opt_cm = optimized_results['confusion_matrix']
    
    orig_negative_precision = orig_cm[0,0] / (orig_cm[0,0] + orig_cm[1,0]) if (orig_cm[0,0] + orig_cm[1,0]) > 0 else 0
    opt_negative_precision = opt_cm[0,0] / (opt_cm[0,0] + opt_cm[1,0]) if (opt_cm[0,0] + opt_cm[1,0]) > 0 else 0
    
    logger.info("💼 业务指标对比:")
    logger.info(f"负面评论精确率: {orig_negative_precision:.4f} → {opt_negative_precision:.4f}")
    logger.info("=" * 60)
    
    return {
        'accuracy_improvement': acc_improvement,
        'f1_improvement': f1_improvement,
        'improvement_percentage': {
            'accuracy': acc_improvement_pct,
            'f1': f1_improvement_pct
        }
    }

def main():
    """主函数"""
    # 1. 加载清洗后的数据
    # 假设你的数据已经保存为Hugging Face数据集格式
    try:
        # 如果你用之前建议的 save_to_disk 保存了数据
        dataset = load_from_disk('/home/zhp//bert/ChnSentiCorp_cleaned')
    except:
        # 如果数据是其他格式，请修改这里的加载方式
        raise ValueError("请确保数据路径正确，且数据已清洗并可用")
    
    # 检查数据集结构
    logger.info("Dataset structure:")
    for split in ['train', 'validation', 'test']:
        if split in dataset:
            logger.info(f"{split}: {len(dataset[split])} samples")
    
    # 2. 初始化分析器
    analyzer = SentimentAnalyzer()
    
    # 3. 开始训练
    trainer = analyzer.train(
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        output_dir="/home/zhp//bert/my-product-sentiment-model"
    )
    
    # 4. 在测试集上评估最终模型
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(dataset['test'].map(analyzer.tokenize_function, batched=True))
    logger.info(f"Test results: {test_results}")
    
    logger.info("开始模型性能评估...")
    evaluation_results = evaluate_model(
        test_dataset=dataset['test'], 
        analyzer=analyzer,
        model_path="/home/zhp/bert/my-product-sentiment-model/final_model"
    )
        
    # 4. 可选：可视化混淆矩阵
    plot_confusion_matrix(evaluation_results['confusion_matrix'])

    # 5. 示例预测
    logger.info("Example predictions:")
    test_texts = [
        "物流速度很快，包装完好，商品质量不错",
        "客服态度很差，退货流程复杂，不推荐购买",
        "性价比一般，不推荐购买"
    ]
    
    for text in test_texts:
        result = analyzer.predict(text,model_path="/home/zhp//bert/my-product-sentiment-model/final_model")
        logger.info(f"文本: {text}")
        logger.info(f"情感: {result['sentiment']} (置信度: {result['confidence']:.4f})")
        logger.info("---")

    # 2. 测试原始模型性能（基准）
    logger.info("步骤1: 测试原始 Bert-base-chinese 模型")
    original_results = benchmark_original_model(dataset['test'])
    
    # 3. 测试优化后模型性能
    logger.info("步骤2: 测试优化后的模型")
    analyzer = SentimentAnalyzer()
    optimized_results = evaluate_model(
        test_dataset=dataset['test'], 
        analyzer=analyzer,
        model_path="/home/zhp/bert/my-product-sentiment-model/final_model"
    )
    
    # 4. 性能对比分析
    logger.info("步骤3: 性能对比分析")
    comparison = compare_models(original_results, optimized_results)
    
    # 5. 结论
    logger.info("🎯 优化效果总结")
    if comparison['accuracy_improvement'] > 0.05:  # 提升超过5%
        logger.info("🏆 优化效果：非常显著！")
    elif comparison['accuracy_improvement'] > 0.02:  # 提升超过2%
        logger.info("✅ 优化效果：明显提升！")
    else:
        logger.info("⚠️ 优化效果：有待改进")

    

if __name__ == "__main__":
    main()