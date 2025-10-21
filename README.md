**github上面的文件结构说明：**

**ChnSentiCorp文件是原始数据集**

**ChnSentiCorp_cleaned文件是清洗后的数据集**

**bert-model是未训练的模型**

**final_model是训练后的模型**

**test1.py主要是初次使用bert模型，熟悉一下相关函数用法**

**test2.py主要是实践bert模型进行情感评价分类**

**test3.py主要是用来查看数据集（train,test,valid）的数据大小和形式**

**data_wash.py主要是一些数据清洗的函数**

**data_enhence.py主要是一些数据增强的函数**

***data_deal.py主要是对数据集进行清洗增强操作，得到新的数据集***

**train_bert_sentiment.py是主要的模型训练代码，包含模型训练和模型效果评估和对比**

本项目基于Bert-base-chinese 模型结构,使用ChnSentiCorp数据集，通过修改模型结构以适应商品评价分类任务，后续有时间会继续完善模型，以求达到最好的效果
