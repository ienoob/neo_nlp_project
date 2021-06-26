# 事件抽取/检测

event extract or event detect
定义：从文本中抽取事件信息。事件信息包括事件的类别以及事件的一些属性。

事件抽取包含触发词识别，事件分类和事件要素识别。
抽取的东西主要包括：  
1）事件类型  event type
2）触发词    event trigger
3）事件描述   event mention
4）事件的要素（角色（role）和元素（argument or entity mention））
事件抽取方法分为pipeline 和joint 两种模式。

## 事件抽取相关论文
1）pipeline
- Event Extraction via Dynamic Multi-Pooling Convolutional Neural Networks
- RBPB: Regularization-based Pattern Balancing Method for Event Extraction
- DCFEE: A Document-level Chinese Financial Event Extraction System based on Automatically Labeled Training Data
- A Language-Independent Neural Network for Event Detection
- Exploiting Argument Information to Improve Event Detection via Supervised Attention Mechanisms
- Document Embedding Enhanced Event Detection with Hierarchical and Supervised Attention
- Self-regulation: Employing a Generative Adversarial Network to Improve Event Detection

2）联合抽取
- JRNN: Joint Event Extraction via Recurrent Neural Networks
- Joint Extraction of Events and Entities within a Document Context
- DBRNN: Jointly Extracting Event Triggers and Arguments by Dependency-Bridge RNN and Tensor-Based Argument Interaction
- Jointly Multiple Events Extraction via Attention-based Graph Information Aggregation
- DeepEventMine: end-to-end neural nested event extraction from biomedical texts

## 事件抽取的相关难点
1）一段文本中存在多个事件，如何将事件以及事件的要素进行匹配
2）文本很长，即如何解决篇章级别的事件抽取任务
    元素（argument）会很分散，一个事件中的元素会分布到多个句子中
    多种事件共存。一个文档中存在多个事件
3) 如何构建joint 模式的事件模型
4) 事件数据稀少，如何增量
5) 

实验结果

数据集
duee_v1.0  

| 模型 |  recall | precision | f1-measure | 
| --- | --- | --- | --- |
| DeepEventMine | 0.0 | 0.0 | 0.0 |
| Pipeline | 0.0274 | 0.0298 | 0.0285 |




    
    




