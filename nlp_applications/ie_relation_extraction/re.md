# 实体关系抽取
任务目标是从文本中抽取三元组信息，即（subject, predicate, object）。英文名称为relation extract 或 relation detect

# 方法
1）基于模板的方法，人工构建正则或者句法的规则。 优点是precision 高，在特定领域效果会很好，缺点是recall会比较低，无法移植到到其他领域
2）基于统计学的方法，将任务分为两部分，分别是实体抽取和关系分类。该方法需要人工构建特征。  
3）基于深度学习的方法， 这是当前的主流方法，不需要人工构建特征
    -- pipeline 方法  该方法将任务分为两部分进行，存在错误累积问题。但是有一篇文章用pipeline 方法取得state of art 效果
    -- joint 方法  该方法让两部分任务共享一部分参数，理论上比pipeline方法好  

# 方法对比
1）规则方法  
2）spert  
3）multi_head  
4）pointer net  
5）tplink  https://github.com/131250208/TPlinker-joint-extraction
6）novel_tagging [5]
7) PURE [6]  https://github.com/princeton-nlp/PURE
8) PRGC[7] 

数据集 DuIE2.0 

| 方法 | recall | precision | f1-value |  
| --- | ------ | --------- | -------- |  
| 基于规则方法 | 0 | 0 | 0 |
| spert | 0.3721 | 0.0322 | 0.0592 |
| multi_head | 0.3919 | 0.2373 | 0.2956 |
| pointer-network[3] | 0.2199 | 0.4392 | 0.2931 |
| tplink[4] | 0.6644 | 0.4880 | 0.5627 |
| etl_span(pytorch) | 0.5415 | 0.6594 | 0.5947 |


# 相关文章
[1] span-based Joint Entity and Relation Extraction with Transformer Pre-training  
[2] Joint entity recognition and relation extraction as a multi-head selection problem
[3] Joint Extraction of Entities and Relations Based on a Novel Decomposition Strategy 
[4] Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking
[5] Joint Extraction of Entities and Relations Based on a Novel Tagging Scheme
[6] Joint Entity and Relation Extraction with Table-Sequence Encoders
[7] Two are Better than One:Joint Entity and Relation Extraction with Table-Sequence Encoders
[8] PRGC: Potential Relation and Global Correspondence Based Joint Relational Triple Extraction
