## text match 
文本匹配

-- 文本检索 （ad-hoc)  
-- 释义识别（paraphrase indentification）  
-- 自然语言推荐 NLI  
-- 问答匹配（QA）  

传统文本匹配算法
-- 构建特征
    -- tf-idf
    -- bm25
    -- 词法
-- 训练
    -- lr\svm等机器学习算法训练

根据文本长度匹配  
1）短文本匹配短文本 
同款商品搜索，计算用户query和网页标题的匹配
2）短文本匹配长文本  
搜索引擎、智能问答、知识检索
3）长文本匹配长文本
新闻推荐，抽象整个长文本内容信息，来识别同领域或同事件的相似文本


# 数据集实验结果  
| 序号 | 数据集 |  模型 | acc |   
| --- | ---   | ---  |   ---   | 
| 1 | paws-x-zh | tf+cosine | 0.573 |
| 2 | paws-x-zh | bm25 | 0.5582 |
| 3 | paws-x-zh | bert | |


