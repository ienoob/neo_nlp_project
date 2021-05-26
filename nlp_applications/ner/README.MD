# name entity regnization 


name entity regnization 的任务目的是从文本中找到指定的实体。实体可以分为人名、地名、组织名、时间等等。ner任务是信息抽取（information extraction）的首要目标。

ner的相关方法

bio 序列标注模型  
1) hmm   
2) crf   
3) rnn/lstm/gru + crf   
4) seq2seq + attention + crf  
5) transformer + crf  
6) bert + crf  
7) pointer_network  

| 方法 | recall | precision |
| ---- | -------| ---------|
| crf  | 0.7185 |  0.8030  |
| bi-lstm | 0.6139 | 0.7107 |
| bi-lstm+crf | 0.6960 | 0.8118 |

## 相关提高方法
1) word-level feature
2) cascade 即第一步先分实体，第二步对实体类别进行分类

相关资料
1) https://zhuanlan.zhihu.com/p/166496466   
2) https://zhuanlan.zhihu.com/p/181732910
3) https://zhuanlan.zhihu.com/p/350669317



