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
8) globalPointer[4]
9) mrc model [5]

msr data

| 方法 | recall | precision | f1-value |
| ---- | -------| ---------| -------- |
| hmm | 0       |  0       |     0    |
| crf  | 0.7202 |  0.8038  |      0.7597   |
| bi-lstm | 0.6139 | 0.7107 |    0    |
| bi-lstm+crf | 0.6960 | 0.8118 |  0  |
| bi-lstm+crf+word | 0.7643 | 0.8052 |  0.7842 |
| seq2seq | 0.0370 | 0.1855 | 0.0617 |
| pointer_net | 0.5948 | 0.8287 |  0.6926 | 
| 

## 相关提高方法
1) word-level feature
2) cascade 即第一步先分实体，第二步对实体类别进行分类

相关资料
1) https://zhuanlan.zhihu.com/p/166496466   
2) https://zhuanlan.zhihu.com/p/181732910
3) https://zhuanlan.zhihu.com/p/350669317
4) https://spaces.ac.cn/archives/8373
5) A Unified MRC Framework for Named Entity Recognition
6) FLAT: Chinese NER Using Flat-Lattice Transformer
7) Lex-BERT: Enhancing BERT based NER with lexicons


