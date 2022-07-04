#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    @Time : 2022/7/4 13:24
    @Author  : jack.li
    @Site    : 
    @File    : bert_classifier.py

"""
import torch
import argparse
import torch.optim as optim
import torch.nn as nn
from functools import partial
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizerFast

"""
    这是一个基于bert的多分类模型，对于二分类、多标签分类，模型比较类似，修改输出结果以及损失函数就可以了
    这里演示了如何训练和预测
"""


def get_train_dataset():
    import numpy as np
    import pandas as pd
    path = "D:\\xxxx\\bert_classifier_sample.csv"
    df = pd.read_csv(path)
    label_list = [0, 1, 2, 3, 4]
    train_list = []
    for _, row in df.iterrows():
        train_list.append({
            "text": row["sentence"],
            "label": np.random.choice(label_list)  # 这里的标签是随机生成的，目的还是为了演示
        })

    return train_list


class ClassifierDataset(Dataset):

    def __init__(self, document_list, tokenizer):
        super(ClassifierDataset, self).__init__()
        self.document_list = document_list
        self.tokenizer = tokenizer

    def __getitem__(self, item):
        """
            Args:
                item: int, idx
        """
        document = self.document_list[item]
        return document

    def __len__(self):
        return len(self.document_list)

    def _create_collate_fn(self, batch_first=False):
        def collate(documents):
            batch_input_ids, batch_attention_mask, batch_token_type_id = [], [], []
            batch_input_labels = []

            for document in documents:
                text_word = document["text"]
                codes = self.tokenizer.encode_plus(text_word,
                                                   return_offsets_mapping=True,
                                                   truncation=True,
                                                   return_length=True,
                                                   padding="max_length")

                input_ids_ = codes["input_ids"]
                # print(text_word)

                # print(codes["offset_mapping"])
                input_ids = torch.tensor(input_ids_).long()
                attention_mask = torch.tensor(codes["attention_mask"]).long()
                token_type_ids = torch.tensor(codes["token_type_ids"]).long()
                # offset_mapping = codes["offset_mapping"]

                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_token_type_id.append(token_type_ids)
                label_id = [document["label"]]
                batch_input_labels.append(label_id)
                # batch_gold_answer.append(document[2])
            batch_input_labels = torch.LongTensor(batch_input_labels)
            batch_input_ids = torch.stack(batch_input_ids, dim=0)
            batch_attention_mask = torch.stack(batch_attention_mask, dim=0).bool()
            batch_token_type_id = torch.stack(batch_token_type_id, dim=0)


            return {
                "batch_input_ids": batch_input_ids,
                "batch_attention_mask": batch_attention_mask,
                "batch_token_type_id": batch_token_type_id,
                "batch_input_labels": batch_input_labels,
            }

        return partial(collate)

    def get_dataloader(self, batch_size, num_workers=0, shuffle=False, batch_first=True, pin_memory=False,
                       drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(batch_first),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)


class BertClassifier(nn.Module):

    def __init__(self, config):
        super(BertClassifier, self).__init__()
        self.bert_encoder = BertModel.from_pretrained(config.pretrain_name)

        self.classifier = nn.Linear(config.hidden_size, config.class_num)

    def forward(self, input_id, input_segment_id, input_mask,):
        outputs = self.bert_encoder(input_id, token_type_ids=input_segment_id, attention_mask=input_mask)
        outputs = outputs[0][:,0]  # 分类任务 使用bert最后一层第一个[cls]字符
        classifier_logits = nn.Softmax(dim=-1)(self.classifier(outputs))

        return classifier_logits


# 训练 模型
def train():
    # 根据gpu 情况选择是否使用gpu
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--pretrain_name", type=str, default="bert-base-chinese", required=False)
    parser.add_argument("--batch_size", type=int, default=4, required=False)
    parser.add_argument("--epoch", type=int, default=30, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    parser.add_argument('--hidden_size', type=int, default=768, required=False)
    parser.add_argument('--class_num', type=int, default=5, required=False)
    parser.add_argument('--device', type=str, default=device, required=False)

    config = parser.parse_args()

    train_data_lists = get_train_dataset()

    tokenizer = BertTokenizerFast.from_pretrained(config.pretrain_name)

    dataset = ClassifierDataset(train_data_lists, tokenizer)
    train_data_loader = dataset.get_dataloader(config.batch_size,
                                               shuffle=config.shuffle,
                                               pin_memory=config.pin_memory)

    model = BertClassifier(config)
    model.to(config.device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adamax(parameters)

    # 交叉熵损失函数
    loss_func = nn.CrossEntropyLoss()

    # 训练模型
    for epoch in range(config.epoch):
        model.train()
        for idx, batch_data in enumerate(train_data_loader):
            classifer_logits = model(batch_data["batch_input_ids"].to(device),
                                  batch_data["batch_token_type_id"].to(device),
                                  batch_data["batch_attention_mask"].to(device))
            loss = loss_func(classifer_logits.view(-1, config.class_num), batch_data["batch_input_labels"].to(device).view(-1))
            print("epoch {0} batch {1} loss value is {2}".format(epoch, idx, loss.data.cpu().numpy()))

            loss.backward()
            optimizer.step()
            model.zero_grad()

            predict("闪捷信息（Secsmart）是一家专注数据安全的高新技术企业", model)

        # 保存模型
        torch.save(model.state_dict(), "bert_classifier.pt")


# 预测/predict/inference
def predict(input_str, model=None):
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--pretrain_name", type=str, default="bert-base-chinese", required=False)
    parser.add_argument("--batch_size", type=int, default=16, required=False)
    parser.add_argument("--epoch", type=int, default=30, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    parser.add_argument('--hidden_size', type=int, default=768, required=False)
    parser.add_argument('--class_num', type=int, default=2, required=False)
    parser.add_argument('--warmup_proportion', type=float, default=0.9, required=False)
    parser.add_argument('--device', type=str, default=device, required=False)

    config = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained(config.pretrain_name)

    batch_input_ids, batch_attention_mask, batch_token_type_id = [], [], []

    # 如果输入文本长于512，需要进行切割程,这里就不展示如何切割了
    text_cut = [input_str]

    for text in text_cut:
        codes = tokenizer.encode_plus(text,
                                       return_offsets_mapping=True,
                                       truncation=True,
                                       return_length=True,
                                       padding="max_length")
        input_ids_ = codes["input_ids"]

        # print(codes["offset_mapping"])
        input_ids = torch.tensor(input_ids_).long()
        attention_mask = torch.tensor(codes["attention_mask"]).long()
        token_type_ids = torch.tensor(codes["token_type_ids"]).long()

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_token_type_id.append(token_type_ids)
    batch_input_ids = torch.stack(batch_input_ids, dim=0)
    batch_token_type_id = torch.stack(batch_token_type_id, dim=0)
    batch_attention_mask = torch.stack(batch_attention_mask, dim=0).bool()

    # 如果模型为空，可以加载模型
    if model is None:
        model = torch.load("bert_classifier.pt")
        model.to(config.device)


    classifier_logits = model(batch_input_ids, batch_token_type_id, batch_attention_mask)

    # 取softmax 中最大值的索引作为输出标签
    classifier_label = torch.argmax(classifier_logits, dim=-1)

    # 这里将torch 转成numpy
    classifier_label = classifier_label.cpu().detach().numpy()

    print("ans: ", classifier_label)

    # 最后根据索引位置输出最终标签，
    # 例如标签编码如下 {“label1”: 0, "label2“: 1, "label3": 2}， 那么索引0就指的是”label1“, 依此类推


if __name__ == "__main__":

    # data  = get_train_dataset()
    # print(data[0])

    train()