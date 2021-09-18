#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import os
import numpy as np
import torch
import argparse
from torch import nn
from torch.nn import functional as F
from transformers import BertModel, BertTokenizer, BertTokenizerFast
# from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset, DataLoader
from nlp_applications.data_loader import LoaderDuReaderChecklist

bert_model_name = "bert-base-chinese"

path = "D:\data\阅读理解\\2021语言与智能技术竞赛：机器阅读理解任务"
data_loader = LoaderDuReaderChecklist(path)

torch.set_num_threads(1)
# bert_path = "D:\data\\bert\\bert-base-chinese"
# vocab_file = os.path.join(bert_path, "vocab.txt")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
# tokenizer = BertTokenizer.from_pretrained(bert_model_name)

qa_list = []
for document in data_loader.documents:
    if len(document.context) > 400:
        continue
    for qa in document.qa_list:
        qa_list.append((document.context, qa, document.id))

qa_dev_list = []
for document in data_loader.dev_documents:
    if len(document.context) > 400:
        continue
    for qa in document.qa_list:
        qa_dev_list.append((document.context, qa, document.id))


class DuReaderDataset(Dataset):
    def __init__(self, qa_list, tokenizer):
        super(DuReaderDataset, self).__init__()
        self.qa_list = qa_list
        self.tokenizer = tokenizer

    def __getitem__(self, item):
        """
            Args:
                item: int, idx
        """
        document = self.qa_list[item]
        return document

    def __len__(self):
        return len(self.qa_list)


def sequence_padding(inputs, length=None, padding=0, is_float=False):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    outputs = np.array([
        np.concatenate([x, [padding] * (length - len(x))])
        if len(x) < length else x[:length] for x in inputs
    ])

    out_tensor = torch.FloatTensor(outputs) if is_float \
        else torch.LongTensor(outputs)
    return out_tensor.clone().detach()

def collate_to_max_length(batch):
    batch_ids = []
    batch_masks = []
    batch_start = []
    batch_end = []
    batch_span = []
    batch_type_ids = []
    batch_loss_mask = []
    batch_answers = []
    batch_middles = []
    batch_question = []
    batch_context = []
    # if len(batch) != 2:
    #     print(batch)
    max_len = 0
    for data in batch:
        # data = list(data)
        context, qa, doc_id = data
        query_context_tokens = tokenizer.encode_plus(qa.q, context, add_special_tokens=True, return_offsets_mapping=True)

        max_len = max(len(query_context_tokens["input_ids"]), max_len)

    for data in batch:
        # print(len(data), "++++++++++++")
        # if len(list(data)) != 2:
        #     print(data)
        # print(data)
        context, qa, doc_id = data
        answer = []

        query_context_tokens = tokenizer.encode_plus(qa.q, context, add_special_tokens=True, return_offsets_mapping=True)
        tokens = query_context_tokens["input_ids"]
        type_ids = query_context_tokens["token_type_ids"]
        attention_mask = query_context_tokens["attention_mask"]
        offset_mapping = query_context_tokens["offset_mapping"]

        start_label = np.zeros(len(tokens))
        end_label = np.zeros(len(tokens))
        # span_label = np.zeros(len(tokens))
        span_label = np.zeros([max_len, max_len], dtype=np.int32)
        loss_mask = np.zeros(len(tokens))

        batch_ids.append(tokens)
        batch_masks.append(type_ids)
        batch_type_ids.append(attention_mask)
        batch_context.append(context)

        batch_question.append(qa.q)

        if qa.start != -1:
            answer.append((qa.start, qa.a))
            qa_start = qa.start
            qa_end = qa.start + len(qa.a)
            if doc_id == 2844 and qa.q == "鲁滨逊漂流记作者":
                qa_start += 1
            if doc_id == 2712 and qa.q == "嵊怎么读":
                qa_start += 1

            faker_start = -1
            faker_end = -1
            # print(doc_id, qa.q, [qa.a])
            for iv, off in enumerate(offset_mapping):
                if off == (0, 0):
                    continue
                if qa_start == off[0]:
                    faker_start = iv
                if qa_end == off[1]:
                    faker_end = iv
            # loss_mask[middle[1] + 1:] = 1
            start_label[faker_start] = 1
            end_label[faker_end] = 1
            span_label[faker_start, faker_end] = 1
        middle = []
        for iv, off in enumerate(offset_mapping):
            if off == (0, 0):
                middle.append(iv)
        assert len(middle) == 3
        loss_mask[middle[1] + 1:] = 1

        batch_middles.append(middle[1])
        batch_start.append(start_label)
        batch_end.append(end_label)
        # print(span_label.shape)
        batch_span.append(span_label)
        batch_loss_mask.append(loss_mask)
        batch_answers.append(answer)

    batch_ids = sequence_padding(batch_ids)
    batch_masks = sequence_padding(batch_masks)
    batch_type_ids = sequence_padding(batch_type_ids)
    batch_start = sequence_padding(batch_start)
    batch_end = sequence_padding(batch_end)
    batch_span = torch.LongTensor(batch_span)
    batch_loss_mask = sequence_padding(batch_loss_mask)

    return {
        "batch_ids": batch_ids,
        "batch_type_ids": batch_type_ids,
        "batch_masks": batch_masks,
        "batch_start": batch_start,
        "batch_end": batch_end,
        "batch_span": batch_span,
        "batch_loss_mask": batch_loss_mask,
        "batch_answers": batch_answers,
        "batch_middles": batch_middles,
        "batch_question": batch_question,
        "batch_context": batch_context
    }


def kmp(mom_string, son_string):
    # 传入一个母串和一个子串
    # 返回子串匹配上的第一个位置，若没有匹配上返回-1
    test = ''
    # if type(mom_string) != type(test) or type(son_string) != type(test):
    #     return -1
    if len(son_string) == 0:
        return 0
    if len(mom_string) == 0:
        return -1
    # 求next数组
    next = [-1] * len(son_string)
    if len(son_string) > 1:  # 这里加if是怕列表越界
        next[1] = 0
        i, j = 1, 0
        while i < len(son_string) - 1:  # 这里一定要-1，不然会像例子中出现next[8]会越界的
            if j == -1 or son_string[i] == son_string[j]:
                i += 1
                j += 1
                next[i] = j
            else:
                j = next[j]

    # kmp框架
    m = s = 0  # 母指针和子指针初始化为0
    while (s < len(son_string) and m < len(mom_string)):
        # 匹配成功,或者遍历完母串匹配失败退出
        if s == -1 or mom_string[m] == son_string[s]:
            m += 1
            s += 1
        else:
            s = next[s]

    if s == len(son_string):  # 匹配成功
        return m - s
    # 匹配失败
    return -1




# for document in data_loader.documents:
#     # print(len(document.context))
#     # if len(document.context) > 512:
#     #     continue
#     doc_id = document.id
#     for qa in document.qa_list:
#         query_context_tokens = tokenizer.encode_plus(qa.q, document.context,
#                                                      add_special_tokens=True,
#                                                      return_offsets_mapping=True)
#         # print(query_context_tokens)
#
#         tokens = query_context_tokens["input_ids"]
#         type_ids = query_context_tokens["token_type_ids"]
#         attention_mask = query_context_tokens["attention_mask"]
#         offset_mapping = query_context_tokens["offset_mapping"]
#         loss_mask = np.zeros(1)
#
#
#         if qa.start != -1:
#             qa_start = qa.start
#             qa_end = qa.start + len(qa.a)
#             if doc_id == 2844 and qa.q == "鲁滨逊漂流记作者":
#                 qa_start += 1
#             if doc_id == 2712 and qa.q == "嵊怎么读":
#                 qa_start += 1
#
#             faker_start = -1
#             faker_end = -1
#             middle = []
#             # print(doc_id, qa.q, [qa.a])
#             for iv, off in enumerate(offset_mapping):
#                 if off == (0, 0):
#                     middle.append(iv)
#                     continue
#                 if qa_start == off[0]:
#                     faker_start = iv
#                 if qa_end == off[1]:
#                     faker_end = iv
#             assert len(middle) == 3
#             loss_mask[middle[1]+1:] = 1
#             print(qa_start, qa_end, faker_start, faker_end)
#             print(offset_mapping[faker_start], offset_mapping[faker_end])



class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, num_label, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.classifier1 = nn.Linear(hidden_size, hidden_size)
        self.classifier2 = nn.Linear(hidden_size, num_label)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        # features_output1 = F.relu(features_output1)
        features_output1 = F.gelu(features_output1)
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2


class BertMRC(nn.Module):

    def __init__(self, config):
        super(BertMRC, self).__init__()

        self.bert_encoder = BertModel.from_pretrained(bert_model_name)
        self.start_indx = nn.Linear(config.hidden_size, 1)
        self.end_indx = nn.Linear(config.hidden_size, 1)
        self.span_embedding = MultiNonLinearClassifier(config.hidden_size * 2, 1, config.mrc_dropout)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        bert_outputs = self.bert_encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        sequence_heatmap = bert_outputs[0]  # [batch, seq_len, hidden]
        # print(sequence_heatmap.shape)
        batch_size, seq_len, hid_size = sequence_heatmap.size()

        start_logits = self.start_indx(sequence_heatmap).squeeze(-1)  # [batch, seq_len, 1]
        end_logits = self.end_indx(sequence_heatmap).squeeze(-1)  # [batch, seq_len, 1]

        # start_extend = sequence_heatmap.unsqueeze(2).expand(-1, -1, seq_len, -1)
        # # [batch, seq_len, seq_len, hidden]
        # end_extend = sequence_heatmap.unsqueeze(1).expand(-1, seq_len, -1, -1)
        # # [batch, seq_len, seq_len, hidden*2]
        # span_matrix = torch.cat([start_extend, end_extend], 3)
        # # [batch, seq_len, seq_len]
        # span_logits = self.span_embedding(span_matrix).squeeze(-1)

        return start_logits, end_logits, None


def eval_model(model, batch_data):
    model.eval()
    sigmoid = torch.nn.Sigmoid()
    start_logits, end_logits, span_logits = model(batch["batch_ids"], batch["batch_type_ids"], batch["batch_masks"])
    start_logits = sigmoid(start_logits).detach().numpy()
    end_logits = sigmoid(end_logits).detach().numpy()

    for b, start_logit in enumerate(start_logits):
        valid_len = batch_data["batch_middles"][b]
        context = batch_data["batch_context"][b]
        end_logit = end_logits[b]
        start_logit_max = np.argmax(start_logit[valid_len:])
        end_logit_max = np.argmax(end_logit[valid_len:])
        print(context)

        que = batch_data["batch_question"][b]
        print("question ", que)

        ans = batch_data["batch_answers"][b]
        print(ans)

        pred = []
        if start_logit[start_logit_max+valid_len] > 0.5 and end_logit[end_logit_max+valid_len] > 0.5:
            pred.append((start_logit_max, end_logit_max))
        print(pred)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--pretrain_name", type=str, default="hfl/chinese-roberta-wwm-ext", required=False)
    parser.add_argument("--batch_size", type=int, default=3, required=False)
    parser.add_argument("--epoch", type=int, default=10, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    parser.add_argument('--hidden_size', type=int, default=768, required=False)
    parser.add_argument('--mrc_dropout', type=float, default=0.5, required=False)
    config = parser.parse_args()

    dataset = DuReaderDataset(qa_list, tokenizer)
    dev_dataset = DuReaderDataset(qa_dev_list, tokenizer)
    train_data_loader = DataLoader(dataset,  batch_size=config.batch_size, num_workers=1,
                            collate_fn=collate_to_max_length, pin_memory=config.pin_memory)
    dev_data_loader = DataLoader(dev_dataset,  batch_size=config.batch_size, num_workers=1,
                            collate_fn=collate_to_max_length, pin_memory=config.pin_memory)

    model = BertMRC(config)
    bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    optimizer = torch.optim.Adamax(model.parameters(), lr=0.0005)
    for epoch in range(config.epoch):
        for step, batch in enumerate(train_data_loader):
            model.train()
            # print(batch["batch_ids"].shape)
            start_logits, end_logits, span_logits = model(batch["batch_ids"], batch["batch_type_ids"], batch["batch_masks"])
            batch_masks = batch["batch_masks"]
            batch_start = batch["batch_start"]
            batch_end = batch["batch_end"]
            batch_span = batch["batch_span"]
            batch_loss_mask = batch["batch_loss_mask"]
            # b, seq_len, h = start_logits.shape

            label_mask = batch_loss_mask.view(-1).float()
            start_loss = bce_loss(start_logits.view(-1), batch_start.view(-1).float())
            start_loss = (start_loss * label_mask).sum() / label_mask.sum()

            end_loss = bce_loss(end_logits.view(-1), batch_end.view(-1).float())
            end_loss = (end_loss * label_mask).sum() / label_mask.sum()

            # print(label_mask.sum())
            # print(end_loss)

            # match_label_mask = (batch_loss_mask.unsqueeze(-1).expand(-1, -1, seq_len)
            #                     & batch_loss_mask.unsqueeze(1).expand(-1, seq_len, -1))
            # match_label_mask = torch.triu(match_label_mask, 0)
            # # print(match_label_mask.shape)
            # # print(span_logits.shape)
            # # print(batch_span.shape)
            # match_label_mask = match_label_mask.view(config.batch_size, -1).float()
            # match_loss = bce_loss(span_logits.view(config.batch_size, -1), batch_span.view(config.batch_size, -1).float())
            # match_loss = (match_loss * match_label_mask).sum() / match_label_mask.sum()

            loss = start_loss + end_loss

            # print(loss)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if step % 20 == 0:
                print(
                    u"step {} / {} of epoch {}, train/loss: {}".format(step, len(train_data_loader),
                                                                       0, loss))
        model.eval()
        for batch_dev in dev_data_loader:
            eval_model(model, batch_dev)
    #
    #     break






