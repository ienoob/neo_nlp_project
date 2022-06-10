#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import json
import torch
import argparse
import torch.nn as nn
from pytorch.layers.crf import CRF
from functools import partial
# from pytorch.event_extraction.event_case1.train_data_v2 import rt_data
from pytorch.layers.bert_optimization import BertAdam
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, BertTokenizerFast
from nlp_applications.ner.evaluation import extract_entity, eval_metrix_v3




class EventModelV1(nn.Module):

    def __init__(self, config):
        super(EventModelV1, self).__init__()
        self.bert_encoder = BertModel.from_pretrained(config.pretrain_name)
        # print(type(self.bert_encoder))

        self.hidden2tag = nn.Linear(config.hidden_size, config.entity_size + 2)
        # self.drop = nn.Dropout(p=config.dropout)

        self.crf = CRF(config.entity_size, config.gpu)

        # self.register_parameter("weight_0_bert", self.bert_encoder.weight)
        # self.register_parameter("weight_0_crf", self.self.crf.weight)


    def forward(self, input_id, input_segment_id, input_mask, batch_label=None):
        outputs = self.bert_encoder(input_id, token_type_ids=input_segment_id, attention_mask=input_mask)
        outputs = outputs[0]
        tags = self.hidden2tag(outputs)
        # tags = self.drop(tags)

        if batch_label is None:
            scores, tag_seq = self.crf._viterbi_decode(tags, input_mask)
            return tag_seq
        else:
            total_loss = self.crf.neg_log_likelihood_loss(tags, input_mask, batch_label)
            scores, tag_seq = self.crf._viterbi_decode(tags, input_mask)
            return total_loss, tag_seq


# max_len = 301


class DUEEFI(Dataset):

    def __init__(self, document_list, tokenizer):
        super(DUEEFI, self).__init__()
        self.qa_list = document_list
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

    def _create_collate_fn(self, batch_first=False):
        def collate(documents):
            batch_input_ids, batch_attention_mask, batch_token_type_id = [], [], []
            batch_input_labels = []
            batch_gold_answer = []

            local_max = 0
            for document in documents:
                text = document["text"]
                local_max = max(len(text), local_max)
            local_max += 2

            for document in documents:
                text = document["text"]
                text_word = [t for t in list(text)]
                codes = self.tokenizer.encode_plus(text_word,
                                                   return_offsets_mapping=True,
                                                   is_split_into_words=True,
                                                   max_length=local_max,
                                                   truncation=True,
                                                   return_length=True,
                                                   padding="max_length")

                input_ids_ = codes["input_ids"]
                # print(text_word)

                # print(codes["offset_mapping"])
                iiv = len(input_ids_)-1
                while input_ids_[iiv]==0:
                    iiv -= 1
                # print(iiv + 1, len(document["label"]), len(text_word), len(input_ids_), local_max)
                # print(input_ids_)
                assert iiv+1 == len(document["label"])+2
                input_ids = torch.tensor(input_ids_).long()
                attention_mask = torch.tensor(codes["attention_mask"]).long()
                token_type_ids = torch.tensor(codes["token_type_ids"]).long()
                # offset_mapping = codes["offset_mapping"]

                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_token_type_id.append(token_type_ids)
                label_id = [0] + document["label"]
                for _ in range(local_max-len(label_id)):
                    label_id.append(0)
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
                "batch_gold_answer": batch_gold_answer
            }

        return partial(collate)

    def get_dataloader(self, batch_size, num_workers=0, shuffle=False, batch_first=True, pin_memory=False,
                       drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(batch_first),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)


def evaluation(model, data_iterator, id2label, id2role, config):
    model.eval()
    hit_num = 0.0
    true_num = 0.0
    pre_num = 0.0

    role_indicate = dict()

    for idx, batch_data in enumerate(data_iterator):
        tag_seqs = model(batch_data["batch_input_ids"].to(config.device),
                              batch_data["batch_token_type_id"].to(config.device),
                              batch_data["batch_attention_mask"].to(config.device))

        tag_seqs = tag_seqs * batch_data["batch_attention_mask"].to(config.device)
        tag_seqs = tag_seqs.cpu().numpy()

        # print(tag_seqs.shape)
        label_seq = batch_data["batch_input_labels"].numpy()
        batch_num_value = label_seq.shape[0]

        for b in range(batch_num_value):
            tag_seq_list = tag_seqs[b]
            tag_seq_list = [id2label.get(tag, "O") for tag in tag_seq_list]
            # print(tag_seq_list)

            true_seq_list = label_seq[b]
            true_seq_list = [id2label.get(tag, "O") for tag in true_seq_list]

            pre_value = extract_entity(tag_seq_list)
            true_value = extract_entity(true_seq_list)

            for e in true_value:
                role_indicate.setdefault(e[2], {"pred": 0, "real": 0, "hit": 0})
                role_indicate[e[2]]["real"] += 1

            for e in pre_value:
                role_indicate.setdefault(e[2], {"pred": 0, "real": 0, "hit": 0})
                role_indicate[e[2]]["pred"] += 1
            # print(true_value)

            pre_num += len(pre_value)
            true_num += len(true_value)

            for p in pre_value:
                if p in true_value:
                    hit_num += 1
                    role_indicate[p[2]]["hit"] += 1
        print(hit_num, true_num, pre_num)
    for role_id, role_ind in role_indicate.items():
        print("{} : {}".format(id2role[int(role_id)], eval_metrix_v3(role_ind["hit"], role_ind["real"], role_ind["pred"])))
    metric = eval_metrix_v3(hit_num, true_num, pre_num)
    return {
        "hit_num": hit_num,
        "true_num": true_num,
        "pre_num": pre_num,
        "recall": metric["recall"],
        "precision": metric["precision"],
        "f1_value": metric["f1_value"]
    }


def train_model(rt_data, model_name=None, train_model="eval"):
    device = "cpu"
    if torch.cuda.is_available():
        device = "gpu"

    if model_name is None:
        import datetime
        model_name = "{}".format(datetime.datetime.now())
    model_name = "{}-{}".format(train_model, model_name)
    if train_model == "eval":
        bert_train_list = rt_data["bert_data"]["train"]
        bert_dev_list = rt_data["bert_data"]["dev"]
    else:
        bert_train_list = rt_data["bert_data"]["all"]
        bert_dev_list = rt_data["bert_data"]["all"]
    role2id = rt_data["role2id"]
    label2id = rt_data["label2id"]

    train_data_lists = bert_train_list
    dev_data_lists = bert_dev_list
    id2label = {v: k for k, v in label2id.items()}
    id2role = {v: k for k, v in role2id.items()}
    print(id2label)

    print(role2id)

    # print("max len {}".format(max_len))

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--pretrain_name", type=str, default="hfl/chinese-roberta-wwm-ext", required=False)
    parser.add_argument("--batch_size", type=int, default=16, required=False)
    parser.add_argument("--epoch", type=int, default=30, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    parser.add_argument('--hidden_size', type=int, default=768, required=False)
    parser.add_argument('--entity_size', type=int, default=len(label2id), required=False)
    parser.add_argument('--warmup_proportion', type=float, default=0.9, required=False)
    parser.add_argument('--dropout', type=float, default=0.5, required=False)
    parser.add_argument('--device', type=str, default=device, required=False)
    parser.add_argument('--gpu', type=bool, default=False, required=False)
    config = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained(config.pretrain_name)

    dataset = DUEEFI(train_data_lists, tokenizer)
    dev_dataset = DUEEFI(dev_data_lists, tokenizer)

    train_data_loader = dataset.get_dataloader(config.batch_size,
                                               shuffle=config.shuffle,
                                               pin_memory=config.pin_memory)
    dev_data_loader = dev_dataset.get_dataloader(config.batch_size,
                                                 shuffle=config.shuffle,
                                                 pin_memory=config.pin_memory)

    model = EventModelV1(config)
    model.to(config.device)

    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=config.warmup_proportion,
                         t_total=int(len(dataset) // config.batch_size + 1) * config.epoch)


    max_f1 = 0
    for epoch in range(config.epoch):
        model.train()
        for idx, batch_data in enumerate(train_data_loader):
            loss, tag_seq = model(batch_data["batch_input_ids"].to(device),
                                  batch_data["batch_token_type_id"].to(device),
                                  batch_data["batch_attention_mask"].to(device),
                                  batch_data["batch_input_labels"].to(device))
            loss_value = loss.data.cpu().numpy()
            print("epoch {0} batch {1} loss value is {2}".format(epoch, idx, loss_value))

            loss.backward()
            optimizer.step()
            model.zero_grad()
        eval = evaluation(model, dev_data_loader, id2label, id2role, config)
        print(eval)
        if eval["f1_value"] > max_f1:
            max_f1 = eval["f1_value"]
            torch.save(model.state_dict(), "{}.pt".format(model_name))


def save_load_model(rt_data):
    label2id = rt_data["label2id"]

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--pretrain_name", type=str, default="hfl/chinese-roberta-wwm-ext", required=False)
    parser.add_argument("--batch_size", type=int, default=16, required=False)
    parser.add_argument("--epoch", type=int, default=30, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    parser.add_argument('--hidden_size', type=int, default=768, required=False)
    parser.add_argument('--entity_size', type=int, default=len(label2id), required=False)
    parser.add_argument('--warmup_proportion', type=float, default=0.9, required=False)
    parser.add_argument('--dropout', type=float, default=0.5, required=False)
    parser.add_argument('--gpu', type=bool, default=False, required=False)
    config = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained(config.pretrain_name)
    model = EventModelV1(config)
    model_pt = torch.load("all-finance.pt")
    model.load_state_dict(model_pt)
    model.eval()


    test_text = "SNOW 51连获A及A+轮共亿元级融资 目前在上海开设12家店\n36氪 思绮? ?2021-04-22 10:25\n" \
                "核心提示：SNOW 51连获A及A+轮共亿元级融资，该品牌定位滑雪新生活方式平台，目前在上海开有12家门店，" \
                "本轮融资后将加快在上海和一线城市落地门店。\n36氪获悉，滑雪产业一站式服务平台SNOW 51完成亿元级A系列轮融资。"
    local_max = len(test_text)
    codes = tokenizer.encode_plus(test_text,
                                       return_offsets_mapping=True,
                                       max_length=local_max,
                                       truncation=True,
                                       return_length=True,
                                       padding="max_length")

    input_ids = torch.tensor(codes["input_ids"]).long()
    attention_mask = torch.tensor(codes["attention_mask"]).long()
    token_type_ids = torch.tensor(codes["token_type_ids"]).long()
    batch_input_ids, batch_attention_mask, batch_token_type_id = [], [], []
    batch_input_ids.append(input_ids)
    batch_attention_mask.append(attention_mask)
    batch_token_type_id.append(token_type_ids)

    batch_input_ids = torch.stack(batch_input_ids, dim=0)
    batch_attention_mask = torch.stack(batch_attention_mask, dim=0)
    batch_token_type_id = torch.stack(batch_token_type_id, dim=0)
    print(batch_input_ids.shape, batch_attention_mask.shape, batch_token_type_id.shape)
    batch_size = 1
    res = model(batch_input_ids,
          batch_token_type_id,
          batch_attention_mask)
    print(res.shape)


    torch.onnx.export(model,  # model being run
                      (batch_input_ids, batch_token_type_id, batch_attention_mask),  # model input (or a tuple for multiple inputs)
                      "finance.onnx",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=14,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input', 'input2', 'input3'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input1': {0: 'batch_size'},  # variable length axes
                                    'input2': {0: 'batch_size'},  # variable length axes
                                    'input3': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})


    # print(evaluation(model, dev_data_loader))

def use_onnx():
    import onnxruntime
    test_text = "SNOW 51连获A及A+轮共亿元级融资 目前在上海开设12家店\n36氪 思绮? ?2021-04-22 10:25\n" \
                "核心提示：SNOW 51连获A及A+轮共亿元级融资，该品牌定位滑雪新生活方式平台，目前在上海开有12家门店，" \
                "本轮融资后将加快在上海和一线城市落地门店。\n36氪获悉，滑雪产业一站式服务平台SNOW 51完成亿元级A系列轮融资。"
    local_max = len(test_text)
    codes = tokenizer.encode_plus(test_text,
                                  return_offsets_mapping=True,
                                  max_length=local_max,
                                  truncation=True,
                                  return_length=True,
                                  padding="max_length")

    input_ids = torch.tensor(codes["input_ids"]).long()
    attention_mask = torch.tensor(codes["attention_mask"]).long()
    token_type_ids = torch.tensor(codes["token_type_ids"]).long()
    batch_input_ids, batch_attention_mask, batch_token_type_id = [], [], []
    batch_input_ids.append(input_ids)
    batch_attention_mask.append(attention_mask)
    batch_token_type_id.append(token_type_ids)

    batch_input_ids = torch.stack(batch_input_ids, dim=0)
    batch_attention_mask = torch.stack(batch_attention_mask, dim=0)
    batch_token_type_id = torch.stack(batch_token_type_id, dim=0)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    ort_session = onnxruntime.InferenceSession("finance.onnx")
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(batch_input_ids),
    #               ort_session.get_inputs()[1].name: to_numpy(batch_token_type_id),
    #               ort_session.get_inputs()[2].name: to_numpy(batch_attention_mask)}
    # ort_outs = ort_session.run(None, ort_inputs)
    # print(ort_outs[0])

def get_model(rt_data, model_type):
    label2id = rt_data["label2id"]
    role2id = rt_data["role2id"]
    # bert_train_list = rt_data["bert_data"]["train"]

    # id2label = {v: k for k, v in label2id.items()}
    # id2role = {v: k for k, v in role2id.items()}

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--pretrain_name", type=str, default="hfl/chinese-roberta-wwm-ext", required=False)
    parser.add_argument("--batch_size", type=int, default=16, required=False)
    parser.add_argument("--epoch", type=int, default=30, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    parser.add_argument('--hidden_size', type=int, default=768, required=False)
    parser.add_argument('--entity_size', type=int, default=len(label2id), required=False)
    parser.add_argument('--warmup_proportion', type=float, default=0.9, required=False)
    parser.add_argument('--dropout', type=float, default=0.5, required=False)
    parser.add_argument('--gpu', type=bool, default=False, required=False)
    config = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained("hfl/chinese-roberta-wwm-ext")
    bert_model = EventModelV1(config)
    model_pt = torch.load("{}.pt".format(model_type))
    bert_model.load_state_dict(model_pt)

    return tokenizer, bert_model

if __name__ == "__main__":
    # save_load_model(rt_data)
    pass

