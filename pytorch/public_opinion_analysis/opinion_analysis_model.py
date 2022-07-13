#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import torch
import argparse
import numpy as np
import torch.nn as nn
from functools import partial
from pytorch.layers.crf import CRF
from pytorch.layers.bert_optimization import BertAdam
from transformers import BertModel, BertTokenizer, BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
from train_data_gen_v3 import train_dataset
from nlp_applications.ner.evaluation import extract_entity, eval_metrix_v3

label2id = {"O": 0, "B-PERSON": 1, "I-PERSON": 2, "B-ORG": 3, "I-ORG": 4}
id2label = {v: k for k, v in label2id.items()}

class ConditionalLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(ConditionalLayerNorm, self).__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

        self.beta_dense = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.gamma_dense = nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(self, x, cond):
        cond = cond.unsqueeze(1)
        beta = self.beta_dense(cond)
        gamma = self.gamma_dense(cond)
        weight = self.weight + gamma
        bias = self.bias + beta

        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return weight * x + bias

def batch_gather(data: torch.Tensor, index: torch.Tensor):
    length = index.shape[0]
    t_index = index.cpu().numpy()
    t_data = data.cpu().data.numpy()
    result = []
    for i in range(length):
        result.append(t_data[i, t_index[i], :])

    return torch.from_numpy(np.array(result)).to(data.device)


class OpinionAnalysisModel(nn.Module):

    def __init__(self, config):
        super(OpinionAnalysisModel, self).__init__()

        self.bert_encoder = BertModel.from_pretrained(config.pretrain_name)
        self.hidden2tag = nn.Linear(config.hidden_size, config.entity_bio_size + 2)

        self.crf = CRF(config.entity_bio_size, config.gpu)
        self.event = nn.Linear(config.hidden_size, config.event_size)

        self.layer_norm = ConditionalLayerNorm(config.hidden_size, eps=1e-12)

        self.pn_predict = nn.Linear(config.hidden_size, config.pn_class)
        self.pn_entity_predict = nn.Linear(config.hidden_size, config.pn_class)

    def forward(self, input_id, input_segment_id, input_mask, batch_label=None, input_entity=None):
        outputs = self.bert_encoder(input_id, token_type_ids=input_segment_id, attention_mask=input_mask)
        outputs = outputs[0]
        tags = self.hidden2tag(outputs)
        event_logits = nn.Sigmoid()(self.event(outputs[:,0]))
        pn_logits = nn.Softmax(dim=-1)(self.pn_predict(outputs[:,0]))

        if batch_label is None:
            scores, tag_seq = self.crf._viterbi_decode(tags, input_mask)
            sentence_ids = []
            entity_span = []
            bert_encoders = []
            for ii, tag_pred in enumerate(tag_seq.detach().numpy()):
                tag_seq_list = [id2label.get(tag, "O") for tag in tag_pred]

                tag_value = extract_entity(tag_seq_list)

                for entity in tag_value:
                    sentence_ids.append(ii)
                    entity_span.append(torch.tensor((entity[0], entity[1]-1), dtype=torch.long))
                    bert_encoders.append(outputs[ii])

            bert_encoders = torch.stack(bert_encoders).to(outputs.device)
            entity_span = torch.stack(entity_span).to(outputs.device)

            # print(bert_encoders.shape)
            # print(entity_span.shape)

            sub_start_encoder = batch_gather(bert_encoders, entity_span[:, 0])
            sub_end_encoder = batch_gather(bert_encoders, entity_span[:, 1])
            entity_feature = torch.cat([sub_start_encoder, sub_end_encoder], 1)
            context_encoder = self.layer_norm(bert_encoders, entity_feature)

            pn_entity_logits = nn.Softmax(dim=-1)(self.pn_entity_predict(context_encoder[:, 0]))

            return tag_seq, event_logits, pn_logits, pn_entity_logits, sentence_ids
        else:
            total_loss = self.crf.neg_log_likelihood_loss(tags, input_mask, batch_label)
            scores, tag_seq = self.crf._viterbi_decode(tags, input_mask)


            sub_start_encoder = batch_gather(outputs, input_entity[:, 0])
            sub_end_encoder = batch_gather(outputs, input_entity[:, 1])

            entity_feature = torch.cat([sub_start_encoder, sub_end_encoder], 1)
            context_encoder = self.layer_norm(outputs, entity_feature)

            pn_entity_logits = nn.Softmax(dim=-1)(self.pn_entity_predict(context_encoder[:, 0]))


            return total_loss, tag_seq, event_logits, pn_logits, pn_entity_logits


class OpinionAnalysisDataset(Dataset):
    def __init__(self, document_list, tokenizer):
        super(OpinionAnalysisDataset, self).__init__()
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
            batch_event_labels = []
            batch_event_s_labels = []
            batch_pos_neg_labels = []
            batch_pn_entity_span = []
            batch_pn_entity_span_labels = []

            local_max = 0
            for document in documents:
                text = document[0]
                local_max = max(len(text), local_max)
            local_max += 2

            for document in documents:
                event_s_labels = []
                text = document[0]
                text_word = [t for t in list(text)]
                codes = self.tokenizer.encode_plus(text_word,
                                                   return_offsets_mapping=True,
                                                   is_split_into_words=True,
                                                   max_length=local_max,
                                                   truncation=True,
                                                   return_length=True,
                                                   padding="max_length")

                input_ids_ = codes["input_ids"]
                iiv = len(input_ids_) - 1
                while input_ids_[iiv] == 0:
                    iiv -= 1

                assert iiv + 1 == len(document[0]) + 2


                input_ids = torch.tensor(input_ids_).long()
                attention_mask = torch.tensor(codes["attention_mask"]).long()
                token_type_ids = torch.tensor(codes["token_type_ids"]).long()

                label_id = [0] + document[1]
                tag_seq_list = [id2label.get(tag, "O") for tag in label_id]

                tag_value = extract_entity(tag_seq_list)
                if len(tag_value) == 0:
                    batch_pn_entity_span.append(torch.LongTensor((0, 0)))
                    batch_pn_entity_span_labels.append(torch.LongTensor([document[3]]))
                else:
                    tag_value_idx = [i for i, _ in enumerate(tag_value)]
                    # print(tag_value_idx)
                    entity_sp = tag_value[np.random.choice(tag_value_idx)]
                    # print(entity_sp)
                    batch_pn_entity_span.append(torch.LongTensor((entity_sp[0], entity_sp[1]-1)))
                    batch_pn_entity_span_labels.append(torch.LongTensor([0]))

                for _ in range(local_max - len(label_id)):
                    label_id.append(0)

                batch_input_labels.append(label_id)

                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_token_type_id.append(token_type_ids)
                event_label_ = [0]*14
                for ei in document[2]:
                    event_label_[ei] = 1
                    event_s_labels.append(ei)

                batch_event_labels.append(torch.LongTensor(event_label_))
                batch_pos_neg_labels.append(torch.LongTensor([document[3]]))

                batch_event_s_labels.append(event_s_labels)

            batch_input_ids = torch.stack(batch_input_ids, dim=0)
            batch_attention_mask = torch.stack(batch_attention_mask, dim=0).byte()
            batch_token_type_id = torch.stack(batch_token_type_id, dim=0)
            batch_event_labels = torch.stack(batch_event_labels, dim=0).float()
            batch_pn_entity_span = torch.stack(batch_pn_entity_span, dim=0)
            batch_pos_neg_labels = torch.stack(batch_pos_neg_labels, dim=0)
            batch_pn_entity_span_labels = torch.stack(batch_pn_entity_span_labels, dim=0)
            batch_input_labels = torch.LongTensor(batch_input_labels)


            return {
                "batch_input_ids": batch_input_ids,
                "batch_attention_mask": batch_attention_mask,
                "batch_token_type_id": batch_token_type_id,
                "batch_input_labels": batch_input_labels,
                "batch_event_labels": batch_event_labels,
                "batch_event_s_labels": batch_event_s_labels,
                "batch_pn_entity_span": batch_pn_entity_span,
                "batch_pn_entity_span_labels": batch_pn_entity_span_labels,
                "batch_pos_neg_labels": batch_pos_neg_labels
            }

        return partial(collate)

    def get_dataloader(self, batch_size, num_workers=0, shuffle=False, batch_first=True, pin_memory=False,
                       drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(batch_first),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)


def evaluation(model, data_iterator, config):
    label2id = {"O": 0, "B-PERSON": 1, "I-PERSON": 2, "B-ORG": 3, "I-ORG": 4}
    id2label = {v: k for k, v in label2id.items()}
    model.eval()
    pre_num = 0.0
    true_num = 0.0
    hit_num = 0.0

    event_hit_num = 0.0
    event_pre_num = 0.0
    event_true_num = 0.0

    pn_hit = 0.0
    pn_all_num = 0.0

    for idx, batch_data in enumerate(data_iterator):
        tag_seqs, event_logits, pn_logits, pn_entity_logits, sentence_ids = model(batch_data["batch_input_ids"].to(config.device),
                              batch_data["batch_token_type_id"].to(config.device),
                              batch_data["batch_attention_mask"].to(config.device))

        tag_seqs = tag_seqs * batch_data["batch_attention_mask"].to(config.device)
        tag_seqs = tag_seqs.cpu().numpy()

        # print(tag_seqs.shape)
        label_seq = batch_data["batch_input_labels"].numpy()
        batch_num_value = label_seq.shape[0]

        pn_all_num += batch_num_value

        for b in range(batch_num_value):
            tag_seq_list = tag_seqs[b]
            # print(tag_seq_list)
            tag_seq_list = [id2label.get(tag, "O") for tag in tag_seq_list]
            # print(tag_seq_list)

            true_seq_list = label_seq[b]
            true_seq_list = [id2label.get(tag, "O") for tag in true_seq_list]

            pre_value = extract_entity(tag_seq_list)
            # print(pre_value)
            true_value = extract_entity(true_seq_list)

            pre_num += len(pre_value)
            true_num += len(true_value)

            for p in pre_value:
                if p in true_value:
                    hit_num += 1

            # print(pre_value)

            event_pre = event_logits[b]
            event_res = []
            for ii, e in enumerate(event_pre.detach().numpy()):
                if e > 0.5:
                    event_res.append(ii)
            pn_lb = batch_data["batch_pos_neg_labels"][b][0]
            if torch.argmax(pn_logits[b]) == pn_lb:
                pn_hit += 1

            true_event_res = batch_data["batch_event_s_labels"][b]
            # print(true_event_res)

            # true_event_res = []
            # for ii, e in enumerate(event_s_labels.detach().numpy()):
            #     if e == 1:
            #         true_event_res.append(ii)

            event_pre_num += len(event_res)
            event_true_num += len(true_event_res)

            for er in event_res:
                if er in true_event_res:
                    event_hit_num += 1
    print(hit_num, true_num, pre_num)
    metric = eval_metrix_v3(hit_num, true_num, pre_num)
    event_metric = eval_metrix_v3(event_hit_num, event_true_num, event_pre_num)

    print("ner evaluation {}".format(metric))
    print("event evaluation {}".format(event_metric))
    print("pn evaluation {}".format(pn_hit/pn_all_num))


def train():
    device = "cpu"
    if torch.cuda.is_available():
        device = "gpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_name", type=str, default="hfl/chinese-roberta-wwm-ext", required=False)
    parser.add_argument("--epoch", type=int, default=50, required=False)
    parser.add_argument('--hidden_size', type=int, default=768, required=False)
    parser.add_argument("--batch_size", type=int, default=10, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument('--warmup_proportion', type=float, default=0.9, required=False)
    parser.add_argument('--entity_bio_size', type=int, default=5, required=False)
    parser.add_argument('--event_size', type=int, default=14, required=False)
    parser.add_argument('--pn_class', type=int, default=3, required=False)
    parser.add_argument('--gpu', type=bool, default=False, required=False)
    parser.add_argument('--device', type=str, default=device, required=False)

    config = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained(config.pretrain_name)

    dataset = OpinionAnalysisDataset(train_dataset, tokenizer)
    # dev_dataset = DUEEFI(dev_data_lists, tokenizer)

    train_data_loader = dataset.get_dataloader(config.batch_size,
                                               shuffle=config.shuffle,
                                               pin_memory=config.pin_memory)
    # dev_data_loader = dev_dataset.get_dataloader(config.batch_size,
    #                                              shuffle=config.shuffle,
    #                                              pin_memory=config.pin_memory)
    loss_fn = torch.nn.BCELoss(reduce=True, size_average=False)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    model = OpinionAnalysisModel(config)
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

    for epoch in range(config.epoch):
        model.train()
        for idx, batch_data in enumerate(train_data_loader):
            # print(idx)
            # print(batch_data["batch_input_ids"].shape)
            # print(batch_data["batch_input_labels"].shape)
            total_loss, tag_seq, event_logits, pn_logits, pn_entity_logits = model(batch_data["batch_input_ids"].to(device),
                                      batch_data["batch_token_type_id"].to(device),
                                      batch_data["batch_attention_mask"].to(device),
                                      batch_data["batch_input_labels"].to(device),
                                      batch_data["batch_pn_entity_span"].to(device)
                                                      )

            # print(total_loss)

            event_loss = loss_fn(event_logits, batch_data["batch_event_labels"])

            pn_loss = criterion(pn_logits.view(-1, config.pn_class), batch_data["batch_pos_neg_labels"].view(-1))
            pn_entity_loss = criterion(pn_entity_logits.view(-1, config.pn_class), batch_data["batch_pn_entity_span_labels"].view(-1))

            # print(event_loss)

            loss = total_loss + event_loss + pn_loss + pn_entity_loss
            if idx % 20 == 0:
                loss_value = loss.data.cpu().numpy()
                print("epoch {0} batch {1} loss value is {2}".format(epoch, idx, loss_value))


            loss.backward()
            optimizer.step()
            model.zero_grad()
        # evaluation(model, train_data_loader, config)
        torch.save(model.state_dict(), "{}.pt".format("opinion_analysis"))
        # break

def load_model_predict():
    device = "cpu"
    if torch.cuda.is_available():
        device = "gpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_name", type=str, default="hfl/chinese-roberta-wwm-ext", required=False)
    parser.add_argument("--epoch", type=int, default=50, required=False)
    parser.add_argument('--hidden_size', type=int, default=768, required=False)
    parser.add_argument("--batch_size", type=int, default=10, required=False)
    parser.add_argument("--shuffle", type=bool, default=True, required=False)
    parser.add_argument('--pin_memory', type=bool, default=False, required=False)
    parser.add_argument("--learning_rate", type=float, default=5e-5, required=False)
    parser.add_argument('--warmup_proportion', type=float, default=0.9, required=False)
    parser.add_argument('--entity_bio_size', type=int, default=5, required=False)
    parser.add_argument('--event_size', type=int, default=14, required=False)
    parser.add_argument('--pn_class', type=int, default=3, required=False)
    parser.add_argument('--gpu', type=bool, default=False, required=False)
    parser.add_argument('--device', type=str, default=device, required=False)

    config = parser.parse_args()

    tokenizer = BertTokenizerFast.from_pretrained(config.pretrain_name)

    dataset = OpinionAnalysisDataset(train_dataset, tokenizer)
    train_data_loader = dataset.get_dataloader(config.batch_size,
                                               shuffle=config.shuffle,
                                               pin_memory=config.pin_memory)

    model = OpinionAnalysisModel(config)
    model.to(config.device)
    model_pt = torch.load("opinion_analysis.pt")
    model.load_state_dict(model_pt)

    evaluation(model, train_data_loader, config)





load_model_predict()




