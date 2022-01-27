#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import json
import torch

from transformers import BertModel, BertTokenizer, BertTokenizerFast


train_data_path = "D:\data\实体链接\DuEL2.0\data\\train.json"
dev_data_path = "D:\data\实体链接\DuEL2.0\data\\dev.json"
kb_data_path = "D:\data\实体链接\DuEL2.0\data\\kb.json"

with open(train_data_path, "r", encoding="utf-8") as f:
    train_data = f.read()

with open(dev_data_path, "r", encoding="utf-8") as f:
    dev_data = f.read()
#
train_data_list = []
for dt in train_data.split("\n"):
    if dt:
        dt = json.loads(dt)
        train_data_list.append(dt)

dev_data_list = []
for dt in dev_data.split("\n"):
    if dt:
        dt = json.loads(dt)
        dev_data_list.append(dt)
#
#     break

with open(kb_data_path, "r", encoding="utf-8") as f:
    kb_data = f.read()

kb_data_list = kb_data.split("\n")
print("kg data {}".format(len(kb_data_list)))
kb_data_dict = dict()
kb_name_index = dict()
for kb_data in kb_data_list:
    if kb_data:
        kb_data = json.loads(kb_data)
        kb_data_dict[kb_data["subject_id"]] = kb_data



        kb_name_index.setdefault(kb_data["subject"], [])
        kb_name_index[kb_data["subject"]].append(kb_data["subject_id"])

        for alias in kb_data["alias"]:
            kb_name_index.setdefault(alias, [])
            kb_name_index[alias].append(kb_data["subject_id"])

        # for

# for k, row in kb_name_index.items():
#     if len(row) > 1:
#         print("name {} count {}".format(k, len(row)))

link_train_list = []

not_find = 0
for data in train_data_list:
    # print(data)
    state = 0
    text = data["text"]
    for mention in data["mention_data"]:
        mention_entity = mention["mention"]
        mention_offset = int(mention["offset"])

        assert text[mention_offset:mention_offset+len(mention_entity)] == mention_entity
        # print(mention)
        if mention["kb_id"] in ["NIL_Work", "NIL_Organization"]:
            # print(kb_data_dict[mention["kb_id"]])
            continue


        if mention_entity in kb_name_index:
            if ["kb_id"] != "NIL_Work" and mention["kb_id"] not in kb_name_index[mention_entity]:
                print(mention_entity)
                state = 1
            else:


                for kb_id in kb_name_index[mention_entity]:
                    kb_info = kb_data_dict[kb_id]
                    desc = [dt["object"] for dt in kb_info["data"] if dt["predicate"] == "摘要"]
                    if not desc:
                        continue
                    desc = desc[0]

                    if kb_id == mention["kb_id"]:
                        link_train_list.append({"mention": mention_entity, "offset": mention_offset, "text": text, "kb_id": kb_id, "link": 1, "desc": desc})
                    else:
                        link_train_list.append(
                            {"mention": mention_entity, "offset": mention_offset, "text": text, "kb_id": kb_id, "link": 0, "desc": desc})

    if state:
        not_find += 1
    # print(data)

link_dev_list = []
for data in dev_data_list:
    # print(data)
    state = 0
    text = data["text"]
    for mention in data["mention_data"]:
        mention_entity = mention["mention"]
        mention_offset = int(mention["offset"])

        assert text[mention_offset:mention_offset+len(mention_entity)] == mention_entity
        # print(mention)
        if mention["kb_id"] in ["NIL_Work", "NIL_Organization"]:
            # print(kb_data_dict[mention["kb_id"]])
            continue


        if mention_entity in kb_name_index:
            if ["kb_id"] != "NIL_Work" and mention["kb_id"] not in kb_name_index[mention_entity]:
                print(mention_entity)
                state = 1
            else:


                for kb_id in kb_name_index[mention_entity]:
                    kb_info = kb_data_dict[kb_id]
                    desc = [dt["object"] for dt in kb_info["data"] if dt["predicate"] == "摘要"]
                    if not desc:
                        continue
                    desc = desc[0]

                    if kb_id == mention["kb_id"]:
                        link_dev_list.append({"mention": mention_entity, "offset": mention_offset, "text": text, "kb_id": kb_id, "link": 1, "desc": desc})
                    else:
                        link_dev_list.append(
                            {"mention": mention_entity, "offset": mention_offset, "text": text, "kb_id": kb_id, "link": 0, "desc": desc})

    if state:
        not_find += 1


print("data all {}".format(len(train_data_list)))
print("link data {}".format(len(link_train_list)))
print("link dev data {}".format(len(link_dev_list)))
print("entity can not find link {}".format(not_find))

from functools import partial
from torch.utils.data import Dataset, DataLoader

class EntityLinkDataset(Dataset):
    def __init__(self, document_list, tokenizer):
        super(EntityLinkDataset, self).__init__()
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

            local_max = 0
            for document in documents:
                text = document["text"]
                local_max = max(len(text), local_max)
            local_max += 2

            for document in documents:
                gold_answer = []
                text = document["text"]
                entity_desc = document["desc"]
                # print(text)
                # print(entity_desc)

                # text_word = [t for t in list(text)]
                codes = self.tokenizer.encode_plus(text,
                                                   entity_desc,
                                                   return_offsets_mapping=True,
                                                   max_length=local_max,
                                                   truncation=True,
                                                   return_length=True,
                                                   padding="max_length")

                input_ids_ = codes["input_ids"]
                # print(text_word)

                # print(codes["offset_mapping"])

                input_ids = torch.tensor(input_ids_).long()
                attention_mask = torch.tensor(codes["attention_mask"]).long()
                token_type_ids = torch.tensor(codes["token_type_ids"]).long()

                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_token_type_id.append(token_type_ids)
                batch_input_labels.append(torch.tensor([document["link"]]))
            batch_input_labels = torch.stack(batch_input_labels, dim=0).float()
            batch_input_ids = torch.stack(batch_input_ids, dim=0)
            batch_attention_mask = torch.stack(batch_attention_mask, dim=0).byte()
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


char2id = {
    "<pad>": 0,
    "<unk>": 1
}

for doc in link_train_list:

    # print(doc)
    for c in doc["text"]:
        if c not in char2id:
            char2id[c] = len(char2id)
    for c in doc["desc"]:
        if c not in char2id:
            char2id[c] = len(char2id)


class EntityLinkDatasetV2(Dataset):
    def __init__(self, document_list, char2id):
        super(EntityLinkDatasetV2, self).__init__()
        self.document_list = document_list
        self.char2id = char2id

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
            batch_mention_ids, batch_entity_ids = [], []
            batch_input_labels = []

            local_mention_max = 0
            local_entity_max = 0
            for document in documents:
                text = document["text"]
                local_mention_max = max(len(text), local_mention_max)
                local_entity_max = max(len(document["desc"]), local_entity_max)

            for document in documents:
                text = document["text"]
                entity_desc = document["desc"]
                mention_id = [self.char2id.get(c, char2id["<unk>"]) for c in text]
                entity_id = [self.char2id.get(c, char2id["<unk>"]) for c in entity_desc]

                for _ in range(len(text), local_mention_max):
                    mention_id.append(char2id["<pad>"])
                for _ in range(len(entity_desc), local_entity_max):
                    entity_id.append(char2id["<pad>"])

                batch_mention_ids.append(torch.tensor(mention_id))
                batch_entity_ids.append(torch.tensor(entity_id))
                batch_input_labels.append(torch.tensor([document["link"]]))

            batch_input_labels = torch.stack(batch_input_labels, dim=0).float()
            batch_mention_ids = torch.stack(batch_mention_ids, dim=0)
            batch_entity_ids = torch.stack(batch_entity_ids, dim=0)


            return {
                "batch_mention_ids": batch_mention_ids,
                "batch_entity_ids": batch_entity_ids,
                "batch_input_labels": batch_input_labels,
            }

        return partial(collate)

    def get_dataloader(self, batch_size, num_workers=0, shuffle=False, batch_first=True, pin_memory=False,
                       drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(batch_first),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)


datasetv1 = EntityLinkDatasetV2(link_train_list, char2id)
