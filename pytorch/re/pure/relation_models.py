#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import BertModel, BertPreTrainedModel
from transformers import AlbertModel, AlbertPreTrainedModel
BertLayerNorm = torch.nn.LayerNorm

class BertForRelation(BertPreTrainedModel):
    def __init__(self, config, num_rel_labels):
        super(BertForRelation, self).__init__(config)
        self.num_labels = num_rel_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(config.hidden_size * 2)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sub_idx=None, obj_idx=None, input_position=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False, position_ids=input_position)
        sequence_output = outputs[0]
        sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, sub_idx)])
        obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_idx)])
        rep = torch.cat((sub_output, obj_output), dim=1)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifier(rep)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class EntityModel():

    def __init__(self, args, num_ner_labels):
        super().__init__()

        bert_model_name = args.model
        vocab_name = bert_model_name

        if args.bert_model_dir is not None:
            bert_model_name = str(args.bert_model_dir) + '/'
            # vocab_name = bert_model_name + 'vocab.txt'
            vocab_name = bert_model_name
            logger.info('Loading BERT model from {}'.format(bert_model_name))

        if args.use_albert:
            self.tokenizer = AlbertTokenizer.from_pretrained(vocab_name)
            self.bert_model = AlbertForEntity.from_pretrained(bert_model_name, num_ner_labels=num_ner_labels,
                                                              max_span_length=args.max_span_length)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(vocab_name)
            self.bert_model = BertForEntity.from_pretrained(bert_model_name, num_ner_labels=num_ner_labels,
                                                            max_span_length=args.max_span_length)

        self._model_device = 'cpu'
        self.move_model_to_cuda()

    def move_model_to_cuda(self):
        if not torch.cuda.is_available():
            logger.error('No CUDA found!')
            exit(-1)
        logger.info('Moving to CUDA...')
        self._model_device = 'cuda'
        self.bert_model.cuda()
        logger.info('# GPUs = %d' % (torch.cuda.device_count()))
        if torch.cuda.device_count() > 1:
            self.bert_model = torch.nn.DataParallel(self.bert_model)

    def _get_input_tensors(self, tokens, spans, spans_ner_label):
        start2idx = []
        end2idx = []

        bert_tokens = []
        bert_tokens.append(self.tokenizer.cls_token)
        for token in tokens:
            start2idx.append(len(bert_tokens))
            sub_tokens = self.tokenizer.tokenize(token)
            bert_tokens += sub_tokens
            end2idx.append(len(bert_tokens) - 1)
        bert_tokens.append(self.tokenizer.sep_token)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])

        bert_spans = [[start2idx[span[0]], end2idx[span[1]], span[2]] for span in spans]
        bert_spans_tensor = torch.tensor([bert_spans])

        spans_ner_label_tensor = torch.tensor([spans_ner_label])

        return tokens_tensor, bert_spans_tensor, spans_ner_label_tensor

    def _get_input_tensors_batch(self, samples_list, training=True):
        tokens_tensor_list = []
        bert_spans_tensor_list = []
        spans_ner_label_tensor_list = []
        sentence_length = []

        max_tokens = 0
        max_spans = 0
        for sample in samples_list:
            tokens = sample['tokens']
            spans = sample['spans']
            spans_ner_label = sample['spans_label']

            tokens_tensor, bert_spans_tensor, spans_ner_label_tensor = self._get_input_tensors(tokens, spans,
                                                                                               spans_ner_label)
            tokens_tensor_list.append(tokens_tensor)
            bert_spans_tensor_list.append(bert_spans_tensor)
            spans_ner_label_tensor_list.append(spans_ner_label_tensor)
            assert (bert_spans_tensor.shape[1] == spans_ner_label_tensor.shape[1])
            if (tokens_tensor.shape[1] > max_tokens):
                max_tokens = tokens_tensor.shape[1]
            if (bert_spans_tensor.shape[1] > max_spans):
                max_spans = bert_spans_tensor.shape[1]
            sentence_length.append(sample['sent_length'])
        sentence_length = torch.Tensor(sentence_length)

        # apply padding and concatenate tensors
        final_tokens_tensor = None
        final_attention_mask = None
        final_bert_spans_tensor = None
        final_spans_ner_label_tensor = None
        final_spans_mask_tensor = None
        for tokens_tensor, bert_spans_tensor, spans_ner_label_tensor in zip(tokens_tensor_list, bert_spans_tensor_list,
                                                                            spans_ner_label_tensor_list):
            # padding for tokens
            num_tokens = tokens_tensor.shape[1]
            tokens_pad_length = max_tokens - num_tokens
            attention_tensor = torch.full([1, num_tokens], 1, dtype=torch.long)
            if tokens_pad_length > 0:
                pad = torch.full([1, tokens_pad_length], self.tokenizer.pad_token_id, dtype=torch.long)
                tokens_tensor = torch.cat((tokens_tensor, pad), dim=1)
                attention_pad = torch.full([1, tokens_pad_length], 0, dtype=torch.long)
                attention_tensor = torch.cat((attention_tensor, attention_pad), dim=1)

            # padding for spans
            num_spans = bert_spans_tensor.shape[1]
            spans_pad_length = max_spans - num_spans
            spans_mask_tensor = torch.full([1, num_spans], 1, dtype=torch.long)
            if spans_pad_length > 0:
                pad = torch.full([1, spans_pad_length, bert_spans_tensor.shape[2]], 0, dtype=torch.long)
                bert_spans_tensor = torch.cat((bert_spans_tensor, pad), dim=1)
                mask_pad = torch.full([1, spans_pad_length], 0, dtype=torch.long)
                spans_mask_tensor = torch.cat((spans_mask_tensor, mask_pad), dim=1)
                spans_ner_label_tensor = torch.cat((spans_ner_label_tensor, mask_pad), dim=1)

            # update final outputs
            if final_tokens_tensor is None:
                final_tokens_tensor = tokens_tensor
                final_attention_mask = attention_tensor
                final_bert_spans_tensor = bert_spans_tensor
                final_spans_ner_label_tensor = spans_ner_label_tensor
                final_spans_mask_tensor = spans_mask_tensor
            else:
                final_tokens_tensor = torch.cat((final_tokens_tensor, tokens_tensor), dim=0)
                final_attention_mask = torch.cat((final_attention_mask, attention_tensor), dim=0)
                final_bert_spans_tensor = torch.cat((final_bert_spans_tensor, bert_spans_tensor), dim=0)
                final_spans_ner_label_tensor = torch.cat((final_spans_ner_label_tensor, spans_ner_label_tensor), dim=0)
                final_spans_mask_tensor = torch.cat((final_spans_mask_tensor, spans_mask_tensor), dim=0)
        # logger.info(final_tokens_tensor)
        # logger.info(final_attention_mask)
        # logger.info(final_bert_spans_tensor)
        # logger.info(final_bert_spans_tensor.shape)
        # logger.info(final_spans_mask_tensor.shape)
        # logger.info(final_spans_ner_label_tensor.shape)
        return final_tokens_tensor, final_attention_mask, final_bert_spans_tensor, final_spans_mask_tensor, final_spans_ner_label_tensor, sentence_length

    def run_batch(self, samples_list, try_cuda=True, training=True):
        # convert samples to input tensors
        tokens_tensor, attention_mask_tensor, bert_spans_tensor, spans_mask_tensor, spans_ner_label_tensor, sentence_length = self._get_input_tensors_batch(
            samples_list, training)

        output_dict = {
            'ner_loss': 0,
        }

        if training:
            self.bert_model.train()
            ner_loss, ner_logits, spans_embedding = self.bert_model(
                input_ids=tokens_tensor.to(self._model_device),
                spans=bert_spans_tensor.to(self._model_device),
                spans_mask=spans_mask_tensor.to(self._model_device),
                spans_ner_label=spans_ner_label_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device),
            )
            output_dict['ner_loss'] = ner_loss.sum()
            output_dict['ner_llh'] = F.log_softmax(ner_logits, dim=-1)
        else:
            self.bert_model.eval()
            with torch.no_grad():
                ner_logits, spans_embedding, last_hidden = self.bert_model(
                    input_ids=tokens_tensor.to(self._model_device),
                    spans=bert_spans_tensor.to(self._model_device),
                    spans_mask=spans_mask_tensor.to(self._model_device),
                    spans_ner_label=None,
                    attention_mask=attention_mask_tensor.to(self._model_device),
                )
            _, predicted_label = ner_logits.max(2)
            predicted_label = predicted_label.cpu().numpy()
            last_hidden = last_hidden.cpu().numpy()

            predicted = []
            pred_prob = []
            hidden = []
            for i, sample in enumerate(samples_list):
                ner = []
                prob = []
                lh = []
                for j in range(len(sample['spans'])):
                    ner.append(predicted_label[i][j])
                    # prob.append(F.softmax(ner_logits[i][j], dim=-1).cpu().numpy())
                    prob.append(ner_logits[i][j].cpu().numpy())
                    lh.append(last_hidden[i][j])
                predicted.append(ner)
                pred_prob.append(prob)
                hidden.append(lh)
            output_dict['pred_ner'] = predicted
            output_dict['ner_probs'] = pred_prob
            output_dict['ner_last_hidden'] = hidden

        return output_dict
