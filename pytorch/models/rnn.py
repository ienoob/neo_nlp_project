#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) ***
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.utils.data as Data

dtype = torch.FloatTensor
sentences = ["i like dog", "i love coffee", "i hate milk"]
word_list = " ".join(sentences).split()
vocab = list(set(word_list))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}
n_class = len(vocab)

batch_size = 2
n_step = 2 # number of cells(= number of Step)
n_hidden = 5 # number of hidden units in one cell

def make_data(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word2idx[n] for n in word[:-1]]
        target = word2idx[word[-1]]

        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return input_batch, target_batch

input_data, input_target = make_data(sentences)
input_batch, target_batch = torch.Tensor(input_data), torch.LongTensor(input_target)
dataset = Data.TensorDataset(input_batch, target_batch)
loader = Data.DataLoader(dataset, batch_size, True)


class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()
        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden)
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, hidden, X):
        X = X.transpose(0, 1)
        out, hidden = self.rnn(X, hidden)

        out = out[-1]
        model = self.fc(out)
        return model

model = TextRNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(50000):
    for x, y in loader:
        hidden = torch.zeros(1, x.shape[0], n_hidden)

        pred = model(hidden, x)

        loss = criterion(pred, y)
        if (epoch+1)%1000==0:
            print("Epoch:", "%04d" % (epoch + 1), "cost = ", "{:.6f}".format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# from torch.autograd import Variable
# dummy_input = Variable(torch.randn(1, 3, n_hidden))
# torch.onnx.export(model, dummy_input, "alexnet.proto", verbose=True)

input = [sen.split()[:2] for sen in sentences]
hidden = torch.zeros(1, len(input), n_hidden)
predict = model(hidden, input_batch).data.max(1, keepdim=True)[1]
print(input, "->", [idx2word[n.item()] for n in predict.squeeze()])
