import torch
import time
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from data_utils import Dictionary, Corpus


class RNNLM(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, num_layers: int, device, dropout: float, batch_size: int):
        super(RNNLM, self).__init__()
        # print('Init model with vocab=%d, emb=%d, hid=%d, batch_size=%d, drop=%.3lf' %
        #       (vocab_size, embed_size, hidden_size, batch_size, dropout))
        self.drop = nn.Dropout(dropout)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size,
                            num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)

        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)
        x = self.drop(x)

        # Forward propagate LSTM
        out, (h, c) = self.lstm(x, h)

        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = self.drop(out)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))

        # Decode hidden states of all time steps
        out = self.linear(out)
        return out, (h, c)

    def save(self, path: str):
        torch.save(self, path)

    @staticmethod
    def load(path: str):
        return torch.load(path)

    def generate_states(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device))
