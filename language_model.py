import torch
import time
from logger import Logger
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
from data_utils import Dictionary, Corpus

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# FIXME: to activate cuda, I think it's a bug of caused by GPU driver
_ = torch.empty(size=[0], device=device)

# Hyper-parameters
embed_size = 1200
hidden_size = 1200
num_layers = 2
num_epochs = 30
batch_size = 20
seq_length = 30
learning_rate = 0.004

corpus = Corpus()
ids = corpus.get_data('./data/language model/ptb.train.txt', batch_size)
eval_data = corpus.get_data('./data/language model/ptb.valid.txt', batch_size)
vocab_size = len(corpus.dictionary)
num_batches = ids.size(1) // seq_length


class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.5):
        super(RNNLM, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size,
                            num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)

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


model = RNNLM(vocab_size, embed_size, hidden_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def evaluate(model, data):
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    with torch.no_grad():
        states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                  torch.zeros(num_layers, batch_size, hidden_size).to(device))
        for i in range(0, data.size(1) - seq_length, seq_length):
            inputs = data[:, i:i+seq_length].to(device)
            targets = data[:, (i+1):(i+1)+seq_length].to(device)
            states = detach(states)
            outputs, states = model(inputs, states)
            total_loss += inputs.size(1) * criterion(outputs,
                                                     targets.reshape(-1)).item()
    avg_loss = total_loss / (data.size(1) - 1)
    print('Evaluate Loss: {:.4f}, Perplexity: {:5.2f}'.format(
        avg_loss, np.exp(avg_loss)))
    return avg_loss


def detach(states):
    return [state.detach() for state in states]


logger = Logger('./log')
best_val_loss = None

for epoch in range(num_epochs):
    # evaluate(model, eval_data)
    model.train()
    states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
              torch.zeros(num_layers, batch_size, hidden_size).to(device))

    total_loss = 0
    t0 = time.time()
    for i in range(0, ids.size(1) - seq_length, seq_length):
        # Get mini-batch inputs and targets
        inputs = ids[:, i:i+seq_length].to(device)
        targets = ids[:, (i+1):(i+1)+seq_length].to(device)

        # Forward pass
        states = detach(states)
        outputs, states = model(inputs, states)
        loss = criterion(outputs, targets.reshape(-1))

        # Backward and optimize
        model.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        step = (i+1) // seq_length
        if step and step % 100 == 0:
            print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}, {:.3f}ms/batch'
                  .format(epoch+1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item()), ((time.time() - t0) * 10)))
            t0 = time.time()

            info = {'ppl': np.exp(loss.item())}
            for tag, value in info.items():
                logger.scalar_summary(tag, value, step+1)

    eval_loss = evaluate(model, eval_data)
    if not best_val_loss or eval_loss < best_val_loss:
        print('>>> Save model, %.3lf -> %.3lf' %
              ((best_val_loss if best_val_loss else 0.0), eval_loss))
        best_val_loss = eval_loss
        model.save('./model/best_language_model.pt')
