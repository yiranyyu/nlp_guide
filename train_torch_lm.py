import torch
import time
import argparse
import os
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from data_utils import Dictionary, Corpus
from torch_lm import RNNLM


def init(args):
    # Set the random seed manually for reproducibility.
    if args.seed:
        seed = args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


def detach(states):
    return [state.detach() for state in states]


def evaluate(model, data, criterion, seq_len, test=False, epoch=None):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        states = model.generate_states()
        for i in range(0, data.size(1) - seq_len, seq_len):
            inputs = data[:, i:i+seq_len].to(model.device)
            targets = data[:, (i+1):(i+1)+seq_len].to(model.device)
            states = detach(states)
            outputs, states = model(inputs, states)
            total_loss += inputs.size(1) * criterion(outputs,
                                                     targets.reshape(-1)).item()
    avg_loss = total_loss / (data.size(1) - 1)
    if not test:
        print('[Epoch {}] evaluate Loss: {:.4f}, ppl: {:5.2f}'.format(
            epoch, avg_loss, np.exp(avg_loss)), end='')
    return avg_loss


def get_model_path(args):
    return os.path.join(args.model_dir, '%s_hid%d_seq%d_bat%d_layers%d_lr%.3lf_drop%.3lf.pt' % (
        ('sample_' if args.use_sample else ''), args.hidden_size, args.seq_length, args.batch_size, args.num_layers, args.lr, args.dropout))


def get_log_path(args):
    return os.path.join(args.log_dir, '%s_hid%d_seq%d_bat%d_layers%d_lr%.3lf_drop%.3lf.pt' % (
        ('sample_' if args.use_sample else ''), args.hidden_size, args.seq_length, args.batch_size, args.num_layers, args.lr, args.dropout))


def train_epoch(nth_epoch: int, model, train_data, criterion, optimizer, args):
    model.train()
    states = model.generate_states()

    step = 0
    t0 = time.time()
    for i in range(0, train_data.size(1) - args.seq_length, args.seq_length):
        inputs = train_data[:, i:i+args.seq_length].to(model.device)
        targets = train_data[:, (i+1):(i+1)+args.seq_length].to(model.device)

        # Forward pass
        states = detach(states)
        outputs, states = model(inputs, states)
        loss = criterion(outputs, targets.reshape(-1))

        # Backward and optimize
        model.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        new_step = (i + 1) // args.seq_length
        step = new_step
        if args.verbose and step and step % 100 == 0:
            print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}, {:.3f}ms/batch'
                  .format(nth_epoch, args.epoch, step, (train_data.size(1) // args.seq_length), loss.item(), np.exp(loss.item()), ((time.time() - t0) * 10)))
            t0 = time.time()


def get_optimizer(type_: str, model):
    if type_ == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        print('Cannot use unsupported optimzer %s ' % type_)
        exit()


def train(args):
    print('Train with hid=%d layers=%d drop=%.3lf seq_len=%d seed=%s' %
          (args.hidden_size, args.num_layers, args.dropout, args.seq_length, args.seed))

    continuous_no_update_epochs = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _ = torch.empty(size=[0], device=device)
    train_data, eval_data, test_data = Corpus().get_data(
        args.data_dir, args.batch_size)
    model = RNNLM(vocab_size=args.vocab_size, embed_size=args.hidden_size, hidden_size=args.hidden_size,
                  num_layers=args.num_layers, device=device, dropout=args.dropout, batch_size=args.batch_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args.optimizer, model)

    best_val_loss = None
    for nth_epoch in range(1, args.epoch + 1):
        train_epoch(nth_epoch, model, train_data, criterion, optimizer, args)
        eval_loss = evaluate(model, data=eval_data, criterion=criterion,
                             seq_len=args.seq_length, epoch=nth_epoch)
        if not best_val_loss or eval_loss < best_val_loss:
            print(' >>> Save model %.3lf -> %.3lf' %
                  ((np.exp(best_val_loss) if best_val_loss else 0.0), np.exp(eval_loss)))
            best_val_loss = eval_loss
            continuous_no_update_epochs = 0
            model.save(get_model_path(args))
        else:
            continuous_no_update_epochs += 1
            print('')
        if continuous_no_update_epochs == args.continuous_no_update_epochs_threshold:
            break

    print('Test result is %s' % (np.exp(evaluate(RNNLM.load(get_model_path(args)),
                                                 data=test_data, criterion=criterion, seq_len=args.seq_length, test=True))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', help='ptb data directory',
                        dest='data_dir', required=True)
    parser.add_argument('-model_dir', help='Output model directory',
                        dest='model_dir', required=True)
    parser.add_argument(
        '-log_dir', help='Place the training log file', dest='log_dir', default=None)
    parser.add_argument('-hidden_size', help='Hidden size',
                        dest='hidden_size', default=300, type=int)
    parser.add_argument('-lr', help='Starting learning rate',
                        dest='lr', default=0.004, type=float)
    parser.add_argument('-epoch', help='Number of training epochs',
                        dest='epoch', default=30, type=int)
    parser.add_argument('-batch_size', help='Number of steps in one batch',
                        dest='batch_size', default=20, type=int)
    parser.add_argument('-seq_length', help='Length of sequence pass to rnn cell',
                        dest='seq_length', default=35, type=int)
    parser.add_argument(
        '-optimizer', help='Type of optimizer to use', dest='optimizer', default='Adam')
    parser.add_argument(
        '-seed', help='Rand seed for numpy and pytorch', dest='seed', default=None)
    parser.add_argument('-max_grad_norm', help='Minimum of gradient to clip',
                        dest='max_grad_norm', default=5, type=float)
    parser.add_argument('-dropout', help='Probability to drop a cell (deactivate it)',
                        dest='dropout', default=0.5, type=float)
    parser.add_argument('-num_layers', help='Number of LSTM layers',
                        dest='num_layers', default=2, type=int)
    parser.add_argument(
        '-use_sample', help='Set it to True to use data in ./sample sub_directory of data directory', dest='use_sample', default=False, type=bool)
    parser.add_argument('-continuous_no_update_epochs_threshold', help='If there is continuos n epochs without new best validation result, break the training early',
                        dest='continuous_no_update_epochs_threshold', default=5, type=int)
    parser.add_argument('-vocab_size', help='Vocab will be reduced to this size',
                        dest='vocab_size', default=10000, type=int)
    parser.add_argument('-verbose', help='Set it to True to see more trainig detail in console',
                        dest='verbose', default=False, type=bool)
    args = parser.parse_args()
    init(args)
    train(args)
