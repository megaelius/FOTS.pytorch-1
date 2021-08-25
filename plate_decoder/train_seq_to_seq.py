from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import os
import tqdm
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler
from torch.nn.utils.rnn import pad_sequence

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''

    ## get sequence lengths
    in_lengths = torch.tensor([ len(s['In_idxs']) for s in batch])
    out_lengths = torch.tensor([ len(s['Out_idxs']) for s in batch])
    ## padd
    in_idxs = [torch.tensor(s['In_idxs']) for s in batch]
    in_idxs = torch.nn.utils.rnn.pad_sequence(in_idxs,batch_first = True)
    out_idxs = [torch.tensor(s['Out_idxs']) for s in batch]
    out_idxs = torch.nn.utils.rnn.pad_sequence(out_idxs,batch_first = True)

    ## words
    in_plates = [s['In_plate'] for s in batch]
    out_plates = [s['Out_plate'] for s in batch]

    b = {"In_plate": in_plates, "In_idxs": in_idxs, 'In_lengths': in_lengths,
              "Out_plate": out_plates, "Out_idxs": out_idxs, 'Out_lengths': out_lengths}
    return b

'''
TODO:
restrict output to only mayus letters, - and numbers
'''
class PlateCorrectionDataset(Dataset):
    def __init__(self, data_pkl, data_alphabet):
        #random.seed(1234)
        with open(data_pkl, 'rb') as f:
            self.plates = list(pickle.load(f))#[:50000]
        print(f'MAX LENGTH: {max([len(p) for p in self.plates])}')
        self.alphabet = set()
        with open(data_alphabet, 'rb') as f:
            for line in f:
                for char in str(line).replace('\n',''):
                    self.alphabet.add(char)
        self.plates = [plate.replace('\t','') for plate in self.plates]
        for plate in self.plates:
            for char in plate:
                self.alphabet.add(char)
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.char_to_idx = {"PAD":0,"SOS":1, "EOS":2}
        self.idx_to_char = ['PAD',"SOS", "EOS"]
        print(self.alphabet)
        for i,c in enumerate(sorted(self.alphabet)):
            self.char_to_idx[c] = i+3
            self.idx_to_char.append(c)
        print(self.char_to_idx)
        print(self.idx_to_char)
        print(len(self.idx_to_char))

    def random_transform_plate(self,plate):
        char_list = list(plate)
        #iterate over characters and randomly change them
        for i in range(len(char_list)):
            #randomly change the character with 0.1 probability
            if random.random() <= 0.075:
                char_list[i] = random.sample(self.alphabet,1)[0]
        #randomly eliminate characters from back or beginning
        if random.random() <= 0.2:
            #get number of chars to erase
            n_erase = random.sample(range(1,4),1)[0]
            #choose front or back
            if random.random() <= 0.5:
                #front
                char_list = char_list[n_erase:]
            else:
                #back
                char_list = char_list[:-n_erase]
        return ''.join(char_list)

    def __len__(self):
        return len(self.plates)

    def __getitem__(self, idx):
        out_plate = self.plates[idx]
        out_idxs = [self.char_to_idx[c] for c in out_plate] + [self.EOS_token]
        in_plate = self.random_transform_plate(out_plate)
        in_idxs = [self.char_to_idx[c] for c in in_plate] + [self.EOS_token]
        sample = {"In_plate": in_plate, "In_idxs": in_idxs,
                  "Out_plate": out_plate, "Out_idxs": out_idxs}
        return sample

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.2):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first = True, enforce_sorted = False)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first = True)
        # Sum bidirectional GRU outputs
        #outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        #hidden = hidden.sum(dim=0).unsqueeze(0)
        # Return output and final hidden state
        return outputs, hidden

# Luong attention layer
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

# TorchScript Notes:
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Similarly to the ``EncoderRNN``, this module does not contain any
# data-dependent control flow. Therefore, we can once again use
# **tracing** to convert this model to TorchScript after it
# is initialized and its parameters are loaded.
#

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.2):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout), bidirectional = True)
        self.concat = nn.Linear(hidden_size * 4, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = self.logsoftmax(output)
        # Return output and final hidden state
        return output, hidden

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, device, SOS_token, decoder_n_layers):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.SOS_token = SOS_token
        self.decoder_n_layers = decoder_n_layers

    def forward(self, input_seq : torch.Tensor, input_length : torch.Tensor, max_length : int):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self._decoder_n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=self._device, dtype=torch.long) * self._SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=self._device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self._device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores

teacher_forcing_ratio = 0.5

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def train(sample, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, device, idx_to_char):
    encoder.train()
    decoder.train()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_tensor = sample['In_idxs'].to(device)
    target_tensor = sample['Out_idxs'].to(device)
    input_length = sample['In_lengths']
    target_length = sample['Out_lengths']

    total_distance = 0
    loss = 0
    encoder_outputs, encoder_hidden = encoder(
        input_tensor, input_length)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for j,(t,tl) in enumerate(zip(target_tensor,target_length)):
            decoder_input = torch.tensor([[dataset.SOS_token]], device=device)
            dh = decoder_hidden[:,j,:].unsqueeze(1).contiguous()
            eo = encoder_outputs[j].unsqueeze(1).contiguous()
            out_plate = ''
            # Teacher forcing: Feed the target as the next input
            for di in range(tl):
                decoder_output, dh = decoder(
                    decoder_input, dh, eo)
                #print(decoder_output.shape, t[di].unsqueeze(0).shape)
                loss += criterion(decoder_output, t[di].unsqueeze(0))
                decoder_input = t[di].unsqueeze(0).unsqueeze(0)  # Teacher forcing
                topv, topi = decoder_output.topk(1)
                out_plate += idx_to_char[topi.squeeze().detach()]
            total_distance += levenshtein(out_plate,sample['Out_plate'][j]) - levenshtein(sample['In_plate'][j],sample['Out_plate'][j])

    else:
        # Without teacher forcing: use its own predictions as the next input
        for j,(t,tl) in enumerate(zip(target_tensor,target_length)):
            decoder_input = torch.tensor([[dataset.SOS_token]], device=device)
            dh = decoder_hidden[:,j,:].unsqueeze(1).contiguous()
            eo = encoder_outputs[j].unsqueeze(1).contiguous()
            out_plate = ''
            for di in range(tl):
                decoder_output, dh = decoder(
                    decoder_input, dh, eo)
                #print(decoder_output.shape, t[di].unsqueeze(0).shape)
                loss += criterion(decoder_output, t[di].unsqueeze(0))
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                out_plate += idx_to_char[decoder_input]
                if decoder_input.item() == dataset.EOS_token:
                    break
                decoder_input = decoder_input.unsqueeze(0).unsqueeze(0)
            total_distance += levenshtein(out_plate,sample['Out_plate'][j]) - levenshtein(sample['In_plate'][j],sample['Out_plate'][j])
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item(), total_distance

def evaluate(sample, encoder, decoder, criterion, device, idx_to_char):
    encoder.eval()
    decoder.eval()
    input_tensor = sample['In_idxs'].to(device)
    target_tensor = sample['Out_idxs'].to(device)
    input_length = sample['In_lengths']
    target_length = sample['Out_lengths']
    with torch.no_grad():
        loss = 0
        total_distance = 0
        encoder_outputs, encoder_hidden = encoder(
            input_tensor, input_length)

        decoder_hidden = encoder_hidden

        # Without teacher forcing: use its own predictions as the next input
        for j,(t,tl) in enumerate(zip(target_tensor,target_length)):
            decoder_input = torch.tensor([[dataset.SOS_token]], device=device)
            dh = decoder_hidden[:,j,:].unsqueeze(1).contiguous()
            eo = encoder_outputs[j].unsqueeze(1).contiguous()
            out_plate = ''
            for di in range(tl):
                decoder_output, dh = decoder(
                    decoder_input, dh, eo)
                #print(decoder_output.shape, t[di].unsqueeze(0).shape)
                loss += criterion(decoder_output, t[di].unsqueeze(0))
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                out_plate += idx_to_char[decoder_input]
                if decoder_input.item() == dataset.EOS_token:
                    break
                decoder_input = decoder_input.unsqueeze(0).unsqueeze(0)
            total_distance += levenshtein(out_plate,sample['Out_plate'][j]) - levenshtein(sample['In_plate'][j],sample['Out_plate'][j])
        return loss.item(), total_distance

######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#

def showPlot(points,filename):
    plt.figure()
    plt.plot(points)
    plt.savefig(os.path.join(model_folder,filename))

def trainIters(encoder, decoder, train_dataloader, val_dataloader, epochs, device, idx_to_char, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    train_losses = []
    train_distances = []
    val_losses = []
    val_distances = []
    train_print_loss_total = 0  # Reset every print_every
    train_plot_loss_total = 0  # Reset every plot_every
    val_print_loss_total = 0  # Reset every print_every
    val_plot_loss_total = 0  # Reset every plot_every
    train_print_dist_total = 0  # Reset every print_every
    train_plot_dist_total = 0  # Reset every plot_every
    val_print_dist_total = 0  # Reset every print_every
    val_plot_dist_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    train_iters = len(train_dataloader)
    val_iters = len(val_dataloader)
    encoder.to(device)
    decoder.to(device)
    for epoch in range(epochs):

        for i,sample in enumerate(tqdm.tqdm(iter(train_dataloader))):
            loss,dist = train(sample, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, device, idx_to_char)
            train_print_loss_total += loss
            train_plot_loss_total += loss
            train_print_dist_total += dist
            train_plot_dist_total += dist

            if (i+1)% print_every == 0:
                train_print_loss_avg = train_print_loss_total / print_every
                train_print_loss_total = 0
                train_print_dist_avg = train_print_dist_total / print_every
                train_print_dist_total = 0
                print('%s (%d %d%%) %.4f %.4f' % (timeSince(start, i / train_iters),
                                             i, i / train_iters * 100,train_print_loss_avg, train_print_dist_avg))

            if (i+1) % plot_every == 0:
                train_plot_loss_avg = train_plot_loss_total / plot_every
                train_plot_dist_avg = train_plot_dist_total / plot_every
                train_losses.append(train_plot_loss_avg)
                train_distances.append(train_plot_dist_avg)
                train_plot_loss_total = 0
                train_plot_dist_total = 0

                showPlot(train_losses,'train_loss.png')
                showPlot(train_distances,'train_distance.png')

        for i,sample in enumerate(tqdm.tqdm(iter(val_dataloader))):
            loss, dist = evaluate(sample, encoder,
                         decoder, criterion, device, idx_to_char)
            val_print_loss_total += loss
            val_plot_loss_total += loss
            val_print_dist_total += dist
            val_plot_dist_total += dist

            if (i+1)% print_every == 0:
                val_print_loss_avg = val_print_loss_total / print_every
                val_print_loss_total = 0
                val_print_dist_avg = val_print_dist_total / print_every
                val_print_dist_total = 0
                print('%s (%d %d%%) %.4f %.4f' % (timeSince(start, i / val_iters),
                                             i, i / val_iters * 100,val_print_loss_avg, val_print_dist_avg))

            if (i+1) % plot_every == 0:
                val_plot_loss_avg = val_plot_loss_total / plot_every
                val_plot_dist_avg = val_plot_dist_total / plot_every
                val_losses.append(val_plot_loss_avg)
                val_distances.append(val_plot_dist_avg)
                val_plot_loss_total = 0
                val_plot_dist_total = 0

                showPlot(val_losses,'val_loss.png')
                showPlot(val_distances,'val_distance.png')

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def demo(encoder, decoder, dataloader):
    with torch.no_grad():
        for sample in iter(dataloader):
            input_tensor = sample['In_idxs'].squeeze(0).to(device)
            input_length = sample['In_lengths'].squeeze(0)

            encoder_outputs, encoder_hidden = encoder(
                input_tensor, input_length)

            decoder_hidden = encoder_hidden

            for j,tl in enumerate(input_length):
                decoded_plate = []
                decoder_input = torch.tensor([[dataset.SOS_token]], device=device)
                dh = decoder_hidden[:,j,:].unsqueeze(1)
                eo = encoder_outputs[j].unsqueeze(1)
                # Teacher forcing: Feed the target as the next input
                for di in range(tl):
                    decoder_output, dh = decoder(
                        decoder_input, dh, eo)
                    topv, topi = decoder_output.data.topk(1)
                    if topi.item() == dataset.EOS_token:
                        decoded_plate.append('<EOS>')
                        break
                    else:
                        decoded_plate.append(dataset.idx_to_char[topi.item()])

                    decoder_input = topi.squeeze().detach().unsqueeze(0).unsqueeze(0)  # Teacher forcing

                #return decoded_plate, decoder_attentions[:di + 1]
                print(sample['In_plate'][j],''.join(decoded_plate[:-1]),sample['Out_plate'][j])
######################################################################
# Training and Evaluating
# =======================
#
# With all these helper functions in place (it looks like extra work, but
# it makes it easier to run multiple experiments) we can actually
# initialize a network and start training.
#
# Remember that the input sentences were heavily filtered. For this small
# dataset we can use relatively small networks of 256 hidden nodes and a
# single GRU layer. After about 40 minutes on a MacBook CPU we'll get some
# reasonable results.
#
# .. Note::
#    If you run this notebook you can train, interrupt the kernel,
#    evaluate, and continue training later. Comment out the lines where the
#    encoder and decoder are initialized and run ``trainIters`` again.
#

#dataset = PlateCorrectionDataset('../../../Data/PlateSet.pkl','../data/alphabet.txt')
dataset = PlateCorrectionDataset('/mnt/DATA/eabad/Data/PlateSet.pkl','../data/alphabet.txt')
print(dataset[0])
seed = 1234
bs = 64
num_workers = 2
epochs = 25

n=len(dataset)
n_train = int(n*0.7)
n_val = n-n_train

train_sample, val_sample = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
train_sampler = RandomSampler(train_sample)
val_sampler = RandomSampler(val_sample)
train_dataloader = DataLoader(train_sample, sampler = train_sampler, batch_size=bs, num_workers = num_workers, collate_fn = collate_fn_padd)
val_dataloader = DataLoader(val_sample, sampler = val_sampler, batch_size=bs, num_workers = num_workers, collate_fn = collate_fn_padd)

model_folder = '../../out/Model_seq-to-seq'
if not Path(model_folder).is_dir():
    Path(model_folder).mkdir()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
hidden_size = 256
encoder1 = EncoderRNN(len(dataset.idx_to_char), hidden_size, n_layers = 2).to(device)
attn_model = 'dot'
attn_decoder1 = LuongAttnDecoderRNN(attn_model, hidden_size, len(dataset.idx_to_char), n_layers = 2).to(device)

trainIters(encoder1, attn_decoder1, train_dataloader, val_dataloader, epochs, device, dataset.idx_to_char, print_every=16, plot_every = 4)

torch.save(encoder1,os.path.join(model_folder,'weights_encoder.pt'))
torch.save(attn_decoder1,os.path.join(model_folder,'weights_decoder.pt'))
'''
encoder1 = torch.load(os.path.join(model_folder,'weights_encoder.pt'), map_location = device)
attn_decoder1 = torch.load(os.path.join(model_folder,'weights_decoder.pt'), map_location = device)
demo(encoder1, attn_decoder1, val_dataloader)
