import os
import torch
import random
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler
from torch.nn.utils.rnn import pad_sequence

def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''

    ## get sequence lengths
    lengths = torch.tensor([ len(s['Indexes']) for s in batch])
    ## padd
    idx = [ torch.tensor(s['Indexes']) for s in batch]
    idx = torch.nn.utils.rnn.pad_sequence(idx,batch_first = True)
    ## compute mask
    mask = (idx != 0)

    ## words
    words = [s['Word'] for s in batch]

    ## words
    classes = torch.Tensor([s['Class'] for s in batch])

    b = {'Word':words, 'Class': classes,
         'Indexes': idx, 'Lengths': lengths,
         'Mask': mask}
    return b

class PlateClassificationDataset(Dataset):
    def __init__(self, data_path):
        df = pd.read_csv(data_path)
        self.words = list(df['Word'])
        self.labels = list(df['Is_plate'])
        self.vocabulary = set()
        for w in self.words:
            chars = []
            for c in str(w):
                self.vocabulary.add(c)
        print(self.vocabulary)
        self.char_to_idx = {c:i+1 for i,c in enumerate(self.vocabulary)}
        self.idx_to_char = [c for c in enumerate(self.vocabulary)]

    def random_transform_word(self,word):
        char_list = list(word)
        #iterate over characters and randomly change them
        for i in range(len(char_list)):
            #randomly change the character with 0.1 probability
            if random.random() <= 0.05:
                char_list[i] = random.sample(self.vocabulary,1)[0]
        return ''.join(char_list)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        label = self.labels[idx]
        word = self.random_transform_word(str(self.words[idx]))
        idxs = [self.char_to_idx[c] for c in word]

        sample = {"Word": word, "Indexes": idxs, "Class": label}
        return sample

class PlateNet(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.emb = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.lin = torch.nn.Linear(embedding_dim, 1, bias=False)
        self.pad = pad_sequence
    # B = Batch size
    # W = Number of context words (left + right)
    # E = embedding_dim
    # V = num_embeddings (number of words)
    def forward(self, input, input_lengths):
        # input shape is (B, W)
        e = self.emb(input)
        # e shape is (B, W, E)
        u = e.sum(dim=1)
        # u shape is (B, E)
        v = self.lin(u)
        # v shape is (B, V)
        return v

class PlateRNN(torch.nn.Module):

    def __init__(self, num_embeddings, embedding_dim, hidden_size, output_size, num_layers=1, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.rnn = torch.nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=bidirectional)
        if bidirectional:
            self.h2o = torch.nn.Linear(2*hidden_size, output_size)
        else:
            self.h2o = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input, input_lengths):
        # T x B
        encoded = self.embed(input)
        # T x B x E
        packed = torch.nn.utils.rnn.pack_padded_sequence(encoded, input_lengths, batch_first = True)
        # Packed T x B x E
        output, _ = self.rnn(packed)
        # Packed T x B x E

        # Important: you may need to replace '-inf' with the default zero padding for other pooling layers
        padded, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first = True)
        # T x B x H
        output = padded.sum(dim=1)
        # B x H
        output = self.h2o(output)
        # B x O
        return output

def train(model, dataloader, criterion, optimizer, batch_size, device, log=False):
    model.train()
    total_loss = 0
    ncorrect = 0
    ntokens = 0
    niterations = 0
    for sample in iter(dataloader):
        X = sample['Indexes'].to(device)
        y = sample['Class'].to(device)

        input_lengths = [len(x) for x in X]
        model.zero_grad()
        output = model(X,input_lengths).squeeze(1)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        # Training statistics
        total_loss += loss.item()
        ncorrect += ((torch.nn.Sigmoid()(output) > 0.8) == y).sum().item()
        ntokens += y.numel()
        niterations += 1
        if niterations == 200 or niterations == 500 or niterations % 1000 == 0:
            print(f'Train: wpb={ntokens//niterations}, num_updates={niterations}, accuracy={100*ncorrect/ntokens:.1f}, loss={total_loss/ntokens:.2f}')

    total_loss = total_loss / ntokens
    accuracy = 100 * ncorrect / ntokens
    if log:
        print(f'Train: wpb={ntokens//niterations}, num_updates={niterations}, accuracy={accuracy:.1f}, loss={total_loss:.2f}')
    return accuracy, total_loss

def validate(model, dataloader, criterion, batch_size, device):
    model.eval()
    total_loss = 0
    ncorrect = 0
    ntokens = 0
    niterations = 0
    y_pred = []
    with torch.no_grad():
        for sample in iter(dataloader):
            X = sample['Indexes'].to(device)
            y = sample['Class'].to(device)
            input_lengths = [len(x) for x in X]
            output = model(X,input_lengths).squeeze(1)
            loss = criterion(output, y)
            total_loss += loss.item()
            ncorrect += ((torch.nn.Sigmoid()(output) > 0.8) == y).sum().item()
            ntokens += y.numel()
            niterations += 1

    total_loss = total_loss / ntokens
    accuracy = 100 * ncorrect / ntokens
    return accuracy, total_loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output',type=str)
    args = parser.parse_args()

    seed = 1234
    bs = 1024
    num_workers = 2
    epochs = 25
    embedding_dim = 128
    hidden_size = 32

    dataset = PlateClassificationDataset(args.data_path)
    n=len(dataset)
    n_train = int(n*0.7)
    n_val = n-n_train

    train_sample, val_sample = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    train_sampler = RandomSampler(train_sample)
    val_sampler = RandomSampler(val_sample)
    train_dataloader = DataLoader(train_sample, sampler = train_sampler, batch_size=bs, num_workers = num_workers, collate_fn = collate_fn_padd)
    val_dataloader = DataLoader(val_sample, sampler = val_sampler, batch_size=bs, num_workers = num_workers, collate_fn = collate_fn_padd)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = PlateRNN(num_embeddings = len(dataset.vocabulary) + 1, embedding_dim = embedding_dim, hidden_size = hidden_size, output_size = 1, bidirectional = True)
    #model = PlateNet(num_embeddings = len(dataset.vocabulary) + 1, embedding_dim = 2)
    model = model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')

    optimizer = torch.optim.Adam(model.parameters())

    train_accuracy = []
    valid_accuracy = []
    for epoch in range(epochs):

        acc, loss = train(model, train_dataloader, criterion, optimizer, bs, device, log=True)
        train_accuracy.append(acc)
        print(f'| epoch {epoch:03d} | train accuracy={acc:.1f}%, train loss={loss:.2f}')

        acc, loss = validate(model, val_dataloader, criterion, bs, device)
        valid_accuracy.append(acc)
        print(f'| epoch {epoch:03d} | valid accuracy={acc:.1f}%, valid loss={loss:.2f}')

    # Save model
    if not Path(args.output).is_dir():
        Path(args.output).mkdir()
    torch.save(model, os.path.join(args.output,'weights.pt'))

    #Save char to index dictionary
    with open(os.path.join(args.output,'char_to_idx.pkl'), 'wb') as handle:
        pickle.dump(dataset.char_to_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #Some analysis
    '''
    model = torch.load(os.path.join(args.output,'weights.pt'), map_location = device)

    #Load char to index dictionary
    with open(os.path.join(args.output,'char_to_idx.pkl'), 'rb') as handle:
        char_to_idx = pickle.load(handle)
    dataset.char_to_idx = char_to_idx
    print(model.emb.weight.cpu().data.numpy())
    for char in sorted(dataset.char_to_idx):
        print(f'{char} : {model.emb.weight[dataset.char_to_idx[char]].cpu().data.numpy()}')
    print(model.lin.weight.cpu().data.numpy())
    plane = model.lin.weight.cpu().data.numpy()[0]
    weights = model.emb.weight.cpu().data.numpy()
    x = weights.transpose(1,0)[0]
    y = weights.transpose(1,0)[1]
    print(x,y,sorted(dataset.char_to_idx))
    plt.scatter(x,y)
    plt.quiver(0,0,plane[0],plane[1])
    #define 2 points of the hyperplane that separates between plate and no plate
    a = [2*plane[1],-2*plane[1]]
    b = [-2*plane[0],2*plane[0]]
    plt.plot(a,b, color='red')
    for i in range(len(x)):
        if i==0:
            plt.text(x[i],y[i],s = 'pad')
        else:
            s = sorted(dataset.char_to_idx)[i-1]
            plt.text(x[dataset.char_to_idx[s]],y[dataset.char_to_idx[s]],s = s)
    plt.savefig(os.path.join(args.output,'weights.png'))
    '''
