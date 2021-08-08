import os
import torch
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

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        label = self.labels[idx]
        word = str(self.words[idx])
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
    def forward(self, input):
        # input shape is (B, W)
        e = self.emb(input)
        # e shape is (B, W, E)
        u = e.sum(dim=1)
        # u shape is (B, E)
        v = self.lin(u)
        # v shape is (B, V)
        return v

def train(model, dataloader, criterion, optimizer, batch_size, device, log=False):
    model.train()
    total_loss = 0
    ncorrect = 0
    ntokens = 0
    niterations = 0
    for sample in iter(dataloader):
        X = sample['Indexes'].to(device)
        y = sample['Class'].to(device)

        model.zero_grad()
        output = model(X).squeeze(1)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        # Training statistics
        total_loss += loss.item()
        ncorrect += ((output > 0.5) == y).sum().item()
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
            output = model(X).squeeze(1)
            loss = criterion(output, y)
            total_loss += loss.item()
            ncorrect += ((output > 0.5) == y).sum().item()
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
    epochs = 1

    dataset = PlateClassificationDataset(args.data_path)
    n=len(dataset)
    n_train = int(n*0.8)
    n_val = n-n_train

    train_sample, val_sample = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    train_sampler = RandomSampler(train_sample)
    val_sampler = RandomSampler(val_sample)
    train_dataloader = DataLoader(train_sample, sampler = train_sampler, batch_size=bs, num_workers = num_workers, collate_fn = collate_fn_padd)
    val_dataloader = DataLoader(val_sample, sampler = val_sampler, batch_size=bs, num_workers = num_workers, collate_fn = collate_fn_padd)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = PlateNet(num_embeddings = len(dataset.vocabulary) + 1, embedding_dim = 2)

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
    print(model.emb.weight.cpu().data.numpy())
    for char in sorted(dataset.char_to_idx):
        print(f'{char} : {model.emb.weight[dataset.char_to_idx[char]].cpu().data.numpy()}')
    print(model.lin.weight)

    weights = model.emb.weight.cpu().data.numpy()
    x = weights.transpose(1,0)[0]
    y = weights.transpose(1,0)[1]
    print(x,y)
    plt.plot(x,y)
    plt.text(x,y,s = sorted(dataset.char_to_idx).keys())
    plt.savefig(os.path.join(args.output,'weigts.png'))
    torch.save(model.state_dict(), os.path.join(args.output,'weigts.pt'))
