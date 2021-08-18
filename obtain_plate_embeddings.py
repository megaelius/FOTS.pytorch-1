import os
import torch
import pickle
import numpy as np

from scipy import spatial
from torch.nn.utils.rnn import pad_sequence

def index_chars(word,char_to_idx):
    result = []
    for char in word:
        if char in char_to_idx:
            result.append(char_to_idx[char])
        else:
            result.append(0)
    return result

class PlateNet(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.emb = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.lin = torch.nn.Linear(embedding_dim, 1, bias=False)
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

if __name__ == '__main__':
    #Load plateset
    with open('../../Data/PlateSet.pkl', 'rb') as f:
        plateset = sorted(list(pickle.load(f)))

    #Load char to index dictionary
    with open('../out/Model_1/char_to_idx.pkl', 'rb') as handle:
        char_to_idx = pickle.load(handle)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    platenet = torch.load('../out/Model_1/weights.pt', map_location = device)
    platenet.eval()
    platenet.to(device)

    with torch.no_grad():
        plates_idx = [torch.tensor(index_chars(w,char_to_idx)) for w in plateset]
        sequences = pad_sequence(plates_idx, batch_first = True).to(device)
        input_lengths = [len(s) for s in sequences]
        output = platenet.emb(sequences).sum(dim=1).numpy()
    tree = spatial.KDTree(output)

    with torch.no_grad():
        plate_idx = [torch.tensor(index_chars(w,char_to_idx)) for w in ['JA6']]
        seq = pad_sequence(plate_idx, batch_first = True).to(device)
        emb = platenet.emb(seq).sum(dim=1).numpy()
    print(emb)
    d,indexes = tree.query(emb[0],k=5)
    print(d,indexes)
    for i in indexes:
        print(plateset[i])
