import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import json
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import heapq
from collections import defaultdict, Counter

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def huffman_encoding(indices):
    # frequencies = Counter(indices)
    # huffman_tree = build_huffman_tree(frequencies)
    # huff_dict = {symbol: code for symbol, code in huffman_tree}
    huff_dict = json.load(open("huff_dict.json", "r"))
    # Each codebook size should have its corresponding Huffman dictionary. Here, we use a simple example with codebook size=4.
    encoded_bits = ''.join(huff_dict[str(symbol)] for symbol in indices)
    return encoded_bits


def huffman_decoding(encoded_data, huff_dict):
    reversed_dict = {code: symbol for symbol, code in huff_dict.items()}
    symbol = ''
    decoded_indices = []
    for bit in encoded_data:
        symbol += bit
        if symbol in reversed_dict:
            decoded_indices.append(reversed_dict[symbol])
            symbol = ''
    return decoded_indices


def build_huffman_tree(frequencies):
    heap = [[weight, [symbol, ""]] for symbol, weight in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))