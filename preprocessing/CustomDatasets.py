from torch.utils.data import Dataset
import pandas as pd
import os.path
import torchaudio
import torch
import torch.nn as nn

"""
CustomDataset1

INPUT
X: list of data (tensors)
y: list of labels

OUTPUTS
waveform: list waveforms (tensors)
label = list of labels (tensors)
"""

class CustomDataset1(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        waveform = torch.Tensor(self.X[idx])
        label = torch.Tensor([self.y[idx]])
        return waveform, label

"""
CustomDataset2

INPUT
X: list of data (tensors)
y: list of labels
n: list of names (for utterances who have been clipped/augmented)

OUTPUTS
waveform: list waveforms (tensors)
label: list of labels (tensors)
name: list of names
"""
class CustomDataset2(Dataset):
    def __init__(self, X, y, n):
        self.X = X
        self.y = y
        self.n = n
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        waveform = torch.Tensor(self.X[idx])
        label = torch.Tensor([self.y[idx]])
        name = self.n[idx]
        return waveform, label, name