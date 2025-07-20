# dataset.py

from torch.utils.data import Dataset, DataLoader
from glob import glob
import numpy as np
import torch
import h5py
import os

class SingleBeatDataset(Dataset):
    def __init__(self, data_array):
        self.data = torch.tensor(data_array, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

def load_dataloader(h5_path, batch_size=16, shuffle=False):
    with h5py.File(h5_path, 'r') as f:
        segments = f['segments'][:]  # shape: (N, 101, 3)
        segments = segments.transpose(0, 2, 1).astype(np.float32)  # → (N, 3, 101)
    dataset = SingleBeatDataset(segments)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle,
                            )
    return dataloader