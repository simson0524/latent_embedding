# dataset.py

from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
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
    

def load_dataloader(file_directory, batch_size=16, shuffle=False):
    h5_path = sorted(glob(os.path.join(file_directory, "*.h5")))[0]  # 첫 번째 파일만 사용
    with h5py.File(h5_path, 'r') as f:
        segments = f['segments'][:]  # shape: (N, 101, 3)
        data = segments.transpose(0, 2, 1).astype(np.float32)  # → (N, 3, 101)

    dataset = SingleBeatDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader