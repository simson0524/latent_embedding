# inference.py

from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from random import sample
from glob import glob
from dataset import load_dataloader
import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
import os

def load_beats_from_h5(file_path): # 필요한가..?
    with h5py.File(file_path, 'r') as h5f:
        segment = h5f['segment'][:]
        if segment.shape[0] == 101:
            segment = np.transpose(segment, (1, 0))
    return segment.astype(np.float32)


def extract_latents(model, dataloader, device):
    model.to(device)
    model.eval()
    all_mu, all_logvar, all_z, all_standardized = [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                inputs = batch[0].to(device)
            else:
                inputs = batch.to(device)

            mu, logvar = model.encoder(inputs)
            z = model.encoder.reparameterize(mu, logvar)
            std = torch.exp(0.5 * logvar)
            standardized = (z - mu) / std

            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())
            all_z.append(z.cpu())
            all_standardized.append(standardized.cpu())

    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)
    all_z = torch.cat(all_z, dim=0)
    all_standardized = torch.cat(all_standardized, dim=0)

    return all_mu, all_logvar, all_z, all_standardized


def plot_original_vs_reconstruction(original, reconstruction, idx, plot_save_dir="./inference_results"):
    os.makedirs(plot_save_dir, exist_ok=True)
    channels = original.shape[0]
    fig, axs = plt.subplots(channels, 1, figsize=(10, 5 * channels))
    if channels == 1:
        axs = [axs]
    for ch in range(channels):
        axs[ch].plot(original[ch], label='Original')
        axs[ch].plot(reconstruction[ch], label='Reconstructed', linestyle='dashed')
        axs[ch].set_title(f'Channel {ch}')
        axs[ch].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_save_dir, f"segment_{idx}.png"))
    plt.close()


def plot_3d_latent_with_color_strips(embeddings, save_dir, title='latent', point_size=5, outlier_threshold=2.5):
    latent_dim = embeddings.shape[1]
    combos = list(combinations(range(latent_dim), 3))
    num_combos = len(combos)

    color_lines = []

    for i, (a, b, c) in enumerate(combos):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x, y, z_ = embeddings[:, a], embeddings[:, b], embeddings[:, c]

        # plot outliers as red
        outlier_mask = (x.abs() > outlier_threshold) | (y.abs() > outlier_threshold) | (z_.abs() > outlier_threshold)
        colors = np.where(outlier_mask, 'red', 'blue')
        color_lines.append(colors)

        # 3D visualization
        red_ratio = outlier_mask.sum().item() / len(embeddings) * 100
        ax.scatter(x, y, z_, c=colors, s=point_size)
        ax.set_xlabel(f'embedding[{a}]')
        ax.set_ylabel(f'embedding[{b}]')
        ax.set_zlabel(f'embedding[{c}]')
        ax.set_title(f"{title}: dims ({a}, {b}, {c})\nOutlier: {red_ratio:.2f}%")
        
        # ✅ Save each 3D scatter plot
        plot_path = os.path.join(save_dir, f'{title}_3d_{a}_{b}_{c}.png')
        plt.savefig(plot_path)
        plt.close()

    # 2D visualization
    fig, ax = plt.subplots(figsize=(360, num_combos))
    color_map = {'red': 1, 'blue': 0}
    color_matrix = np.array([[color_map[c] for c in line] for line in color_lines])

    cmap = ListedColormap(['blue', 'red'])
    ax.imshow(color_matrix, cmap=cmap, aspect='auto')

    ax.set_yticks(range(num_combos))
    ax.set_yticklabels([f'({a},{b},{c})' for (a, b, c) in combos])
    ax.set_xlabel("Sample Index")
    ax.set_title(f"{title}: Outlier Strip View (per 3D Combo)")
    
    strip_path = os.path.join(save_dir, f'{title}_strip_view.png')
    plt.tight_layout()
    plt.savefig(strip_path)
    plt.close()

    outlier_indices = np.where(color_matrix.max(axis=0) == 1)[0]  # 빨간색이 하나라도 있는 column
    return embeddings, outlier_indices