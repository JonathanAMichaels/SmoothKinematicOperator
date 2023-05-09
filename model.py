import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# Preprocess the data
def preprocess_data(data, norm_constant=1000):
    data = data.apply(pd.to_numeric, errors='coerce')
    total_groups = data.shape[1] // 4
    # Create the index array with the desired pattern
    index_array = np.hstack([np.arange(i * 4, i * 4 + 3) for i in range(total_groups)])
    data.iloc[1:, index_array] = data.iloc[1:, index_array].interpolate()
    data.fillna(value=0, inplace=True)
    # create time slices from recording
    duration = 300
    batches = int(data.shape[0] / duration * (duration / 10))
    cropped_data = np.zeros((batches, duration, data.shape[1]), dtype=np.float32)
    for b in range(batches):
        t0 = np.random.randint(1, data.shape[0] - duration)
        cropped_data[b, :, :] = data.iloc[t0:t0 + duration, :].values

    normalized_data = cropped_data / norm_constant
    return normalized_data


def preprocess_for_test(data, norm_constant=1000):
    data = data.apply(pd.to_numeric, errors='coerce')
    total_groups = data.shape[1] // 4
    # Create the index array with the desired pattern
    index_array = np.hstack([np.arange(i * 4, i * 4 + 3) for i in range(total_groups)])
    data.iloc[1:, index_array] = data.iloc[1:, index_array].interpolate()
    data.fillna(value=0, inplace=True)
    normalized_data = np.expand_dims(np.array(data.iloc[1:, :].values, dtype=np.float32) / norm_constant, axis=0)
    return torch.tensor(normalized_data, dtype=torch.float32)

# Create a custom dataset
class CSVDataset(Dataset):
    def __init__(self, data, norm_constant=1000):
        self.data = data
        self.norm_constant = norm_constant

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        inputs = np.copy(self.data[idx, :, :])
        targets = np.copy(self.data[idx, :, :])
        confidence = np.copy(self.data[idx, :, :])
        for i in range(0, confidence.shape[1], 4):
            confidence[confidence[:, i+3] < 0, i+3] = 0
            confidence[:, i:i+3] = np.expand_dims(confidence[:, i+3], axis=1) * self.norm_constant
            confidence[:, i+3] = 0

        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32),\
            torch.tensor(confidence, dtype=torch.float32)

# Define the Bidirectional GRU model
class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):
        super(BiGRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Custom loss function
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, targets, confidence):
        initial_loss = (torch.abs(output - targets) * confidence) * 100
        return torch.mean(initial_loss)


def plot_heatmaps(array1, array2):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    im1 = axes[0].imshow(array1, cmap='viridis', aspect='auto', interpolation='none')
    axes[0].set_title("Array 1")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(array2, cmap='viridis', aspect='auto', interpolation='none')
    axes[1].set_title("Array 2")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()