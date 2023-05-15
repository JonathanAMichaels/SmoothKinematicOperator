import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

def preprocess_common(data):
    data = data.apply(pd.to_numeric, errors='coerce')
    total_groups = data.shape[1] // 4
    # Interpolate away NaNs so that the input to the RNN is sensible, but don't interpolate confidence
    index_array = np.hstack([np.arange(i * 4, i * 4 + 3) for i in range(total_groups)])
    data.iloc[1:, index_array] = data.iloc[1:, index_array].interpolate()
    # Instead, fill NaN confidence values with 0
    data.fillna(value=0, inplace=True)
    return data


def preprocess_data(data, norm_values=[0, 1], batch_size=64, duration=300):
    # create time slices from recording
    assert data.shape[0] > duration
    cropped_data = np.zeros((batch_size, duration, data.shape[1]), dtype=np.float32)
    for b in range(batch_size):
        t0 = np.random.randint(1, data.shape[0] - duration)
        cropped_data[b, :, :] = data.iloc[t0:t0 + duration, :].values
    normalized_data = normalize_data(cropped_data, norm_values)

    return normalized_data



def preprocess_for_test(data, norm_values=[0, 1]):
    cropped_data = np.expand_dims(np.array(data.iloc[1:, :].values, dtype=np.float32), axis=0)
    normalized_data = normalize_data(cropped_data, norm_values=norm_values)

    return normalized_data


def get_normalization_values(data):
    data = data.apply(pd.to_numeric, errors='coerce')
    m = np.nanmean(data.iloc[1:, :].values, axis=0)
    sd = np.nanstd(data.iloc[1:, :].values, axis=0)
    # don't normalize confidence
    m[3::4] = 0
    sd[3::4] = 1

    return [m, sd]


def normalize_data(data, norm_values, forward=True):
    expand_axes = [0, 1]
    m = np.expand_dims(norm_values[0], axis=[0, 1])
    sd = np.expand_dims(norm_values[1], axis=[0, 1])
    for j in expand_axes:
        m = np.repeat(m, data.shape[j], axis=j)
        sd = np.repeat(sd, data.shape[j], axis=j)
    if forward:
        normalized_data = (data - m) / sd
    else:
        normalized_data = (data * sd) + m

    return normalized_data


# Create a custom dataset
class CSVDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        inputs = np.copy(self.data[idx, :, :])
        targets = np.copy(self.data[idx, :, :])
        confidence = np.copy(self.data[idx, :, :])
        for i in range(0, confidence.shape[1], 4):
            #confidence[confidence[:, i+3] < 0.1, i+3] = 0.1
            #confidence[confidence[:, i + 3] >= 0.1, i + 3] = 1
            confidence[:, i:i+3] = np.expand_dims(confidence[:, i+3], axis=1)
            confidence[:, i+3] = 0
            inputs[:, i+3] = 0
            targets[:, i+3] = 0

        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32),\
            torch.tensor(confidence, dtype=torch.float32)

# Define the Bidirectional GRU model
class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(BiGRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)

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
        initial_loss = torch.square(output - targets) * confidence
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

