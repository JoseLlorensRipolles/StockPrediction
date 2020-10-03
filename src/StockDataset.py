import torch
import pandas as pd
import numpy as np

class StockDataset(torch.utils.data.Dataset):

    TRAIN_SPLIT = 0.8
    LOOKBACK = 20

    def __init__(self, ticker, train=True):
        full_data_frame = pd.read_csv('resources/stocks/{ticker}.us.txt'.format(ticker=ticker), index_col=0, parse_dates=True)
        data_raw = full_data_frame.to_numpy()
        data = sliding_windows(data_raw, self.LOOKBACK)

        idx, length = get_split_idx_and_length(train, data, self.TRAIN_SPLIT)

        self.inputs = data[idx:idx + length,:-1]
        self.targets = data[idx:idx + length, -1]

    def __len__(self):
        return len(self.inputs.shape[0])

    def __getitem__(self, idx):
        input = self.inputs[idx]
        target = self.targets[idx]
        return {'input': input, 'target': target}


def get_split_idx_and_length(train, data, train_split):
    if train:
        idx = 0
        length = int(np.round(train_split*data.shape[0]))
    else:
        idx = int(np.round(train_split*data.shape[0]))
        length = data.shape[0] - int(np.round(train_split*data.shape[0]))
    return idx, length


def sliding_windows(data, lookback):
    windows = []
    for index in range(len(data) - (lookback - 1)): 
        windows.append(data[index: index + lookback])
    windows = np.array(windows)
    return windows