import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class StockDataset(torch.utils.data.Dataset):

    TRAIN_SPLIT = 0.8
    LOOKBACK = 20

    def __init__(self, ticker, train=True):
        full_data_frame = pd.read_csv('resources/stocks/{ticker}.us.txt'.format(ticker=ticker), index_col=0, parse_dates=True)
        price = full_data_frame[['Close']]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))
        data_raw = price.to_numpy()
        data = sliding_windows(data_raw, self.LOOKBACK)

        idx, length = get_split_idx_and_length(train, data, self.TRAIN_SPLIT)

        self.inputs = torch.FloatTensor(data[idx:idx + length,:-1, :]).to('cuda')
        self.targets = torch.FloatTensor(data[idx:idx + length, -1, :]).to('cuda')

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        sample_input = self.inputs[idx]
        sample_target = self.targets[idx]
        return {'input': sample_input, 'target': sample_target}


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


if __name__ == "__main__":
    dataset = StockDataset('msft')
    print(dataset.__getitem__(1)['input'])
    print(dataset.__getitem__(1)['target'])