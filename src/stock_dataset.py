import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class StockDataset(torch.utils.data.Dataset):

    TRAIN_SPLIT = 0.8
    LOOKBACK = 20

    def __init__(self, ticker, is_train=True):
        full_data_frame = pd.read_csv(
            "resources/stocks/{ticker}.us.txt".format(ticker=ticker),
            index_col=0,
            parse_dates=True,
        )
        price = full_data_frame[["Close"]]

        data_raw = price.to_numpy()
        data_raw = self.get_split(is_train, data_raw)
        data_raw = self.min_max_scaling(data_raw)

        windows = sliding_windows(data_raw, self.LOOKBACK)
        windows = torch.FloatTensor(windows).to("cuda")

        self.inputs = windows[:, :-1, :]
        self.targets = windows[:, -1, :]

    def get_split(self, is_train, data_raw):
        idx, length = get_split_idx_and_length(is_train, data_raw, self.TRAIN_SPLIT)
        data_raw = data_raw[idx : idx + length, :]
        return data_raw

    def min_max_scaling(self, data):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler = scaler
        return scaler.fit_transform(data)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        sample_input = self.inputs[idx]
        sample_target = self.targets[idx]
        return {"input": sample_input, "target": sample_target}


def get_split_idx_and_length(is_train, data, train_split):
    if is_train:
        idx = 0
        length = int(np.round(train_split * data.shape[0]))
    else:
        idx = int(np.round(train_split * data.shape[0]))
        length = data.shape[0] - int(np.round(train_split * data.shape[0]))
    return idx, length


def sliding_windows(data, lookback):
    windows = []
    for index in range(len(data) - (lookback - 1)):
        windows.append(data[index : index + lookback])
    windows = np.array(windows)
    return windows


if __name__ == "__main__":
    dataset = StockDataset("msft")
    print(dataset.__getitem__(1)["input"])
    print(dataset.__getitem__(1)["target"])
