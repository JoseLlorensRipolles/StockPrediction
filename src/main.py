import time
from Architecture import LSTM, GRU
from StockDataset import StockDataset
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def train():
    num_epochs = 100

    model = GRU(input_dim=1, hidden_dim=500, output_dim=1, num_layers=3).to("cuda")
    criterion = torch.nn.MSELoss(reduction="mean")
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    dataset = StockDataset(ticker="msft", train=True)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)

    hist = []

    for t in range(num_epochs):
        for samples in dataloader:
            x_train = samples["input"]
            y_train = samples["target"]
            y_train_pred = model(x_train)
            loss = criterion(y_train_pred, y_train)
            print("Epoch ", t, "MSE: ", loss.item())
            hist.append(loss.item())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    torch.save(model.state_dict(), "resources/models/model.pt")


def test_visualization(train=True):
    model = GRU(input_dim=1, hidden_dim=500, output_dim=1, num_layers=3).to("cuda")
    model.load_state_dict(torch.load("resources/models/model.pt"))
    dataset = StockDataset(ticker="msft", train=train)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)

    outputs = np.array([])
    targets = np.array([])

    model.eval()
    for samples in dataloader:
        x_train = samples["input"]
        y_train = samples["target"]
        y_train_pred = model(x_train)

        outputs = np.hstack((outputs, y_train_pred.cpu().detach().numpy().flatten()))
        targets = np.hstack((targets, y_train.cpu().detach().numpy().flatten()))

    plt.plot(targets)
    plt.plot(outputs)
    plt.show()


if __name__ == "__main__":
    train()
    test_visualization(train=True)
