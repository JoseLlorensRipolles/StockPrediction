import time
from architecture import LSTM, GRU
from stock_dataset import StockDataset
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def train():
    num_epochs = 100

    model = GRU(input_dim=1, hidden_dim=500, output_dim=1, num_layers=3).to("cuda")
    criterion = torch.nn.MSELoss(reduction="mean")
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    dataset = StockDataset(ticker="msft", is_train=True)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)

    for t in range(num_epochs):
        for samples in dataloader:
            x_train = samples["input"]
            y_train = samples["target"]
            y_train_pred = model(x_train)
            loss = criterion(y_train_pred, y_train)
            print("Epoch ", t, "MSE: ", loss.item())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    torch.save(model.state_dict(), "resources/models/model.pt")


def prediction_visualization(train=True):
    model = GRU(input_dim=1, hidden_dim=500, output_dim=1, num_layers=3).to("cuda")
    model.load_state_dict(torch.load("resources/models/model.pt"))
    dataset = StockDataset(ticker="msft", is_train=train)
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

    outputs = dataset.scaler.inverse_transform(outputs.reshape(-1, 1))
    targets = dataset.scaler.inverse_transform(targets.reshape(-1, 1))

    plt.figure(figsize=(15, 6))
    plt.plot(targets, label="Ground truth")
    plt.plot(outputs, label="Prediction")
    plt.title("One day ahead prediction")
    plt.savefig("resources/prediction.png")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train()
    prediction_visualization(train=False)
