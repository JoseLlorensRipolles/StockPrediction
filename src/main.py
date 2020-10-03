import time
from Architecture import LSTM
from StockDataset import StockDataset
import torch
import numpy as np
from torch.utils.data import DataLoader

num_epochs = 10

model = LSTM(input_dim=1, hidden_dim=500, output_dim=1, num_layers=3)
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
dataset = StockDataset(ticker='msft', train=True)
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

hist = np.zeros(num_epochs)
start_time = time.time()
lstm = []

for t in range(num_epochs):
    for sample in dataloader:
        x_train = sample['input']
        y_train = sample['target']
        y_train_pred = model(x_train)    
        loss = criterion(y_train_pred, y_train)
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()    
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    
training_time = time.time()-start_time
print("Training time: {}".format(training_time))