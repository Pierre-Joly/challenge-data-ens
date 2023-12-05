import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        f1, f2, f3, f4 = 16, 16, 8, 4
        self.conv1 = nn.Conv2d(1, f1, 3)
        self.batchnorm1 = nn.BatchNorm2d(f1)
        self.conv2 = nn.Conv2d(f1, f2, 3)
        self.batchnorm2 = nn.BatchNorm2d(f2)
        self.conv3 = nn.Conv2d(f2, f3, 3)
        self.batchnorm3 = nn.BatchNorm2d(f3)
        self.conv4 = nn.Conv2d(f3, f4, 3)
        self.batchnorm4 = nn.BatchNorm2d(f4)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(4, -1, 1, 128, 128)
        x = [self.pool(self.relu(self.conv1(x[i]))) for i in range(4)]
        x = [self.batchnorm1(x[i]) for i in range(4)]
        x = [self.pool(self.relu(self.conv2(x[i]))) for i in range(4)]
        x = [self.batchnorm2(x[i]) for i in range(4)]
        x = [self.pool(self.relu(self.conv3(x[i]))) for i in range(4)]
        x = [self.batchnorm3(x[i]) for i in range(4)]
        x = [self.pool(self.relu(self.conv4(x[i]))) for i in range(4)]
        x = [self.batchnorm4(x[i]) for i in range(4)]
        x = torch.stack(x)
        x = x.view(-1, 4, 144)
        return x

class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.gru = nn.GRU(144, 64, 1, batch_first=True)
        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.cnn = CNN()

    def forward(self, x):
        x = self.cnn(x)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return x
    
def benchmark(x_test_dir, model):
    benchmark_prediction = []
    benchmark_ids = []
    n = len(os.listdir(x_test_dir))
    for i, file in enumerate(os.listdir(x_test_dir)):
        x_test = np.load(f'{x_test_dir}/{file}')
        y_bench = model(torch.tensor(x_test['data'], dtype=torch.float32).to(device)).cpu().detach().numpy()
        benchmark_prediction.append(y_bench)
        benchmark_ids.append(x_test['target_ids'])
        print(benchmark_prediction)
        print(benchmark_ids)
        exit()
        print(f"file : {i}/{n}")
    return pd.DataFrame({
        'ID': np.concatenate(benchmark_ids), 
        'TARGET': np.concatenate(benchmark_prediction)
    })

device = torch.device('mps')

model = GRU()
model.to(device)

model.load_state_dict(torch.load('model/model.pth'))

benchmark_prediction = benchmark('data/x_test', model)
benchmark_prediction.to_csv('data/y_bench.csv', index=False)