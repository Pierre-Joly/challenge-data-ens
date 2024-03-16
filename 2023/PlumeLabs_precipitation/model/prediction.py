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
    
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.lstm = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        _, hidden = self.lstm(x)  # hidden is a tuple (hidden_state, cell_state)
        return hidden

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.lstm = nn.GRU(input_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        prediction = self.relu(self.out(output))
        return prediction, hidden

class Model(nn.Module):
    def __init__(self,):
        super(Model, self).__init__()
        self.encoder_input_size = 144
        self.encoder_hidden_size = 64
        self.decoder_input_size = 1
        self.decoder_output_size = 1
        self.cnn = CNN()
        self.encoder = Encoder(self.encoder_input_size, self.encoder_hidden_size)
        self.decoder = Decoder(self.decoder_input_size, self.encoder_hidden_size, self.decoder_output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # Process input through CNN and then the encoder
        x = self.cnn(x)
        encoder_hidden = self.encoder(x)

        # Initialize decoder input (e.g., start with zeros)
        decoder_input = torch.zeros(batch_size, 1, 1).to(x.device)

        # Collect decoder outputs
        outputs = []
        for _ in range(8):  # Assuming you want to generate 8 time steps
            decoder_output, encoder_hidden = self.decoder(decoder_input, encoder_hidden)
            outputs.append(decoder_output)
            decoder_input = decoder_output  # Use output as input for next step

        return torch.cat(outputs, dim=1).view(batch_size, -1)
    
def benchmark(x_test_dir, model):
    benchmark_prediction = []
    benchmark_ids = []
    n = len(os.listdir(x_test_dir))
    for i, file in enumerate(os.listdir(x_test_dir)):
        x_test = np.load(f'{x_test_dir}/{file}')
        y_bench = model(torch.tensor(x_test['data'], dtype=torch.float32).view(1, 4, 128, 128).to(device)).view(8).cpu().detach().numpy()
        print(y_bench)
        for j in range(8):
            benchmark_prediction.append(y_bench[j])
            benchmark_ids.append(x_test['target_ids'][j])
        print(f"file : {i}/{n}")
    return pd.DataFrame({
        'ID': benchmark_ids, 
        'TARGET': benchmark_prediction
    })

device = torch.device('mps')

model = Model()
model.to(device)

model.load_state_dict(torch.load('model/model.pth'))

benchmark_prediction = benchmark('data/x_test', model)

benchmark_prediction.to_csv('data/y_bench.csv', index=False)