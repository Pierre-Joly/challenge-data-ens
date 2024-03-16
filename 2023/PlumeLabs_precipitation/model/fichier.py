# Let's first define commun class/function needed
import torch
import torch.nn as nn
# Importing the libraries
import os
import numpy as np
import pandas as pd

# First i need a custom dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, x_dir, y_dir, transform=None):
        self.x_dir = x_dir
        self.y = pd.read_csv(f'data/{y_dir}.csv')

    def __len__(self):
        return len(os.listdir(f'data/{self.x_dir}'))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        L = np.load(f'data/{self.x_dir}/{idx}.npz')
        x = L['data']
        y = self.y[self.y['ID'].isin(L['target_ids'])]['TARGET'].values

        # Convert to PyTorch tensors
        x = torch.from_numpy(x).type(torch.float32)  # Assuming x is a numpy array
        y = torch.tensor(y, dtype=torch.float32)     # Adjust dtype as per your data

        return x, y
    
class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

# Then i need a training loop
def train_loop(model, loss_fn, optimizer, train_loader, device):
    size = len(train_loader.dataset)
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Let's validate the model with evaluation metrics : Mean Squared Logarithmic Error
def test_loop(model, loss_fn, test_loader, device):
    size = len(test_loader.dataset)
    test_loss = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= size
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

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
        self.relu = nn.LeakyReLU()

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
        prediction = self.out(output)
        print(f"prediction : {torch.isnan(prediction).any()}")
        return self.relu(prediction), hidden

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
        print(f"before : {torch.isnan(x).any()}")
        x = self.cnn(x)
        print(f"cnn : {torch.isnan(x).any()}")
        encoder_hidden = self.encoder(x)
        print(f"encoder : {torch.isnan(x).any()}")

        # Initialize decoder input (e.g., start with zeros)
        decoder_input = torch.zeros(batch_size, 1, 1).to(x.device)

        # Collect decoder outputs
        outputs = []
        for _ in range(8):  # Assuming you want to generate 8 time steps
            decoder_output, encoder_hidden = self.decoder(decoder_input, encoder_hidden)
            outputs.append(decoder_output)
            decoder_input = decoder_output  # Use output as input for next step

        return torch.cat(outputs, dim=1).view(batch_size, -1)

# Hyperparameters
batch_size = 64
epochs = 5
learning_rate = 1e-3
min_learning_rate = 1e-4
loss_fn = RMSLELoss()

# Device
device = torch.device('mps')

# Training data
split = 0.8
dataset = Dataset('x_train', 'y_train')

# Split data
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * split), len(dataset) - int(len(dataset) * split)])

# Data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = Model()
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_learning_rate)

# Prediction
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(model, loss_fn, optimizer, train_loader, device)
    test_loop(model, loss_fn, test_loader, device)
    scheduler.step()

print("Done!")

# Save the model
torch.save(model.state_dict(), 'model/model.pth')

