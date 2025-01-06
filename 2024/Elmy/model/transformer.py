# Import
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import math
import matplotlib.pyplot as plt

# Get Data
X_train = pd.read_csv('../ELMY/data/X_train.csv')
Y_train = pd.read_csv('../ELMY/data/Y_train.csv')

# Get the hours and months from the DELIVERY_START column
hours = [(int(X_train["DELIVERY_START"].iloc[i][11:13]) + 1 ) % 24 for i in range(len(X_train["DELIVERY_START"]))]
months = [int(X_train["DELIVERY_START"].iloc[i][5:7]) for i in range(len(X_train["DELIVERY_START"]))]
X_train["hours"] = hours
X_train["months"] = months
X_train = X_train.drop(columns=["DELIVERY_START"])
 
# Normalize the data
X_train = (X_train - X_train.mean()) / X_train.std()

# Impute the missing values
imputer = KNNImputer(n_neighbors=5)
imputer.fit_transform(X_train)
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)

# Drop the hours and months columns
X_train.drop(columns=["hours", "months"], inplace=True)

# Normalize the data
X_train = (X_train - X_train.mean()) / X_train.std()

# Apply PCA
pca = PCA(n_components=0.95)
pca.fit(X_train)
X_pca = pca.transform(X_train)

# Add hours and months with sinusoidal encoding
hours = np.array(hours)
months = np.array(months)
hours_sin = np.sin(2 * np.pi * hours / 24)
hours_cos= np.cos(2 * np.pi * hours / 24)
months_sin = np.sin(2 * np.pi * hours / 12)
months_cos = np.cos(2 * np.pi * hours / 12)
X_pca = np.concatenate((X_pca, hours_sin.reshape(-1, 1), hours_cos.reshape(-1, 1), months_sin.reshape(-1, 1), months_cos.reshape(-1, 1)), axis=1)

# Define the model

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # Adjust for odd d_model: compute cosine for up to the second last if d_model is odd
        pe[:, 1::2] = torch.cos(position * div_term)[:,:d_model//2]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[ : x.size(1), 0, :] 
       
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_encoder_layers, output_dim, window_size, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.pos_encoder = PositionalEncoding(model_dim, window_size)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.dense_layer = nn.Linear(model_dim, model_dim)  # Additional dense layer
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(model_dim, output_dim)  # Adjusted for potential multiple outputs
        self.relu = nn.ReLU()
        
        # For a single output per sequence, ensure output_dim is 1 and adjust forward accordingly
        self.output_dim = output_dim
        self.window_size = window_size

    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        encoded = self.transformer_encoder(src)
        encoded = self.relu(self.dense_layer(encoded))
        output = self.output_layer(encoded)
        output = output[:, -1, :]
        output = torch.sigmoid(output)
        
        return output
    
def padding(data, input_window):
    # Assuming padding adds necessary elements to make the data compatible with window size
    # Update this function if necessary to efficiently pad your data
    return torch.nn.functional.pad(data, (0, 0, input_window//2, input_window//2), "constant", 0)

def get_data(x, y, input_window, device):
    """Optimized version of splitting data into sequences of equal length."""

    # Efficiently pad the data
    x = padding(x, input_window)
    y = padding(y, input_window)
    
    # Utilize tensor operations to create sequences without explicit loops
    x_seq = torch.stack([x[i:i+input_window] for i in range(len(x) - input_window + 1)]).to(device)
    y_seq = torch.stack([y[i+input_window//2] for i in range(len(y) - input_window + 1)]).to(device)

    return x_seq, y_seq

def train(train_loader, model, criterion, optimizer, scheduler, epoch, device):
    model.train()  # Set model to training mode
    total_loss = 0
    start_time = time.time()

    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # Clip gradient norm
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 10 == 0:  # Adjust print frequency as needed
            elapsed = time.time() - start_time
            print('| Epoch: {:3d} | Batch: {:5d} | Loss: {:.6f} | Time: {:.2f}s'.format(
                epoch, batch_idx, total_loss / (batch_idx + 1), elapsed))
            start_time = time.time()  # Resetting start time
        
        scheduler.step()  # Update learning rate after each batch
    print(f'LR: {scheduler.get_last_lr()}')
    # Update the learning rate after each epoch

    return total_loss / len(train_loader)

def custom_loss(output, target):
    weight = torch.abs(target)
    target = (torch.sign(target) > 0).float()
    bce = (target*torch.log(output+1e-9) + (1-target)*torch.log(1-output+1e-9))
    return torch.mean(bce*weight*-1)

def evaluate(test_loader, model, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_weighted_accuracy = 0.0
    total_weight = 0.0

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            y_pred = output
            y_test = targets
            y_pred = torch.sign(y_pred - 0.5)
            correct_predictions_proportion = (torch.sign(y_pred*y_test)>0).float()
            weights = torch.abs(y_test)
            total_weighted_accuracy += torch.sum(correct_predictions_proportion * weights)
            total_weight += torch.sum(weights)
    
    score = total_weighted_accuracy / total_weight
    score = score.cpu().numpy()
    print(f'Weighted Accuracy: {score:.2f}')
    return score

# Define the hyperparameters

Y_train = Y_train['spot_id_delta']

N = 5
window_size = 2*N+1

device = torch.device("mps")

# Convert numpy arrays to PyTorch tensors and ensure they are of dtype float32
x = torch.from_numpy(np.array(X_pca)).float()
y = torch.from_numpy(np.array(Y_train)).float().view(-1, 1)

x_seq, y_seq = get_data(x, y, input_window=window_size, device=device)

dataset = CustomDataset(x_seq, y_seq)

loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

input_dim = x_seq.size(2)  # x_train_seq is of shape [samples, window, features]
model_dim = 512
num_heads = 16
num_encoder_layers = 2
output_dim = 1

model = TransformerModel(input_dim, model_dim, num_heads, num_encoder_layers, output_dim, window_size).to(device)

# binary cross entropy loss
criterion = custom_loss
lr = 1e-4 # learning rate
# I want L2 and L1 regularization with standard values
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
#optimizer = torch.optim.SGD(model.parameters(), lr=lr)
epochs = 50 # Number of epochs
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = lr, epochs=epochs, steps_per_epoch=len(loader)+1)
n = 10 # Number of epochs between each validation

Loss = np.zeros(epochs)

# Train the model
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    loss = train(train_loader=loader, model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, epoch=epoch, device=device)
    Loss[epoch-1] = loss

    scheduler.step()

# Plot the loss
plt.figure()
plt.plot(Loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------
# Prediction on test set
# ------------------------------------------------------------------------------------------------------------------------------------

X_test = pd.read_csv('../ELMY/data/X_test.csv')

# Analyse for each column the correlation with the hours and months to replace NA by meaninful values

hours = np.array([(int(X_test["DELIVERY_START"].iloc[i][11:13]) + 1 ) % 24 for i in range(len(X_test["DELIVERY_START"]))])
months = np.array([int(X_test["DELIVERY_START"].iloc[i][5:7]) for i in range(len(X_test["DELIVERY_START"]))])
X_test["hours"] = hours
X_test["months"] = months
delivery_start = X_test["DELIVERY_START"]
X_test = X_test.drop(columns=["DELIVERY_START"])

X_test = (X_test - X_test.mean()) / X_test.std()
imputer = KNNImputer(n_neighbors=5)
imputer.fit_transform(X_test)
X_test = pd.DataFrame(imputer.fit_transform(X_test), columns=X_test.columns)
X_test.drop(columns=["hours", "months"], inplace=True)

X_test = (X_test - X_test.mean()) / X_test.std()
X_test = pca.transform(X_test)

hours_cos = np.cos(2 * np.pi * hours / 24)
hours_sin = np.sin(2 * np.pi * hours / 24)
months_cos = np.cos(2 * np.pi * months / 12)
months_sin = np.sin(2 * np.pi * months / 12)
X_test = np.concatenate((X_test, hours_cos.reshape(-1, 1), hours_sin.reshape(-1, 1), months_cos.reshape(-1, 1), months_sin.reshape(-1, 1)), axis=1)

X_test = torch.from_numpy(X_test).float()
X_test = padding(X_test, window_size)
X_test_seq = torch.stack([X_test[i:i+window_size] for i in range(len(X_test) - window_size + 1)]).to(device)

dataset = CustomDataset(X_test_seq, torch.zeros(X_test_seq.size(0), window_size, 1))
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

model.eval()

with torch.no_grad():
    predictions = []
    for data, _ in loader:
        data = data.to(device)
        output = model(data)
        y_pred = torch.sign(output-0.5)
        predictions.append(y_pred.cpu().numpy())

predictions = np.concatenate(predictions, axis=0)

prediction = pd.DataFrame(columns=["DELIVERY_START", "spot_id_delta"])
prediction["DELIVERY_START"] = delivery_start
prediction["spot_id_delta"] = predictions

# Save the predictions to a CSV file
prediction.to_csv('submission_model.csv', index=False)

# Plot the predictions
plt.figure()
plt.plot(prediction["spot_id_delta"])
plt.xlabel('Time')
plt.ylabel('predictions')
plt.show()

