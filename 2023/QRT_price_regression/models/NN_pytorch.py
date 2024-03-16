import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

class Net(nn.Module):
    def __init__(self, input_features):
        super(Net, self).__init__()
        self.batch_norm = nn.BatchNorm1d(input_features)
        self.fc1 = nn.Linear(input_features, 64)  # input_features should be the number of features in your dataset
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Output layer for regression

    def forward(self, x):
        x = self.batch_norm(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))*5
        return x
    
def metric(output, y):
    return spearmanr(output, y).correlation

def preprocess(X, Y=None):
    X['COUNTRY'] = X['COUNTRY'].apply(lambda x: -1 if x == 'FR' else (1 if x == 'DE' else 0))
    X = X.apply(lambda x: x.fillna(x.mean()), axis=0)
    if Y is not None:
        Y = Y['TARGET']
        return X, Y
    return X
    
X_train = pd.read_csv('data/X_train.csv')
Y_train = pd.read_csv('data/Y_train.csv')
X_train, Y_train = preprocess(X_train, Y_train)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)
input_features = len(X_train.columns)

device = torch.device('mps')
epochs = 100
model = Net(input_features).to(device)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.AdamW(model.parameters(), lr=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max = epochs, eta_min = 1e-4)

# Convert your dataset to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32).to(device)

# Training loop
for epoch in range(epochs):  # Number of epochs
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, Y_train_tensor)
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
output_train = model(X_train_tensor).cpu().detach().numpy()
output_test = model(X_test_tensor).cpu().detach().numpy()

print('Spearman correlation for the train set: {:.1f}%'.format(100 * metric(output_train, Y_train)))
print('Spearman correlation for the test set: {:.1f}%'.format(100 * metric(output_test, Y_test)))