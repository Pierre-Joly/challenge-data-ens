import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import average_precision_score, classification_report
import torchvision
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class CustomDataset(Dataset):
    def __init__(self, X, y):
        # Convert each sequence in each row of X to a tensor
        self.data = [torch.tensor(X.iloc[idx].tolist(), dtype=torch.float32) for idx in range(len(X))]
        self.labels = torch.tensor(y.values, dtype=torch.float32).squeeze()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get the sequence data and label for the given index
        sequence_data = self.data[idx]
        label = self.labels[idx].unsqueeze(0)
        return sequence_data, label

class FraudTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_classes, dropout=0.1):
        super(FraudTransformer, self).__init__()
        self.categorical_columns = ['item', 'make', 'model']
        self.numerical_columns = ['cash_price', 'Nbr_of_prod_purchas']
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=d_model)
        d_model = d_model*len(self.categorical_columns)+len(self.numerical_columns)
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.fc = nn.Linear(in_features=d_model, out_features=num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Forward pass through the FraudTransformer network.

        Parameters:
        - x (torch.Tensor): The input tensor.

        Returns:
        - x (torch.Tensor): The output tensor after processing.
        """
        # Shape: [B, 5, 24]
        x_emb = self.embedding(x[:, :len(self.categorical_columns)])  # Shape: [B, 3, 24, 64]
        x_emb = x_emb.view(x_emb.size(0), -1, x_emb.size(2)) # Shape: [B, 256, 24]
        x_num = x[:, len(self.categorical_columns):] # Shape: [B, 2, 24]
        x = torch.cat((x_emb, x_num), dim=1) # Shape: [B, 258, 24]
        x = x.transpose(2, 1) # Shape: [B, 24, 258]
        x = self.pos_encoder(x)
        x = self.layer_norm1(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.layer_norm2(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        data, target = batch
        data, target = data.long().to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            data, target = batch
            data, target = data.long().to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(test_loader)

def create_mapping_dict(df, column_name):
    unique_categories = df[column_name].explode().unique()
    category_to_code = {category: idx for idx, category in enumerate(unique_categories)}
    return category_to_code

def apply_mapping(df, mapping_dict, column_name):
    df[column_name] = df[column_name].apply(lambda x: [mapping_dict[category] if category in mapping_dict.keys() else -1 for category in x])
    return df

def preprocess_data(df, y=None, dict=None):
    # Drop unnecessary columns
    columns_to_drop = ['ID']
    df = df.drop(columns_to_drop, axis=1)

    # Map categorical columns to integer IDs
    categorical_columns = ['item', 'make', 'model']
    numerical_columns = ['cash_price', 'Nbr_of_prod_purchas']
    columns = categorical_columns + numerical_columns

    # Pad sequences for each categorical feature group
    max_seq_length = 24
    for col in columns:
        df[col] = df[[col + str(i) for i in range(1, max_seq_length + 1)]].values.tolist()
        if col in categorical_columns:
            df[col] = df[col].apply(lambda x: x + [-1] * (max_seq_length - len(x)))  # Padding with -1

    # Combine categorical sequences with numerical data
    df = df[columns]

    if dict is None:
        dict = {}

    for col in columns:
        if col in categorical_columns:
            if col not in dict.keys():
                dict[col] = create_mapping_dict(df, col)
                df = apply_mapping(df, dict[col], col)
            else:
                df = apply_mapping(df, dict[col], col)
        else:
            df[col] = df[col].apply(lambda x: [float(i) for i in x])
            df[col] = df[col].apply(lambda x: [0 if pd.isna([x[i]]) else x[i] for i in range(len(x))])

    return df, dict

def main():
    # Load and preprocess your data
    X_path = 'data/X_train.csv'
    Y_path = 'data/Y_train.csv'

    mixed_columns = ['item' + str(i) for i in range(1, 25)] + \
                    ['make' + str(i) for i in range(1, 25)] + \
                    ['model' + str(i) for i in range(1, 25)] + \
                    ['goods_code' + str(i) for i in range(1, 25)]
    
    mixed_columns_dtype = {col: str for col in mixed_columns}

    X = pd.read_csv(X_path, dtype=mixed_columns_dtype)
    y = pd.read_csv(Y_path)
    y = y.drop(['index', 'ID'], axis=1)

    X, dict = preprocess_data(X, y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Create data loaders
    batch_size = 64
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Set up device, model, criterion, and optimizer
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    #device = torch.device('cpu')
    print(f"Using device: {device}")

    model = FraudTransformer(input_dim=11000,
                            d_model=64,
                            nhead=2,
                            num_layers=2,
                            num_classes=1,
                            dropout=0).to(device)

    # Train and evaluate the model
    num_epochs = 5
    patience = 3
    best_test_loss = float('inf')
    epochs_without_improvement = 0

    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        # Update the learning rate scheduler
        scheduler.step()

        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}. Best Test Loss: {best_test_loss:.4f}")
            break

    # Evaluate model performance on train set
    y_pred, y_true = [], []
    a = -1
    with torch.no_grad():
        for batch in train_loader:
            data, target = batch
            data = data.long().to(device)
            output = model(data)
            a = max(a, float(torch.sigmoid(output).max()))
            preds = (torch.sigmoid(output) > 0.5)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(target.cpu().numpy())

    print(f"Max proba : {a}")

    print("Train Set Performance:")

    # Calculate PR-AUC
    pr_auc = average_precision_score(y_true, y_pred)
    print(f"PR-AUC: {pr_auc:.4f}")

    # Print classification report
    report = classification_report(y_true, y_pred, target_names=["Non-Fraud", "Fraud"])
    print("Classification Report:\n", report)

    exit()

    # Evaluate model performance on dev test
    y_pred, y_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            data, target = batch
            data = data.long().to(device)
            output = model(data)
            preds = (torch.sigmoid(output, dim=1) > 0.5)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(target.cpu().numpy())

    print("Test Set Performance:")

    # Calculate PR-AUC
    pr_auc = average_precision_score(y_true, y_pred)
    print(f"PR-AUC: {pr_auc:.4f}")

    # Print classification report
    report = classification_report(y_true, y_pred, target_names=["Non-Fraud", "Fraud"])
    print("Classification Report:\n", report)

    # Save the model
    torch.save(model.state_dict(), "trained_transformer.pt")

    # Get predictions for the test set
    X_test = pd.read_csv('data/X_test.csv', dtype=mixed_columns_dtype)
    X_test, _ = preprocess_data(X_test, dict=dict)
    test_dataset = CustomDataset(X_test, y=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            data, _ = batch
            data = data.long().to(device)
            output = model(data)
            preds = torch.argmax(output)
            y_pred.extend(preds.cpu().numpy())
    
    # Save predictions to CSV file
    y_pred = pd.DataFrame(y_pred, columns=['fraud_flag'])
    y_pred.to_csv('predictions.csv', index=False)

if __name__ == "__main__":
    main()
