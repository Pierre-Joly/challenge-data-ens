import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from torch.optim.lr_scheduler import StepLR
import category_encoders as ce


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float(
        ) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class FraudTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_classes, dropout=0.1):
        super(FraudTransformer, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=input_dim, embedding_dim=d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)
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
        x = self.embedding(x)
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


def preprocess_data(df, y=None):
    # Drop columns
    cols_base = ['goods_code', 'model']
    columns_to_drop = ['ID'] + [col + str(i)
                                for col in cols_base for i in range(1, 25)]
    df = df.drop(columns_to_drop, axis=1)

    # Identify the columns to apply RNN tokenization
    rnn_columns = ['make', 'item']  # Add more columns as needed
    rnn_columns = [col + str(i) for col in rnn_columns for i in range(1, 25)]

    # Identify the categorical and numerical columns
    categorical_columns = rnn_columns
    numerical_columns = [
        col for col in df.columns if col not in categorical_columns]

    # Clean data
    for col in categorical_columns:
        df[col] = df[col].fillna('')
    for col in numerical_columns:
        df[col] = df[col].fillna(0)

    # Define transformers
    cat_pipeline = ce.CatBoostEncoder(cols=categorical_columns)
    num_pipeline = make_pipeline(StandardScaler())

    # Create the preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('cat_pipeline', cat_pipeline, categorical_columns),
        ('num_pipeline', num_pipeline, numerical_columns)
    ])

    # Preprocess the data
    if y is not None:
        df_preprocessed = preprocessor.fit_transform(df, y)
    else:
        df_preprocessed = preprocessor.transform(df)

    df_preprocessed = pd.DataFrame(df_preprocessed, columns=df.columns)

    return df_preprocessed


def main():
    # Load and preprocess your data
    X_path = 'data/X_train.csv'
    Y_path = 'data/Y_train.csv'

    mixed_columns = ['item' + str(i) for i in range(1, 25)] + ['make' + str(i) for i in range(1, 25)] + [
        'model' + str(i) for i in range(1, 25)] + ['goods_code' + str(i) for i in range(1, 25)]
    mixed_columns_dtype = {col: str for col in mixed_columns}

    X = pd.read_csv(X_path, dtype=mixed_columns_dtype)
    y = pd.read_csv(Y_path)
    y = y.drop(['index', 'ID'], axis=1)

    X = preprocess_data(X, y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # Create data loaders
    batch_size = 64
    train_dataset = TensorDataset(torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(
        y_train.values, dtype=torch.long).squeeze())
    test_dataset = TensorDataset(torch.tensor(X_test.values, dtype=torch.float32), torch.tensor(
        y_test.values, dtype=torch.long).squeeze())
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # Set up device, model, criterion, and optimizer
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model = FraudTransformer(input_dim=X_train.shape[1], d_model=64, nhead=4, num_layers=int(
        2), num_classes=2, dropout=0.5).to(device)
    class_weights = torch.tensor([1.0, (y_train == 0).sum(
    ) / (y_train == 1).sum()], dtype=torch.float32).to(device)
    criterion = nn.BCELoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Train and evaluate the model
    num_epochs = 10
    patience = 3
    best_test_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        # Update the learning rate scheduler
        scheduler.step()

        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}. Best Test Loss: {best_test_loss:.4f}")
            break

    # Evaluate model performance
    y_pred, y_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            data, target = batch
            data = data.long().to(device)
            output = model(data)
            preds = torch.argmax(output, dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(target.cpu().numpy())

    from sklearn.metrics import average_precision_score, classification_report

    # Calculate PR-AUC
    pr_auc = average_precision_score(y_true, y_pred)
    print(f"PR-AUC: {pr_auc:.4f}")

    # Print classification report
    report = classification_report(
        y_true, y_pred, target_names=["Non-Fraud", "Fraud"])
    print("Classification Report:\n", report)


if __name__ == "__main__":
    main()
