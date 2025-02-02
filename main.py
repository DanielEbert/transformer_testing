import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'{device=}')

class StockChartDataset(Dataset):
    def __init__(self, root_dir, transform=None, scaler=None, fit_scaler=False):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.filepaths = []
        self.labels = []
        self.scaler = scaler

        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for filename in os.listdir(class_dir):
                if filename.endswith(".npy"):
                    self.filepaths.append(os.path.join(class_dir, filename))
                    self.labels.append(class_idx)

        if self.scaler is None:
            self.scaler = StandardScaler()
            if fit_scaler:
                all_data = []
                for filepath in self.filepaths:
                    chart_data = np.load(filepath)
                    all_data.append(chart_data)
                all_data_np = np.concatenate(all_data).reshape(-1, 1)
                self.scaler.fit(all_data_np)


    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        chart_data = np.load(filepath)

        # normalize
        chart_data_scaled = self.scaler.transform(chart_data.reshape(-1, 1)).flatten()
        # chart data shape (1, 100)
        chart_data = torch.tensor(chart_data_scaled, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            chart_data = self.transform(chart_data)

        return chart_data, label

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class StockChartTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, dropout=0.1):
        super(StockChartTransformer, self).__init__()

        self.embedding = nn.Linear(input_dim, d_model) # input embedding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, output_dim) # output layer

    def forward(self, src):
        # src shape will be (batch_size, seq_len, input_dim) -> (N, 100, 1)
        src = self.embedding(src)               # (N, 100, d_model)
        src = self.pos_encoder(src)             # (N, 100, d_model)
        output = self.transformer_encoder(src)  # (N, 100, d_model)
        output = output.mean(dim=1)             # (N, d_model) - average pooling over sequence
        return self.fc(output)                  # (N, output_dim)


input_dim = 1  # currently input number is a single feature
sequence_length = 100  # Length of each input sequence
d_model = 64  # Reduced d_model
nhead = 4  # Reduced nhead, should divide d_model
num_layers = 4  # Reduced num_layers significantly
dim_feedforward = 256  # Reduced dim_feedforward
output_dim = 2  # Binary classification
dropout = 0.1
batch_size = 64  # Increased batch size
learning_rate = 1e-4  # Increased learning rate
num_epochs = 50  # Increased epochs

full_dataset = StockChartDataset(root_dir='data', fit_scaler=True)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


model = StockChartTransformer(input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct_predictions_train = 0
    total_samples_train = 0

    for batch_idx, (data, targets) in enumerate(train_dataloader):
        data = data.to(device)  # shape (N, 1, 100) -> need (N, 100, 1)
        data = data.permute(0, 2, 1)  # Reshape to (batch_size, seq_len, input_dim) = (N, 100, 1)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_samples_train += targets.size(0)
        correct_predictions_train += (predicted == targets).sum().item()

    model.eval()
    val_loss = 0.0
    correct_predictions_val = 0
    total_samples_val = 0
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(val_dataloader):
            data = data.to(device)
            data = data.permute(0, 2, 1) # Reshape for transformer input
            targets = targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_samples_val += targets.size(0)
            correct_predictions_val += (predicted == targets).sum().item()

    avg_train_loss = train_loss / len(train_dataloader)
    avg_val_loss = val_loss / len(val_dataloader)
    train_accuracy = correct_predictions_train / total_samples_train
    val_accuracy = correct_predictions_val / total_samples_val

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')



def infer(chart_data_np, model, scaler, device, target_label = None):
    model.eval()
    model.to(device)

    # Preprocess the input data
    chart_data_scaled = scaler.transform(chart_data_np.reshape(-1, 1)).flatten()
    chart_tensor = torch.tensor(chart_data_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 100) -> (batch_size, channel, seq_len)

    chart_tensor = chart_tensor.permute(0, 2, 1) # (1, 100, 1) -> (batch_size, seq_len, input_dim)

    with torch.no_grad():
        output = model(chart_tensor)
        _, predicted_class = torch.max(output, 1)

        print(f'output={output.cpu().numpy()}')
        if target_label is not None:
            target_tensor = torch.tensor([target_label], dtype=torch.long).to(device)
            loss = criterion(output, target_tensor).item()
            print(f'{loss=}')

    print('prediction', predicted_class.item())
    print()

    return predicted_class.item()


import random

NUMBERS_PER_FILE = 100

def generate(increment):
    numbers = []
    cur = random.uniform(0, 10)
    for _ in range(NUMBERS_PER_FILE):
        numbers.append(cur)
        cur += random.uniform(*increment)
    return np.array(numbers)


inference_scaler = full_dataset.scaler
infer(generate((-1, 5)), model, inference_scaler, device, 1)
infer(generate((-5, 1)), model, inference_scaler, device, 0)
infer(generate((-1, 5)), model, inference_scaler, device, 1)
