import torch.nn as nn

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
# from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'{device=}')

class StockChartDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.filepaths = []
        self.labels = []

        # Load file paths and assign labels
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for filename in os.listdir(class_dir):
                if filename.endswith(".npy"):
                    self.filepaths.append(os.path.join(class_dir, filename))
                    self.labels.append(class_idx)  # Assign label based on folder index

        # Initialize a scaler (you can fit it later if needed)
        # self.scaler = StandardScaler()

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Load NumPy array
        chart_data = np.load(filepath)

        # Normalize/Standardize (example with standardization)
        # chart_data = self.scaler.fit_transform(chart_data.reshape(-1, 1)).flatten() # Reshape for scaler

        chart_data = torch.tensor(chart_data, dtype=torch.float32)

        if self.transform:
            chart_data = self.transform(chart_data)

        return chart_data, label

# Example usage
dataset = StockChartDataset(root_dir='data')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class StockChartTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, dropout=0.1):
        super(StockChartTransformer, self).__init__()

        # Embedding (you may not need this for your numerical input, but I've included it for generality)
        # self.embedding = nn.Linear(input_dim, d_model)  # Alternatively: nn.Embedding(vocab_size, d_model)

        # Positional Encoding (consider adding if the order of your data points is important)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer Encoder Layers
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Classification Head
        self.fc = nn.Linear(d_model, output_dim)

        # Activation
        self.activation = nn.ReLU()

    def forward(self, src):
        # src = self.embedding(src)  # If using embedding
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)  # Aggregate the sequence output (e.g., average pooling)
        output = self.fc(output)
        output = self.activation(output)
        return output

# Positional Encoding (if needed)
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


import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


input_dim = 1  # Replace with the actual number of features per day
d_model = 100 # Example hidden dimension
nhead = 5
num_layers = 30
dim_feedforward = 2048
output_dim = 2
dropout = 0.1
batch_size = 32

# Initialize Model, Loss, Optimizer
model = StockChartTransformer(input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, dropout)
model.to(device)

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-7)  # Start with a small learning rate

# Learning rate scheduler (warm-up and decay)
def lr_lambda(current_step):
    warmup_steps = 4000
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 0.5 ** float(current_step // 10000)

scheduler = LambdaLR(optimizer, lr_lambda)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # print(f"Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (data, targets) in enumerate(dataloader):
        data = data.to(device) # Move data to GPU
        targets = targets.to(device) # Move targets to GPU

        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent or Adam step
        optimizer.step()
        scheduler.step()  # Update learning rate
        if batch_idx % 1 == 0: # print every batch for example
            print(f'  Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}')


def classify_stock_chart(model, chart_data, scaler):
  """Classifies a single stock chart.

  Args:
    model: The trained PyTorch model.
    chart_data: The stock chart data as a NumPy array.
    scaler: The fitted StandardScaler used during training.

  Returns:
    The predicted class label.
  """
  model.eval()  # Set model to evaluation mode
  model.to(device)

  # Preprocess the chart data (same as during training)
  # chart_data = scaler.transform(chart_data)
  chart_data = torch.tensor(chart_data, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
  chart_data = chart_data.view(-1, 100, input_dim)  # Reshape input data to (batch_size, seq_len, input_dim)

  # Make prediction
  with torch.no_grad():
    output = model(chart_data)
    print(output)
    predicted_class = output.argmax(dim=1).item()

  return predicted_class


import random

NUMBERS_PER_FILE = 100

def generate(increment):
    numbers = []
    cur = random.uniform(0, 10)
    for _ in range(NUMBERS_PER_FILE):
        numbers.append(cur)
        cur += random.uniform(*increment)
    return np.array(numbers)


increment = (-1, 5)

print(classify_stock_chart(model, generate((-1, 5)), None))
print(classify_stock_chart(model, generate((-5, 1)), None))
