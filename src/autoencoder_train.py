import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

# === 1. Параметры ===
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
WINDOW_SIZE = 512

# === 2. Архитектура Autoencoder ===
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(WINDOW_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, WINDOW_SIZE),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# === 3. Загрузка данных ===
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(root, "data", "dataset.npy")
data = np.load(data_path)
data = torch.tensor(data, dtype=torch.float32)

# === 4. Подготовка DataLoader ===
dataset = TensorDataset(data, data)  # X = Y (unsupervised)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === 5. Обучение ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"Начинаем обучение на {device}...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_x, _ in loader:
        batch_x = batch_x.to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Эпоха {epoch+1}/{EPOCHS}, потери: {total_loss:.4f}")

# === 6. Сохранение модели ===
os.makedirs(os.path.join(root, "models"), exist_ok=True)
model_path = os.path.join(root, "models", "autoencoder_synthetic.pth")
torch.save(model.state_dict(), model_path)
print(f"✅ Модель сохранена в: {model_path}")