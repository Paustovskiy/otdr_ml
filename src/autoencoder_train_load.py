# autoencoder_train_load.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

# === 1. Параметры ===
EPOCHS = 30          # меньше, так как fine-tuning
BATCH_SIZE = 32
LR = 1e-4            # маленький шаг для дообучения
WINDOW_SIZE = 512

# Имена моделей
PRETRAINED = "autoencoder_synthetic.pth"
FINE_TUNED = "autoencoder_real(based_synth).pth"

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
        return self.decoder(self.encoder(x))


def main():
    # === 3. Путь к данным ===
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root, "data", "dataset(real).npy")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Набор данных не найден: {data_path}")
    data = np.load(data_path)
    data = torch.tensor(data, dtype=torch.float32)

    loader = DataLoader(TensorDataset(data, data),
                        batch_size=BATCH_SIZE,
                        shuffle=True)

    # === 4. Устройство и модель ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)

    # Пути к моделям
    load_path = os.path.join(root, "models", PRETRAINED)
    save_path = os.path.join(root, "models", FINE_TUNED)

    # === FINE-TUNING: загрузка предобученной модели ===
    if os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path, map_location=device))
        print(f"✅ Загружена предобученная модель: {PRETRAINED}")
    else:
        print(f"⚠ Предобученная модель {PRETRAINED} не найдена — обучение с нуля")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # === 5. Обучение ===
    print(f"Начинаем fine-tuning на {device}...")
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Эпоха {epoch}/{EPOCHS}, Потери: {total_loss:.4f}")

    # === 6. Сохранение fine-tuned модели ===
    os.makedirs(os.path.join(root, "models"), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Fine-tuned модель сохранена в: models/{FINE_TUNED}")


if __name__ == "__main__":
    main()
