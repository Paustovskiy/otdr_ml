import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

WINDOW_SIZE = 512  # размер входа модели

# === Модель (должна совпадать с обучением) ===
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

# === Визуализация сигнала до/после ===
def plot_comparison(original, reconstructed, save_path=None):
    plt.figure(figsize=(12, 4))
    plt.plot(original, label="Оригинал", alpha=0.6)
    plt.plot(reconstructed, label="Восстановленный", linewidth=1.5)
    plt.title("Восстановление сигнала через автоэнкодер")
    plt.xlabel("Индекс")
    plt.ylabel("Амплитуда")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"✅ График сохранён: {save_path}")
    else:
        plt.show()
    plt.close()

# === Основной код ===
if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root, "models", "autoencoder.pth")
    signal_path = os.path.join(root, "data", "real_preprocessed", "100601_12015_7_preprocessed.npy")
    save_plot = os.path.join(root, "data", "real_preprocessed", "100601_12015_7_compare.png")

    # Загружаем модель
    model = Autoencoder()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Загружаем сигнал
    signal = np.load(signal_path)
    signal = signal[:WINDOW_SIZE]  # только первые 512 точек
    x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)  # [1, 512]

    # Инференс
    with torch.no_grad():
        output = model(x).squeeze().numpy()

    # MSE
    mse = np.mean((signal - output) ** 2)
    print(f"📊 MSE восстановленного сигнала: {mse:.6f}")

    # Визуализация
    plot_comparison(signal, output, save_path=save_plot)
