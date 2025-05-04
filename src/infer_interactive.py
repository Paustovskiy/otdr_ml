import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use('Agg')

WINDOW_SIZE = 512
ANOMALY_THRESHOLD = 0.1  # чувствительность подсветки

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


def plot_comparison(original, reconstructed, diffs, title, save_path=None):
    plt.figure(figsize=(12, 4))
    plt.plot(original, label="Оригинал", alpha=0.6)
    plt.plot(reconstructed, label="Восстановленный", linewidth=1.5)
    if diffs.any():
        plt.scatter(np.where(diffs)[0], original[diffs], color="red", label="Аномалии", s=20)
    plt.title(title)
    plt.xlabel("Индекс")
    plt.ylabel("Амплитуда")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def infer_on_file(model, file_path, save_dir, log_path):
    signal = np.load(file_path)
    if len(signal) < WINDOW_SIZE:
        print("⚠ Файл слишком короткий!")
        return

    x = torch.tensor(signal[:WINDOW_SIZE], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(x).squeeze().numpy()

    mse = np.mean((signal[:WINDOW_SIZE] - output) ** 2)
    sim = ssim(signal[:WINDOW_SIZE], output, data_range=signal[:WINDOW_SIZE].max() - signal[:WINDOW_SIZE].min())
    diffs = np.abs(signal[:WINDOW_SIZE] - output) > ANOMALY_THRESHOLD

    filename = os.path.splitext(os.path.basename(file_path))[0]
    save_path = os.path.join(save_dir, f"{filename}_reconstructed.png")

    print(f"\n✅ Результаты для {filename}:")
    print(f"MSE:  {mse:.6f}")
    print(f"SSIM: {sim:.4f}")
    print(f"📸 Сохранено: {save_path}")

    plot_comparison(signal[:WINDOW_SIZE], output, diffs,
                    title=f"{filename}\nMSE: {mse:.6f}, SSIM: {sim:.4f}",
                    save_path=save_path)

    # === лог в CSV
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    row = pd.DataFrame([{
        "file": filename,
        "mse": mse,
        "ssim": sim,
        "anomalies_detected": int(diffs.sum())
    }])
    if os.path.exists(log_path):
        row.to_csv(log_path, mode="a", header=False, index=False)
    else:
        row.to_csv(log_path, index=False)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root, "models", "autoencoder.pth")
    save_dir = os.path.join(root, "data", "inference_samples")
    log_path = os.path.join(save_dir, "inference_samples_log.csv")
    os.makedirs(save_dir, exist_ok=True)

    model = Autoencoder()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    data_dir = os.path.join(root, "data")

    while True:
        available_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        print("\n📁 Доступные папки:")
        for i, name in enumerate(available_dirs):
            print(f"{i+1}. {name}")

        try:
            d_idx = int(input("Выберите номер папки (или 0 для выхода): ")) - 1
            if d_idx == -1:
                print("👋 Выход.")
                break
            assert 0 <= d_idx < len(available_dirs)
        except:
            print("❌ Неверный выбор папки.")
            continue

        target_dir = os.path.join(data_dir, available_dirs[d_idx])
        files = [f for f in os.listdir(target_dir) if f.endswith(".npy")]

        if not files:
            print("❌ Нет .npy файлов в выбранной папке.")
            continue

        print(f"\n📂 Файлы в {available_dirs[d_idx]}:")
        for i, f in enumerate(files):
            print(f"{i + 1}. {f}")

        try:
            f_idx = int(input("Выберите номер файла (или 0 для новой папки): ")) - 1
            if f_idx == -1:
                continue
            assert 0 <= f_idx < len(files)
        except:
            print("❌ Неверный выбор файла.")
            continue

        selected_path = os.path.join(target_dir, files[f_idx])
        infer_on_file(model, selected_path, save_dir, log_path)
