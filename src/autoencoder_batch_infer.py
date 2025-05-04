import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from skimage.metrics import structural_similarity as ssim  # ⬅ добавили SSIM

matplotlib.use("Agg")

WINDOW_SIZE = 512

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


def plot_comparison(original, reconstructed, save_path):
    plt.figure(figsize=(12, 4))
    plt.plot(original, label="Оригинал", alpha=0.6)
    plt.plot(reconstructed, label="Восстановленный", linewidth=1.5)
    plt.title("Восстановление сигнала")
    plt.xlabel("Индекс")
    plt.ylabel("Амплитуда")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root, "models", "autoencoder.pth")
    input_dir = os.path.join(root, "data", "real_preprocessed")
    output_csv = os.path.join(root, "data", "inference_results.csv")

    model = Autoencoder()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    results = []

    for fname in os.listdir(input_dir):
        if fname.endswith(".npy"):
            path = os.path.join(input_dir, fname)
            signal = np.load(path)
            if len(signal) < WINDOW_SIZE:
                print(f"⚠ Пропущен (короткий): {fname}")
                continue

            x = torch.tensor(signal[:WINDOW_SIZE], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                output = model(x).squeeze().numpy()

            mse_val = np.mean((signal[:WINDOW_SIZE] - output) ** 2)
            ssim_val = ssim(signal[:WINDOW_SIZE], output, data_range=output.max() - output.min())  # ⬅ SSIM

            results.append({
                "file": fname,
                "mse": mse_val,
                "ssim": ssim_val
            })

            # Сохранить график
            plot_path = os.path.join(input_dir, fname.replace(".npy", "_reconstructed.png"))
            plot_comparison(signal[:WINDOW_SIZE], output, plot_path)
            print(f"✅ {fname}: MSE = {mse_val:.6f}, SSIM = {ssim_val:.4f}")

    # Анализ и сохранение
    df = pd.DataFrame(results)
    anomaly_threshold = df["mse"].mean() + 2 * df["mse"].std()
    df["anomaly"] = df["mse"] > anomaly_threshold
    df_sorted = df.sort_values(by="mse", ascending=False)
    df_sorted.to_csv(output_csv, index=False)
    print(f"\n📋 Обновлённая таблица с MSE и SSIM сохранена: {output_csv}")
    print(f"🚨 Порог аномалии (по MSE): {anomaly_threshold:.6f}")
    print(df_sorted.head(5))

    anomalies_only = df_sorted[df_sorted["anomaly"] == True]
    anomalies_only.to_csv(
        os.path.join(os.path.dirname(output_csv), "anomalies_only.csv"), index=False
    )
    print(f"📌 Отдельно сохранены аномальные сигналы: anomalies_only.csv")
