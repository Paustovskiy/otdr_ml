import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import structural_similarity as ssim

WINDOW_SIZE = 512
ANOMALY_THRESHOLD = 0.02  # фиксированный порог

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


def plot_comparison(original, reconstructed, anomalies, title, save_path):
    plt.figure(figsize=(12, 4))
    plt.plot(original, label="Оригинал", alpha=0.6)
    plt.plot(reconstructed, label="Восстановленный", linewidth=1.5)
    if anomalies.any():
        plt.scatter(np.where(anomalies)[0], original[anomalies],
                    color="red", label="Аномалии", s=20)
    plt.title(title)
    plt.xlabel("Индекс")
    plt.ylabel("Амплитуда")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def infer_on_file(model, file_path, save_dir, log_path):
    signal = np.load(file_path)
    if len(signal) < WINDOW_SIZE:
        print("⚠ Файл слишком короткий!")
        return

    orig = signal[:WINDOW_SIZE]
    x = torch.tensor(orig, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        rec = model(x).squeeze().numpy()

    mse = np.mean((orig - rec) ** 2)
    dr = orig.max() - orig.min()
    sim = ssim(orig, rec, data_range=dr)

    diffs = np.abs(orig - rec)

    # 1) фиксированный порог
    anoms_fix = diffs > ANOMALY_THRESHOLD
    # 2) динамический порог: mean + 2*std
    thr_dyn = diffs.mean() + 2 * diffs.std()
    anoms_dyn = diffs > thr_dyn
    # 3) percentile-порог (95%)
    thr_pct = np.percentile(diffs, 95)
    anoms_pct = diffs > thr_pct

    filename = os.path.splitext(os.path.basename(file_path))[0]

    # Сохраняем гистограмму ошибок
    hist_path = os.path.join(save_dir, f"{filename}_error_hist.png")
    plt.figure(figsize=(6, 4))
    plt.hist(diffs, bins=50, alpha=0.7)
    plt.axvline(ANOMALY_THRESHOLD, color='r', linestyle='--',
                label=f'fix {ANOMALY_THRESHOLD:.3f}')
    plt.axvline(thr_dyn, color='g', linestyle='--',
                label=f'dyn {thr_dyn:.3f}')
    plt.axvline(thr_pct, color='b', linestyle='--',
                label=f'95% {thr_pct:.3f}')
    plt.legend()
    plt.xlabel('Ошибка |оригинал – восстановление|')
    plt.ylabel('Частота')
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()

    # Сохраняем график восстановления (с динамическим порогом)
    recon_path = os.path.join(save_dir, f"{filename}_reconstructed.png")
    title = f"{filename}\nMSE={mse:.6f}, SSIM={sim:.4f}"
    plot_comparison(orig, rec, anoms_dyn, title, recon_path)

    # Лог в CSV
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    row = pd.DataFrame([{
        "file": filename,
        "mse": mse,
        "ssim": sim,
        "anom_fix": int(anoms_fix.sum()),
        "anom_dyn": int(anoms_dyn.sum()),
        "anom_pct": int(anoms_pct.sum())
    }])
    if os.path.exists(log_path):
        row.to_csv(log_path, mode="a", header=False, index=False)
    else:
        row.to_csv(log_path, index=False)

    # Консольный вывод
    print(f"\n✅ {filename}: MSE={mse:.6f}, SSIM={sim:.4f}")
    print(f" Аномалий fix/dyn/pct: {anoms_fix.sum()}/{anoms_dyn.sum()}/{anoms_pct.sum()}")
    print(f"📊 Гистограмма сохранена: {os.path.basename(hist_path)}")
    print(f"📸 Реконструкция сохранена: {os.path.basename(recon_path)}")


if __name__ == "__main__":
    root       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root, "models", "autoencoder_synthetic.pth")
    save_dir   = os.path.join(root, "data", "inference_samples")
    log_path   = os.path.join(save_dir, "inference_samples_log.csv")
    os.makedirs(save_dir, exist_ok=True)

    # Загружаем модель
    model = Autoencoder()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    data_dir = os.path.join(root, "data")
    while True:
        # 1) Список папок
        dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        print("\n📁 Доступные папки:")
        for i, d in enumerate(dirs, 1):
            print(f"{i}. {d}")

        try:
            d_idx = int(input("Выберите номер папки (0 — выход): "))
            if d_idx == 0:
                break
            target_dir = os.path.join(data_dir, dirs[d_idx-1])
        except:
            print("❌ Неверный ввод, попробуйте ещё раз.")
            continue

        # 2) Список файлов .npy в выбранной папке
        files = [f for f in os.listdir(target_dir) if f.endswith(".npy")]
        if not files:
            print("❌ В папке нет .npy файлов.")
            continue

        print(f"\n📂 Файлы в «{dirs[d_idx-1]}»:")
        for i, f in enumerate(files, 1):
            print(f"{i}. {f}")

        try:
            f_idx = int(input("Выберите номер файла (0 — выбрать папку): "))
            if f_idx == 0:
                continue
            file_path = os.path.join(target_dir, files[f_idx-1])
        except:
            print("❌ Неверный ввод, попробуйте ещё раз.")
            continue

        # 3) Запуск инференса
        infer_on_file(model, file_path, save_dir, log_path)
