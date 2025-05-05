import numpy as np
import matplotlib.pyplot as plt
import os

import matplotlib
matplotlib.use('Agg')

# === Папка сохранения
save_dir = os.path.join("data", "synthetic")
os.makedirs(save_dir, exist_ok=True)

# === Генерация clean сигнала
def generate_clean_signal(length=1000):
    x = np.linspace(0, 1, length)
    return np.exp(-5 * x)  # плавное затухание

# === Шум
def add_noise(signal, noise_level=0.02):
    return signal + np.random.normal(0, noise_level, size=signal.shape)

# === Дефект — резкое падение амплитуды в случайной позиции
def insert_defect(signal, depth=0.3, width=10):
    signal = signal.copy()
    pos = np.random.randint(200, len(signal) - width - 100)
    signal[pos:pos + width] -= depth
    return np.clip(signal, 0, 1)

# === Визуализация
def plot_signals(clean, noisy, defected):
    plt.figure(figsize=(12, 4))
    plt.plot(clean, label="Clean", linewidth=1)
    plt.plot(noisy, label="Noisy", linewidth=1)
    plt.plot(defected, label="Defected", linewidth=1)
    plt.xlabel("Индекс")
    plt.ylabel("Амплитуда")
    plt.title("Синтетические OTDR-сигналы")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# === Основной блок
if __name__ == "__main__":
    clean = generate_clean_signal()
    noisy = add_noise(clean)
    defected = insert_defect(noisy)

    np.save(os.path.join(save_dir, "clean.npy"), clean)
    np.save(os.path.join(save_dir, "noisy.npy"), noisy)
    np.save(os.path.join(save_dir, "defected.npy"), defected)

    plot_signals(clean, noisy, defected)
    print("✅ Сигналы сохранены в:", save_dir)
