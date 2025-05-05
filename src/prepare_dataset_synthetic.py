import numpy as np
import os

WINDOW_SIZE = 512
STEP = 100 # перекрытие фрагментов (для увеличения количества данных)

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(root, "data", "synthetic", "clean.npy")
output_path = os.path.join(root, "data", "dataset.npy")

# === Загрузка clean-сигнала
signal = np.load(input_path)
segments = []

for i in range(0, len(signal) - WINDOW_SIZE + 1, STEP):
    segments.append(signal[i:i + WINDOW_SIZE])

segments = np.array(segments)
print(f"✅ Нарезано {segments.shape[0]} фрагментов из clean.npy")

# === Сохранение как dataset.npy
np.save(output_path, segments)
print(f"📦 Сохранено в: {output_path}")