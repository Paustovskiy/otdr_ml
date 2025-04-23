import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.use('Agg')

# Определяем путь к data/synthetic из корня проекта
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_dir = os.path.join(project_root, "data", "synthetic")
os.makedirs(save_dir, exist_ok=True)

def generate_clean_signal(length=1000):
    """Создание чистого сигнала — плавный спуск амплитуды (затухание в волокне)"""
    x = np.linspace(0, 1, length)
    signal = np.exp(-5 * x)  # имитация экспоненциального затухания
    return signal

def add_noise(signal, noise_level=0.02):
    """Добавление белого гауссовского шума"""
    noise = np.random.normal(0, noise_level, size=signal.shape)
    return signal + noise

def insert_defect(signal, position=600, depth=0.4, width=10):
    """Добавление аномалии (всплеск)"""
    signal_def = signal.copy()
    signal_def[position:position + width] -= depth
    return signal_def

def plot_signals(*signals, labels=None, title="Сигналы"):
    plt.figure(figsize=(12, 4))
    for i, sig in enumerate(signals):
        plt.plot(sig, label=labels[i] if labels else f"Signal {i+1}")
    plt.xlabel("Индекс")
    plt.ylabel("Амплитуда")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "synthetic_signals.png"))


# Генерация
clean = generate_clean_signal()
noisy = add_noise(clean)
defected = insert_defect(noisy)

# Сохраняем
np.save(os.path.join(save_dir, "clean.npy"), clean)
np.save(os.path.join(save_dir, "noisy.npy"), noisy)
np.save(os.path.join(save_dir, "defected.npy"), defected)

# Визуализация
plot_signals(clean, noisy, defected,
             labels=["Чистый", "С шумом", "С дефектом"],
             title="Генерация синтетических сигналов OTDR")

print("Сигналы сохранены в data/synthetic")