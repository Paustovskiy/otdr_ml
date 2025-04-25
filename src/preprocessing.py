import numpy as np
import matplotlib.pyplot as plt
import pywt
import os


def zscore_normalize(signal):
    """Z-нормализация сигнала"""
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / std


def wavelet_denoise(signal, wavelet='db4', level=3, threshold_ratio=0.04):
    """
    Вейвлет-фильтрация сигнала с мягким порогом.
    - wavelet: базовая вейвлет-функция
    - level: уровень декомпозиции
    - threshold_ratio: доля максимального коэффициента, используемая как порог
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = threshold_ratio * max(np.abs(coeffs[-level]))
    coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs_thresh, wavelet)


def plot_signals(original, processed, title="Предобработка сигнала", save_path=None):
    plt.figure(figsize=(12, 4))
    plt.plot(original, label="Оригинал", alpha=0.6)
    plt.plot(processed, label="После фильтрации", linewidth=1.5)
    plt.xlabel("Индекс")
    plt.ylabel("Амплитуда")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"✅ График сохранён: {save_path}")
    else:
        plt.show()
    plt.close()


def preprocess_file(input_path, output_dir, name):
    signal = np.load(input_path)

    denoised = wavelet_denoise(signal)
    normalized = zscore_normalize(denoised)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{name}_preprocessed.npy"), normalized)

    plot_path = os.path.join(output_dir, f"{name}_compare.png")
    plot_signals(signal, normalized, title=f"Фильтрация и нормализация: {name}", save_path=plot_path)
    print(f"✅ Обработан и сохранён: {name}")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(root, "data", "real")
    output_dir = os.path.join(root, "data", "real_preprocessed")

    # обрабатываем все .npy в папке real/
    for filename in os.listdir(input_dir):
        if filename.endswith(".npy"):
            file_path = os.path.join(input_dir, filename)
            name = filename.replace(".npy", "")
            preprocess_file(file_path, output_dir, name)
