import os
import numpy as np

def split_signal(signal, window_size=512, step=256):
    """
    Делит сигнал на фрагменты фиксированной длины с возможным перекрытием.
    """
    segments = []
    for i in range(0, len(signal) - window_size + 1, step):
        segment = signal[i:i + window_size]
        segments.append(segment)
    return np.array(segments)


def build_dataset(input_dir, output_path, window_size=512, step=256):
    all_segments = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".npy"):
            path = os.path.join(input_dir, filename)
            signal = np.load(path)
            segments = split_signal(signal, window_size, step)
            all_segments.append(segments)

    dataset = np.vstack(all_segments)
    np.save(output_path, dataset)
    print(f"✅ Датасет сохранён: {output_path} | shape: {dataset.shape}")
    return dataset


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(root, "data", "real_preprocessed")
    output_path = os.path.join(root, "data", "dataset.npy")

    build_dataset(input_dir, output_path)
