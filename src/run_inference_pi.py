# src/run_inference_pi.py
import torch
import numpy as np
import os
import time

WINDOW_SIZE = 512

class AutoencoderTS:
    def __init__(self, model_path):
        # Загружаем TorchScript
        self.model = torch.jit.load(model_path, map_location="cpu")
        self.model.eval()

    def infer(self, signal):
        # signal: numpy array length >= WINDOW_SIZE
        x = torch.from_numpy(signal[:WINDOW_SIZE].astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            out = self.model(x).squeeze(0).numpy()
        return out

def main():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(ROOT, "models", "autoencoder_real(based_synth).ts")
    data_dir = os.path.join(ROOT, "data", "real_preprocessed")

    ae = AutoencoderTS(model_path)

    times = []
    for fname in os.listdir(data_dir):
        if not fname.endswith(".npy"):
            continue
        sig = np.load(os.path.join(data_dir, fname))
        # прогрев
        _ = ae.infer(sig)
        # замер
        start = time.time()
        _ = ae.infer(sig)
        end   = time.time()
        dt = (end - start) * 1000  # ms
        times.append(dt)
        print(f"{fname}: {dt:.2f} ms")

    print(f"\nСреднее время инференса: {np.mean(times):.2f} ms")

if __name__ == "__main__":
    main()
