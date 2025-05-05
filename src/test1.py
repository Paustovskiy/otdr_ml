import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from autoencoder_train import Autoencoder  # или куда вынес класс

# 1) загрузить модель
ae = Autoencoder()
ae.load_state_dict(torch.load("models/autoencoder_synthetic.pth", map_location="cpu"))
ae.eval()

# 2) вспомогательная функция
def eval_signal(name):
    sig = np.load(f"data/synthetic/{name}.npy")[:512]
    with torch.no_grad():
        rec = ae(torch.tensor(sig, dtype=torch.float32).unsqueeze(0)).numpy().squeeze()
    dr = sig.max() - sig.min()
    m = ((sig - rec)**2).mean()
    s = ssim(sig, rec, data_range=dr)
    print(f"{name:8s} → MSE = {m:.6f}, SSIM = {s:.4f}")

# 3) проверяем
for nm in ("clean","noisy","defected"):
    eval_signal(nm)
