# src/export_model.py
import torch
import os
from autoencoder_train_load import Autoencoder  # или откуда у тебя класс

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_IN = os.path.join(ROOT, "models", "autoencoder_real(based_synth).pth")
MODEL_OUT = os.path.join(ROOT, "models", "autoencoder_real(based_synth).ts")

# 1) Загрузим fine-tuned модель
model = Autoencoder()
model.load_state_dict(torch.load(MODEL_IN, map_location="cpu"))
model.eval()

# 2) Примера вход (батч 1×512)
example = torch.randn(1, 512)

# 3) Трассировка TorchScript
traced = torch.jit.trace(model, example)

# 4) Сохраняем
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
traced.save(MODEL_OUT)
print(f"✅ TorchScript-модель сохранена: {MODEL_OUT}")