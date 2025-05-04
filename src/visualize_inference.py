import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg')

def visualize_results(csv_path, save_dir):
    df = pd.read_csv(csv_path)
    os.makedirs(save_dir, exist_ok=True)

    # 1. Гистограмма MSE
    plt.figure(figsize=(8, 4))
    plt.hist(df["mse"], bins=20, color="skyblue", edgecolor="black")
    plt.axvline(df["mse"].mean(), color="red", linestyle="--", label="Среднее")
    plt.title("Распределение MSE по сигналам")
    plt.xlabel("MSE")
    plt.ylabel("Количество")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "hist_mse.png"))
    plt.close()

    # 2. Scatter MSE vs SSIM
    plt.figure(figsize=(6, 6))
    plt.scatter(df["mse"], df["ssim"], alpha=0.7)
    plt.title("MSE vs SSIM")
    plt.xlabel("MSE")
    plt.ylabel("SSIM")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "scatter_mse_ssim.png"))
    plt.close()

    # 3. Топ-10 аномалий по MSE
    top_anomalies = df.sort_values(by="mse", ascending=False).head(10)
    plt.figure(figsize=(10, 4))
    plt.barh(top_anomalies["file"], top_anomalies["mse"], color="salmon")
    plt.title("Топ-10 сигналов с наибольшим MSE")
    plt.xlabel("MSE")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "top10_mse.png"))
    plt.close()

    # 4. Топ-10 по наименьшему SSIM (наихудшее структурное сходство)
    bottom_ssim = df.sort_values(by="ssim", ascending=True).head(10)
    plt.figure(figsize=(10, 4))
    plt.barh(bottom_ssim["file"], bottom_ssim["ssim"], color="orange")
    plt.title("Топ-10 сигналов с наименьшим SSIM")
    plt.xlabel("SSIM")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bottom10_ssim.png"))
    plt.close()

    print("✅ Визуализации сохранены в:", save_dir)

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(root, "data", "inference_results.csv")
    save_dir = os.path.join(root, "data", "visualizations")

    visualize_results(csv_path, save_dir)
