from fpdf import FPDF
import pandas as pd
import os

class PDF(FPDF):
    def __init__(self):
        super().__init__()
        font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
        self.add_font("DejaVu", "", os.path.join(font_dir, "DejaVuSans.ttf"), uni=True)
        self.add_font("DejaVu", "B", os.path.join(font_dir, "DejaVuSans-Bold.ttf"), uni=True)
        self.set_font("DejaVu", "", 12)

def add_image(pdf, path, title, w=180):
    if os.path.exists(path):
        pdf.set_font("DejaVu", "", 12)
        pdf.cell(0, 10, title, ln=1)
        pdf.image(path, w=w)
        pdf.ln(5)
    else:
        pdf.set_font("DejaVu", "I", 12)
        pdf.cell(0, 10, f"[Изображение не найдено: {title}]", ln=1)
        pdf.ln(5)

def generate_report(csv_path, vis_dir, save_path):
    if not os.path.exists(csv_path):
        print("❌ CSV-файл не найден:", csv_path)
        return

    df = pd.read_csv(csv_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Заголовок
    pdf.set_font("DejaVu", "", 16)
    pdf.cell(0, 10, "Отчёт по восстановлению сигналов с использованием Autoencoder", ln=1)

    # Основная статистика
    pdf.set_font("DejaVu", "", 12)
    pdf.ln(5)
    pdf.cell(0, 10, f"Всего сигналов: {len(df)}", ln=1)
    pdf.cell(0, 10, f"Средний MSE: {df['mse'].mean():.6f}", ln=1)
    pdf.cell(0, 10, f"Средний SSIM: {df['ssim'].mean():.4f}", ln=1)
    pdf.cell(0, 10, f"Аномалий обнаружено: {df['anomaly'].sum()}", ln=1)
    pdf.ln(10)

    # Визуализации
    add_image(pdf, os.path.join(vis_dir, "hist_mse.png"), "Гистограмма распределения MSE")
    add_image(pdf, os.path.join(vis_dir, "scatter_mse_ssim.png"), "Диаграмма рассеяния: MSE vs SSIM")
    add_image(pdf, os.path.join(vis_dir, "top10_mse.png"), "Топ-10 сигналов по MSE")
    add_image(pdf, os.path.join(vis_dir, "bottom10_ssim.png"), "Топ-10 сигналов по наименьшему SSIM")

    # Аномалии
    pdf.add_page()
    pdf.set_font("DejaVu", "", 14)
    pdf.cell(0, 10, "Топ-аномалии по MSE", ln=1)
    pdf.set_font("DejaVu", "", 11)
    df_anomalies = df[df["anomaly"] == True].sort_values(by="mse", ascending=False).head(10)
    for _, row in df_anomalies.iterrows():
        pdf.cell(0, 10, f"{row['file']} — MSE: {row['mse']:.6f}, SSIM: {row['ssim']:.4f}", ln=1)

    # Приложение
    pdf.add_page()
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(0, 10, "Приложение: Фрагмент таблицы результатов", ln=1)
    pdf.set_font("DejaVu", "", 9)

    df_short = df.head(30)
    for _, row in df_short.iterrows():
        pdf.cell(0, 10,
                 f"{row['file']:<35} MSE: {row['mse']:.6f}  SSIM: {row['ssim']:.4f}  Anomaly: {row['anomaly']}",
                 ln=1)

    pdf.output(save_path)
    print(f"\n✅ PDF-отчёт сохранён в: {save_path}")

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(root, "data", "inference_results.csv")
    vis_dir = os.path.join(root, "data", "visualizations")
    report_dir = os.path.join(root, "data", "reports")
    report_path = os.path.join(report_dir, "inference_report.pdf")

    generate_report(csv_path, vis_dir, report_path)
