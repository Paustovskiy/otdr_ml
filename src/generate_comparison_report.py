# src/generate_comparison_report.py

import os
import pandas as pd
from fpdf import FPDF

# --- Настройки путей ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(ROOT, "data", "inference_samples", "inference_samples_log.csv")
VIS_DIR  = os.path.join(ROOT, "data", "inference_samples")
REPORT_DIR = os.path.join(ROOT, "data", "reports")
REPORT_PATH = os.path.join(REPORT_DIR, "comparison_report.pdf")

# Известные synthetic-имена
SYN_FILES = {"clean", "noisy", "defected"}

# --- PDF-класс с DejaVu-шрифтами ---
class PDF(FPDF):
    def __init__(self):
        super().__init__()
        font_dir = os.path.join(os.path.dirname(__file__), "fonts")
        self.add_font("DejaVu", "", os.path.join(font_dir, "DejaVuSans.ttf"), uni=True)
        self.add_font("DejaVu", "B", os.path.join(font_dir, "DejaVuSans-Bold.ttf"), uni=True)
        self.set_font("DejaVu", "", 12)

    def header(self):
        self.set_font("DejaVu", "B", 16)
        self.cell(0, 10, "Сравнительный отчёт: Synthetic vs Real", ln=1, align="C")
        self.ln(5)

    def section_title(self, txt):
        self.set_font("DejaVu", "B", 14)
        self.cell(0, 8, txt, ln=1)
        self.ln(2)
        self.set_font("DejaVu", "", 12)

    def add_image(self, path, w=160):
        if os.path.exists(path):
            self.image(path, w=w)
            self.ln(5)
        else:
            self.cell(0, 8, f"[Не найдено изображение: {os.path.basename(path)}]", ln=1)
            self.ln(3)

def main():
    # Проверки
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(f"Лог не найден: {LOG_PATH}")
    os.makedirs(REPORT_DIR, exist_ok=True)

    # Читаем CSV
    df = pd.read_csv(LOG_PATH)
    # Источник: synthetic vs real
    df["source"] = df["file"].apply(lambda x: "synthetic" if x in SYN_FILES else "real")

    # Группировка
    agg = df.groupby("source").agg({
        "file": "count",
        "mse":  "mean",
        "ssim": "mean",
        "anom_fix": "mean",
        "anom_dyn": "mean",
        "anom_pct": "mean"
    }).rename(columns={"file": "count"})

    # Выбираем по 1 файлу с наибольшим mse из каждой группы для примеров
    examples = {}
    for src in ["synthetic", "real"]:
        sub = df[df["source"] == src]
        if not sub.empty:
            top = sub.sort_values("mse", ascending=False).iloc[0]["file"]
            examples[src] = top

    # Создаём PDF
    pdf = PDF()
    pdf.add_page()

    # Раздел 1: агрегированные метрики
    pdf.section_title("1. Агрегированные метрики")
    pdf.set_font("DejaVu", "", 11)
    for src, row in agg.iterrows():
        pdf.cell(0, 6, f"{src.capitalize():<10} | Count={int(row['count'])}  "
                       f"MSE={row['mse']:.6f}  SSIM={row['ssim']:.4f}  "
                       f"Fix={row['anom_fix']:.1f}  Dyn={row['anom_dyn']:.1f}  Pct={row['anom_pct']:.1f}",
                 ln=1)
    pdf.ln(5)

    # Раздел 2: примеры графиков
    pdf.section_title("2. Примеры восстановления и ошибок")
    for src, fname in examples.items():
        pdf.set_font("DejaVu", "B", 12)
        pdf.cell(0, 8, f"{src.capitalize()} example: {fname}", ln=1)
        pdf.set_font("DejaVu", "", 11)

        # Гистограмма ошибок
        hist = os.path.join(VIS_DIR, f"{fname}_error_hist.png")
        pdf.add_image(hist)

        # Реконструкция
        recon = os.path.join(VIS_DIR, f"{fname}_reconstructed.png")
        pdf.add_image(recon)

        pdf.ln(3)

    # Раздел 3: полный список результатов (первые 20)
    pdf.add_page()
    pdf.section_title("3. Таблица результатов (фрагмент)")
    pdf.set_font("DejaVu", "", 9)
    subset = df.head(20)
    for _, r in subset.iterrows():
        line = (f"{r['source']:<9} | {r['file']:<15} | "
                f"MSE={r['mse']:.6f} | SSIM={r['ssim']:.4f} | "
                f"fix={int(r['anom_fix'])} dyn={int(r['anom_dyn'])} pct={int(r['anom_pct'])}")
        pdf.cell(0, 5, line, ln=1)

    # Сохраняем
    pdf.output(REPORT_PATH)
    print(f"✅ Отчёт сохранён: {REPORT_PATH}")

if __name__ == "__main__":
    main()
