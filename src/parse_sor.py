import os
import numpy as np
import matplotlib.pyplot as plt
import otdrparser

# Путь до папки с .sor-файлами
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sor_dir = os.path.join(project_root, "data", "sor_files")
save_dir = os.path.join(project_root, "data", "real")
os.makedirs(save_dir, exist_ok=True)

def parse_sor(filepath):
    with open(filepath, 'rb') as f:
        blocks = otdrparser.parse(f)
    # Извлечение данных трассы из блока 'DATA'
    for block in blocks:
        if block['name'] == 'DATA':
            y = np.array(block['data'])  # амплитуда в дБ
            x = np.linspace(0, len(y) - 1, len(y))  # индексы точек
            return x, y
    raise ValueError("Блок 'DATA' не найден в файле.")

def save_and_plot(name, x, y):
    # Сохраняем как .npy и .csv
    np.save(os.path.join(save_dir, f"{name}.npy"), np.array([x, y]))
    np.savetxt(os.path.join(save_dir, f"{name}.csv"),
               np.column_stack((x, y)), delimiter=",", header="Index,Amplitude(dB)")

    # Строим график
    plt.figure(figsize=(12, 4))
    plt.plot(x, y)
    plt.title(f"Рефлектограмма: {name}")
    plt.xlabel("Индекс")
    plt.ylabel("Амплитуда (дБ)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}.png"))
    plt.close()
    print(f"✅ Обработан: {name}")

# Список файлов (можно сделать автоматическим)
file_list = [
    "kostkovo_1.sor",
    "kostkovo_2.sor",
    "PON_1.sor",
    "PON_2.sor",
    "PON_3.sor",
    "PON_4.sor",
    "St3.sor",
    "St4.sor"
]

for filename in file_list:
    full_path = os.path.join(sor_dir, filename)
    name = filename.replace(".sor", "")
    try:
        x, y = parse_sor(full_path)
        save_and_plot(name, x, y)
    except Exception as e:
        print(f"❌ Ошибка с {filename}: {e}")
