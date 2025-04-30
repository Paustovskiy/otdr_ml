import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def extract_trace(filepath, out_dir, name):
    with open(filepath, "rb") as f:
        raw = f.read()

    # Ищем сигнатуру блока "DataPts"
    keyword = b"DataPts"
    start = raw.find(keyword)
    if start == -1:
        print(f"❌ 'DataPts' не найден в {name}")
        return

    # После "DataPts" идут 2 байта padding, затем длина блока (4 байта, little-endian)
    block_len_offset = start + len(keyword) + 2
    block_len = int.from_bytes(raw[block_len_offset:block_len_offset+4], byteorder="little")

    # Затем идут сами данные трассы
    data_offset = block_len_offset + 4
    raw_data = raw[data_offset:data_offset + block_len]

    # Преобразуем байты в массив (пробуем uint8, uint16, int16 — как визуально ближе)
    signal = np.frombuffer(raw_data, dtype=np.uint8)  # можно поменять на int16

    # Сохраняем
    np.save(os.path.join(out_dir, f"{name}.npy"), signal)
    np.savetxt(os.path.join(out_dir, f"{name}.csv"),
               np.column_stack((np.arange(len(signal)), signal)),
               delimiter=",", header="Index,Amplitude")

    # Строим график
    plt.figure(figsize=(12, 4))
    plt.plot(signal)
    plt.title(f"Рефлектограмма: {name}")
    plt.xlabel("Индекс")
    plt.ylabel("Амплитуда")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}.png"))
    plt.close()
    print(f"✅ Готово: {name} ({len(signal)} точек)")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(root, "data", "sor_files")  # положи сюда .sor
    output_dir = os.path.join(root, "data", "real")
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".sor"):
            extract_trace(os.path.join(input_dir, filename),
                          output_dir,
                          name=filename.replace(".sor", ""))