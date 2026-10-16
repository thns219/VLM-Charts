"""
Script mẫu: Tải và khám phá bộ dữ liệu CharXiv (princeton-nlp/CharXiv)
------------------------------------------------------------------
Cài đặt trước khi chạy:
    pip install datasets pillow matplotlib pandas

Chạy:
    python explore_charxiv.py
"""

import random
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset


def load_charxiv():
    """Tải cả 2 split: validation (có đáp án) và test (không có đáp án công khai)."""
    print("Đang tải split 'validation' (có nhãn, dùng để phát triển/debug)...")
    val_ds = load_dataset("princeton-nlp/CharXiv", split="validation")

    print("Đang tải split 'test' (dùng để đánh giá cuối, không có đáp án công khai)...")
    test_ds = load_dataset("princeton-nlp/CharXiv", split="test")

    print(f"\nSố mẫu validation: {len(val_ds)}")
    print(f"Số mẫu test: {len(test_ds)}")
    return val_ds, test_ds


def inspect_schema(ds):
    """In cấu trúc (tên cột, kiểu dữ liệu) và 1 mẫu đầy đủ."""
    print("\n=== CẤU TRÚC DỮ LIỆU (features) ===")
    print(ds.features)

    print("\n=== MẪU ĐẦU TIÊN (các trường không phải ảnh) ===")
    sample = ds[0]
    for key, value in sample.items():
        if key.lower() not in ("image", "figure", "img"):
            print(f"- {key}: {value}")
        else:
            print(f"- {key}: <đối tượng ảnh, kích thước {value.size if hasattr(value, 'size') else 'N/A'}>")


def find_image_column(ds):
    """Tự dò cột chứa ảnh (tên có thể khác nhau tùy phiên bản dataset)."""
    for col in ds.column_names:
        if col.lower() in ("image", "figure", "img", "figure_path"):
            return col
    return None


def show_random_examples(ds, n=3, save_dir="charxiv_samples"):
    """Lưu ra file ảnh + in câu hỏi/đáp án của vài mẫu ngẫu nhiên để xem trực quan."""
    import os
    os.makedirs(save_dir, exist_ok=True)

    img_col = find_image_column(ds)
    indices = random.sample(range(len(ds)), n)

    for i, idx in enumerate(indices):
        sample = ds[idx]
        print(f"\n--- Mẫu #{idx} ---")
        for key, value in sample.items():
            if key == img_col:
                continue
            print(f"{key}: {value}")

        if img_col and sample[img_col] is not None:
            img = sample[img_col]
            path = f"{save_dir}/sample_{idx}.png"
            img.save(path)
            print(f"Đã lưu ảnh biểu đồ tại: {path}")


def export_all_images(ds, save_dir="charxiv_all_images"):
    """Xuất TOÀN BỘ ảnh biểu đồ trong dataset ra thư mục (dùng cho bước index hoá / đưa vào VLM sau này)."""
    import os
    os.makedirs(save_dir, exist_ok=True)

    img_col = find_image_column(ds)
    if img_col is None:
        print("Không tìm thấy cột ảnh trong dataset — kiểm tra ds.column_names.")
        return

    total = len(ds)
    saved = 0
    for idx in range(total):
        img = ds[idx][img_col]
        if img is None:
            continue
        img.save(f"{save_dir}/chart_{idx:05d}.png")
        saved += 1
        if saved % 200 == 0:
            print(f"  Đã lưu {saved}/{total} ảnh...")

    print(f"Hoàn tất: đã lưu {saved}/{total} ảnh vào thư mục '{save_dir}/'")


def analyze_question_types(ds):
    """
    Thống kê nhanh nếu dataset có cột phân loại câu hỏi
    (ví dụ 'question_type', 'qtype', hoặc tương tự - kiểm tra tên cột thực tế
    bằng inspect_schema() trước rồi chỉnh lại tên cột bên dưới cho khớp).
    """
    possible_cols = [c for c in ds.column_names if "type" in c.lower() or "category" in c.lower()]
    if not possible_cols:
        print("\nKhông tìm thấy cột phân loại câu hỏi rõ ràng — hãy kiểm tra ds.column_names thủ công.")
        return

    for col in possible_cols:
        counts = Counter(ds[col])
        print(f"\n=== Phân bố theo cột '{col}' ===")
        for k, v in counts.most_common():
            print(f"  {k}: {v}")

        # Vẽ biểu đồ cột đơn giản
        plt.figure(figsize=(6, 4))
        plt.bar(list(counts.keys()), list(counts.values()))
        plt.title(f"Phân bố theo {col}")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        out_path = f"distribution_{col}.png"
        plt.savefig(out_path)
        print(f"Đã lưu biểu đồ phân bố tại: {out_path}")


def to_dataframe_preview(ds, n=5):
    """Chuyển n mẫu đầu (bỏ cột ảnh) sang pandas DataFrame để xem dạng bảng."""
    img_col = find_image_column(ds)
    records = []
    for i in range(min(n, len(ds))):
        row = dict(ds[i])
        row.pop(img_col, None)
        records.append(row)
    df = pd.DataFrame(records)
    print("\n=== XEM DẠNG BẢNG (5 mẫu đầu) ===")
    print(df.to_string())
    return df


if __name__ == "__main__":
    val_ds, test_ds = load_charxiv()

    inspect_schema(val_ds)

    to_dataframe_preview(val_ds, n=5)

    analyze_question_types(val_ds)

    # Xem nhanh vài mẫu (kèm in thông tin câu hỏi/đáp án)
    show_random_examples(val_ds, n=3)

    # Xuất TOÀN BỘ ảnh của cả 2 split (validation + test)
    print("\nĐang xuất toàn bộ ảnh của split 'validation'...")
    export_all_images(val_ds, save_dir="charxiv_all_images/validation")

    print("\nĐang xuất toàn bộ ảnh của split 'test'...")
    export_all_images(test_ds, save_dir="charxiv_all_images/test")

    print("\nHoàn tất khám phá dữ liệu.")
    print("- Ảnh mẫu ngẫu nhiên: 'charxiv_samples/'")
    print("- Toàn bộ ảnh: 'charxiv_all_images/validation/' và 'charxiv_all_images/test/'")