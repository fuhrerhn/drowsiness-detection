"""Menampilkan jumlah gambar pada dataset train, val, dan test."""

from pathlib import Path

DATASET_DIRECTORY = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images():
    print("JUMLAH GAMBAR DATASET\n")
    print(f"{'Split':<10} {'Awake':>8} {'Sleepy':>8} {'Total':>8}")
    print("-" * 38)

    total_images = 0
    for split_name in ("train", "val", "test"):
        counts = {}
        for class_name in ("awake", "sleepy"):
            class_directory = DATASET_DIRECTORY / split_name / class_name
            if not class_directory.is_dir():
                raise FileNotFoundError(f"Folder dataset tidak ditemukan: {class_directory}")

            counts[class_name] = sum(
                1 for file in class_directory.iterdir()
                if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
            )

        split_total = counts["awake"] + counts["sleepy"]
        total_images += split_total
        print(f"{split_name:<10} {counts['awake']:>8} {counts['sleepy']:>8} {split_total:>8}")

    print("-" * 38)
    print(f"Total gambar: {total_images}")


if __name__ == "__main__":
    count_images()
