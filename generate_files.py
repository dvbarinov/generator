"""
Генератор тестовых файлов для загрузчика.

Использование:
    python generate_files.py --count 100 --size 1024 --output ./test_data

Создаст 100 файлов по 1024 КБ (1 МБ) в папке ./test_data/
"""

import argparse
import os
from pathlib import Path
import struct
import hashlib


def generate_deterministic_chunk(seed: int, size: int) -> bytes:
    """
    Генерирует повторяемый блок данных заданного размера.
    Использует хеш как "псевдослучайный" источник.
    """
    data = b""
    counter = 0
    while len(data) < size:
        # Хешируем seed + counter → получаем 32 байта
        chunk_seed = f"{seed}:{counter}".encode()
        hash_bytes = hashlib.sha256(chunk_seed).digest()
        data += hash_bytes
        counter += 1
    return data[:size]


def create_test_file(filepath: Path, size_kb: int):
    """Создаёт один файл заданного размера (в килобайтах)"""
    size_bytes = size_kb * 1024
    if size_bytes == 0:
        filepath.write_bytes(b"")
        return

    # Генерируем данные с seed = hash от имени файла (для детерминизма)
    seed = hash(str(filepath.name)) & 0xFFFFFFFF
    content = generate_deterministic_chunk(seed, size_bytes)
    filepath.write_bytes(content)


def main():
    parser = argparse.ArgumentParser(description="Генератор тестовых файлов")
    parser.add_argument("--count", "-n", type=int, required=True, help="Количество файлов")
    parser.add_argument("--size", "-s", type=int, required=True, help="Размер одного файла в килобайтах")
    parser.add_argument("--output", "-o", type=str, default="./test_files", help="Папка вывода")
    parser.add_argument("--prefix", "-p", type=str, default="test", help="Префикс имени файла")
    parser.add_argument("--extension", "-e", type=str, default=".bin", help="Расширение файла")

    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("Количество файлов должно быть > 0")
    if args.size < 0:
        raise ValueError("Размер не может быть отрицательным")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Определяем ширину номера (для ведущих нулей)
    width = len(str(args.count))

    print(f"Генерация {args.count} файлов по {args.size} КБ в '{output_dir}'...")

    for i in range(1, args.count + 1):
        filename = f"{args.prefix}_{str(i).zfill(width)}{args.extension}"
        filepath = output_dir / filename
        create_test_file(filepath, args.size)
        if i % 100 == 0 or i == args.count:
            print(f"  Создано: {i}/{args.count}")

    total_mb = args.count * args.size / 1024
    print(f"\n✅ Готово! Всего создано: {args.count} файлов ({total_mb:.2f} МБ)")


if __name__ == "__main__":
    main()