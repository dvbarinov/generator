"""
Параллельный генератор тестовых файлов.

Использование:
    python generate_files.py --count 1000 --size 1024 --output ./test_data --workers 8
"""

import argparse
import os
from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def generate_deterministic_chunk(seed: int, size: int) -> bytes:
    """Генерирует повторяемый блок данных заданного размера."""
    if size == 0:
        return b""
    data = b""
    counter = 0
    while len(data) < size:
        chunk_seed = f"{seed}:{counter}".encode()
        hash_bytes = hashlib.sha256(chunk_seed).digest()
        data += hash_bytes
        counter += 1
    return data[:size]


def create_test_file(args):
    """Создаёт один файл. Принимает кортеж (filepath, size_kb)."""
    filepath, size_kb = args
    size_bytes = size_kb * 1024
    if size_bytes == 0:
        filepath.write_bytes(b"")
        return str(filepath)
    
    seed = hash(str(filepath.name)) & 0xFFFFFFFF
    content = generate_deterministic_chunk(seed, size_bytes)
    filepath.write_bytes(content)
    return str(filepath)


def main():
    parser = argparse.ArgumentParser(description="Параллельный генератор тестовых файлов")
    parser.add_argument("--count", "-n", type=int, required=True, help="Количество файлов")
    parser.add_argument("--size", "-s", type=int, required=True, help="Размер одного файла в килобайтах")
    parser.add_argument("--output", "-o", type=str, default="./test_files", help="Папка вывода")
    parser.add_argument("--prefix", "-p", type=str, default="test", help="Префикс имени файла")
    parser.add_argument("--extension", "-e", type=str, default=".bin", help="Расширение файла")
    parser.add_argument("--workers", "-w", type=int, default=8, help="Число потоков (по умолчанию: 8)")

    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("Количество файлов должно быть > 0")
    if args.size < 0:
        raise ValueError("Размер не может быть отрицательным")
    if args.workers < 1:
        raise ValueError("Число потоков должно быть >= 1")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    width = len(str(args.count))
    file_tasks = []

    for i in range(1, args.count + 1):
        filename = f"{args.prefix}_{str(i).zfill(width)}{args.extension}"
        filepath = output_dir / filename
        file_tasks.append((filepath, args.size))

    print(f"Генерация {args.count} файлов по {args.size} КБ в '{output_dir}' с {args.workers} потоками...")

    start_time = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Отправляем все задачи
        future_to_path = {executor.submit(create_test_file, task): task[0] for task in file_tasks}

        # Обрабатываем завершённые задачи
        for future in as_completed(future_to_path):
            try:
                filepath = future.result()
                completed += 1
                if completed % max(1, args.count // 10) == 0 or completed == args.count:
                    print(f"  Создано: {completed}/{args.count}")
            except Exception as exc:
                filepath = future_to_path[future]
                print(f"  Ошибка при создании {filepath}: {exc}")

    elapsed = time.time() - start_time
    total_mb = args.count * args.size / 1024
    speed = total_mb / elapsed if elapsed > 0 else 0

    print(f"\n✅ Готово! Всего: {args.count} файлов ({total_mb:.2f} МБ) за {elapsed:.2f} сек ({speed:.2f} МБ/с)")


if __name__ == "__main__":
    main()