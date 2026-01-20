"""
Параллельный генератор тестовых файлов с поддержкой диапазона размеров.

Использование:
    # Фиксированный размер
    python generate_files.py --count 10 --size 1024

    # Диапазон: 10 файлов от 100 до 1000 КБ (равномерно)
    python generate_files.py --count 10 --size-range 100 1000

    # Случайные размеры в диапазоне
    python generate_files.py --count 100 --size-range 50 200 --random-sizes
"""

import argparse
import os
from pathlib import Path
import hashlib
import random
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
    
    # Взаимоисключающие аргументы: либо --size, либо --size-range
    size_group = parser.add_mutually_exclusive_group(required=True)
    size_group.add_argument("--size", "-s", type=int, help="Фиксированный размер файла в КБ")
    size_group.add_argument("--size-range", type=int, nargs=2, metavar=("MIN", "MAX"),
                            help="Диапазон размеров в КБ (например: 100 500)")

    parser.add_argument("--random-sizes", action="store_true",
                        help="Если указано с --size-range, размеры будут случайными")
    parser.add_argument("--output", "-o", type=str, default="./test_files", help="Папка вывода")
    parser.add_argument("--prefix", "-p", type=str, default="test", help="Префикс имени файла")
    parser.add_argument("--extension", "-e", type=str, default=".bin", help="Расширение файла")
    parser.add_argument("--workers", "-w", type=int, default=8, help="Число потоков")

    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("Количество файлов должно быть > 0")
    if args.workers < 1:
        raise ValueError("Число потоков должно быть >= 1")

    # Определяем список размеров
    if args.size is not None:
        if args.size < 0:
            raise ValueError("Размер не может быть отрицательным")
        sizes = [args.size] * args.count
    else:
        min_size, max_size = args.size_range
        if min_size < 0 or max_size < 0:
            raise ValueError("Размеры не могут быть отрицательными")
        if min_size > max_size:
            raise ValueError("MIN не может быть больше MAX")

        if args.random_sizes:
            # Случайные размеры в диапазоне
            random.seed(42)  # для воспроизводимости
            sizes = [random.randint(min_size, max_size) for _ in range(args.count)]
        else:
            # Равномерное распределение
            if args.count == 1:
                sizes = [(min_size + max_size) // 2]
            else:
                step = (max_size - min_size) / (args.count - 1)
                sizes = [int(min_size + i * step) for i in range(args.count)]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    width = len(str(args.count))
    file_tasks = []

    for i in range(1, args.count + 1):
        filename = f"{args.prefix}_{str(i).zfill(width)}{args.extension}"
        filepath = output_dir / filename
        file_tasks.append((filepath, sizes[i - 1]))

    print(f"Генерация {args.count} файлов в '{output_dir}' с {args.workers} потоками...")
    if args.size is not None:
        print(f"  Размер: {args.size} КБ (фиксированный)")
    else:
        avg_size = sum(sizes) / len(sizes)
        print(f"  Размеры: от {min(sizes)} до {max(sizes)} КБ (среднее: {avg_size:.1f} КБ)")
        if args.random_sizes:
            print("  Режим: случайные размеры")
        else:
            print("  Режим: равномерное распределение")

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
    total_mb = sum(sizes) / 1024
    speed = total_mb / elapsed if elapsed > 0 else 0

    print(f"\n✅ Готово! Всего: {args.count} файлов ({total_mb:.2f} МБ) за {elapsed:.2f} сек ({speed:.2f} МБ/с)")


if __name__ == "__main__":
    main()