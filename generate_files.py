"""
Параллельный генератор тестовых файлов с поддержкой:
- фиксированного размера,
- линейного диапазона,
- случайного диапазона,
- логарифмического распределения.

Использование:
    # Фиксированный размер
    python generate_files.py --count 10 --size 1024

    # Диапазон: 10 файлов от 100 до 1000 КБ (равномерно)
    python generate_files.py --count 10 --size-range 100 1000

    # Случайные размеры в диапазоне
    python generate_files.py --count 100 --size-range 50 200 --random-sizes

    # Логарифмическое распределение: сильный перекос в мелкие файлы (skew=2.0)
    python generate_files.py -n 500 --log-distribution 1 500 2.0

    # С визуализацией
    ## Равномерное
    python generate_test_files.py -n 500 --size-range 100 5000 --plot -o ./uniform

    ## Логарифмическое
    python generate_test_files.py -n 500 --log-distribution 100 5000 --plot -o ./lognorm
"""

import argparse
from pathlib import Path
import hashlib
import random
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


# Импорты для визуализации (опционально)
PLOT_AVAILABLE = True
try:
    import matplotlib.pyplot as plt
    import numpy as np
    from numpy.linalg import LinAlgError
    from scipy.stats import gaussian_kde
except ImportError:
    PLOT_AVAILABLE = False


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


def generate_log_sizes(count: int, min_kb: int, max_kb: int, skew: float = 1.0) -> list[int]:
    """
    Генерирует размеры по логнормальному распределению.
    skew > 1 → больше мелких файлов.
    """
    if min_kb <= 0 or max_kb <= 0:
        raise ValueError("Размеры должны быть > 0 для лог-распределения")
    if min_kb >= max_kb:
        raise ValueError("MIN должен быть < MAX")

    # Фиксируем seed для воспроизводимости
    random.seed(42)

    sizes = []
    for _ in range(count):
        # Генерируем log-normal значение
        # mu=0, sigma=skew — управляет "хвостом"
        u = random.gauss(0, skew)
        val = math.exp(u)
        # Нормализуем в диапазон [min, max]
        normalized = min_kb + (val / (val + 1)) * (max_kb - min_kb)
        sizes.append(int(normalized))

    # Убедимся, что хотя бы один файл близок к min и max
    if count >= 2:
        sizes[0] = min_kb
        sizes[-1] = max_kb

    return sizes


def plot_distribution(sizes: list[int], output_dir: Path):
    """Строит и сохраняет гистограмму распределения размеров."""
    if not PLOT_AVAILABLE:
        print("⚠️  matplotlib не установлен. Установите его для визуализации.")
        return

    _, ax = plt.subplots(figsize=(10, 6))

    # Гистограмма
    ax.hist(sizes, bins=min(50, len(set(sizes))),
            color='skyblue', edgecolor='black', alpha=0.7, density=True)

    # Плотность (ядровое сглаживание)
    try:
        kde = gaussian_kde(sizes)
        x_range = np.linspace(min(sizes), max(sizes), 500)
        ax.plot(x_range, kde(x_range), color='red', linewidth=2, label='Плотность')
        ax.legend()
    except (ValueError, LinAlgError, TypeError):
        pass  # игнорируем, если KDE не сработал

    ax.set_xlabel('Размер файла (КБ)')
    ax.set_ylabel('Плотность')
    ax.set_title('Распределение размеров тестовых файлов')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Логарифмическая шкала по X, если диапазон большой
    if max(sizes) / min(sizes) > 100 and min(sizes) > 0:
        ax.set_xscale('log')
        ax.set_xlabel('Размер файла (КБ, лог. шкала)')

    plt.tight_layout()
    plot_path = output_dir / "size_distribution.png"
    plt.savefig(plot_path, dpi=150)
    print(f"График сохранён: {plot_path}")

    # Показываем окно, только если запущено интерактивно
    if plt.get_backend() != 'agg':
        plt.show()
    plt.close()


def main():
    """Основной запуск"""
    parser = argparse.ArgumentParser(description="Параллельный генератор тестовых файлов")
    parser.add_argument("--count", "-n", type=int, required=True, help="Количество файлов")

    # Группа выбора режима размера
    size_mode = parser.add_mutually_exclusive_group(required=True)
    size_mode.add_argument("--size", "-s", type=int, help="Фиксированный размер файла в КБ")
    size_mode.add_argument("--size-range", type=int, nargs=2, metavar=("MIN", "MAX"),
                           help="Диапазон размеров (равномерный или случайный)")
    size_mode.add_argument("--log-distribution", type=str, nargs='+',
                           metavar=("MIN", "MAX", "SKEW"),
                           help="Логарифмическое распределение: MIN MAX [SKEW=1.0]")

    parser.add_argument("--random-sizes", action="store_true",
                        help="Случайные размеры в --size-range (иначе — равномерные)")
    parser.add_argument("--output", "-o", type=str, default="./test_files", help="Папка вывода")
    parser.add_argument("--prefix", "-p", type=str, default="test", help="Префикс имени файла")
    parser.add_argument("--extension", "-e", type=str, default=".bin", help="Расширение файла")
    parser.add_argument("--workers", "-w", type=int, default=8, help="Число потоков")
    parser.add_argument("--plot", action="store_true",
                        help="Показать и сохранить график распределения размеров")

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

    elif args.size_range is not None:
        min_size, max_size = args.size_range
        if min_size < 0 or max_size < 0:
            raise ValueError("Размеры не могут быть отрицательными")
        if min_size > max_size:
            raise ValueError("MIN не может быть больше MAX")

        if args.random_sizes:
            random.seed(42)
            sizes = [random.randint(min_size, max_size) for _ in range(args.count)]
        else:
            if args.count == 1:
                sizes = [(min_size + max_size) // 2]
            else:
                step = (max_size - min_size) / (args.count - 1)
                sizes = [int(min_size + i * step) for i in range(args.count)]

    elif args.log_distribution is not None:
        if len(args.log_distribution) == 2:
            min_kb, max_kb = map(int, args.log_distribution)
            skew = 1.0
        elif len(args.log_distribution) == 3:
            min_kb, max_kb = int(args.log_distribution[0]), int(args.log_distribution[1])
            skew = float(args.log_distribution[2])
        else:
            raise ValueError("Нужно указать MIN MAX [SKEW] для --log-distribution")

        sizes = generate_log_sizes(args.count, min_kb, max_kb, skew)

    else:
        raise RuntimeError("Не выбран режим размера")

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
        print(f"  Режим: фиксированный размер {args.size} КБ")
    elif args.size_range is not None:
        avg = sum(sizes) / len(sizes)
        print(f"  Режим: {'случайный' if args.random_sizes else 'равномерный'} диапазон")
        print(f"  Размеры: {min(sizes)}–{max(sizes)} КБ (среднее: {avg:.1f} КБ)")
    else:
        avg = sum(sizes) / len(sizes)
        print(f"  Режим: логарифмическое распределение (skew={skew})")
        print(f"  Размеры: {min(sizes)}–{max(sizes)} КБ (среднее: {avg:.1f} КБ)")

    start_time = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Отправляем все задачи
        future_to_path = {executor.submit(create_test_file, task): task[0] for task in file_tasks}

        # Обрабатываем завершённые задачи
        for future in as_completed(future_to_path):
            try:
                future.result()
                completed += 1
                if completed % max(1, args.count // 10) == 0 or completed == args.count:
                    print(f"  Создано: {completed}/{args.count}")
            except (OSError, IOError, ValueError) as exc:
                filepath = future_to_path[future]
                print(f"  Ошибка при создании {filepath}: {exc}")

    elapsed = time.time() - start_time
    total_mb = sum(sizes) / 1024
    speed = total_mb / elapsed if elapsed > 0 else 0

    print(f"\n✅ Готово! Всего: {args.count} файлов ({total_mb:.2f} МБ) за {elapsed:.2f} сек\
            ({speed:.2f} МБ/с)")
    # Визуализация
    if args.plot:
        plot_distribution(sizes, output_dir)

if __name__ == "__main__":
    main()
