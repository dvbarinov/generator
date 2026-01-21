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
python generate_files.py -n 100 --log-distribution 1 500 2.0

# Типовые файлы
python generate_files.py -n 100 --size 0 --content-type text --text-lines 20 --extension .txt
python generate_files.py -n 100 --size 0 --content-type json --json-schema log --extension .json
python generate_files.py -n 100 --size 0 --content-type image --image-size 640x480 --image-format png --extension .png
"""

import argparse
from pathlib import Path
import hashlib
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random
import json
from io import BytesIO
from PIL import Image, ImageDraw


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


def generate_text_content(lines: int = 10, words_per_line: int = 8) -> bytes:
    """Генерирует случайный текст (похожий на лорем ипсум)."""
    lorem_words = [
        "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit",
        "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore", "et",
        "dolore", "magna", "aliqua", "quis", "nostrud", "exercitation", "ullamco",
        "laboris", "nisi", "aliquip", "ex", "ea", "commodo", "consequat"
    ]
    lines_list = []
    for _ in range(lines):
        line_words = [random.choice(lorem_words) for _ in range(words_per_line)]
        lines_list.append(" ".join(line_words).capitalize() + ".")
    return "\n".join(lines_list).encode("utf-8")


def generate_json_content(schema: str = "user") -> bytes:
    """Генерирует JSON по заданной схеме."""
    if schema == "user":
        data = {
            "id": random.randint(1000, 9999),
            "name": f"User_{random.randint(1, 1000)}",
            "email": f"user{random.randint(1, 1000)}@example.com",
            "active": random.choice([True, False]),
            "tags": random.sample(["admin", "guest", "premium", "trial"], k=random.randint(1, 3))
        }
    elif schema == "log":
        data = {
            "timestamp": f"2026-01-{random.randint(1, 31):02d}\
            T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}Z",
            "level": random.choice(["INFO", "WARN", "ERROR"]),
            "message": f"Operation completed in {random.uniform(10, 500):.2f}ms",
            "service": random.choice(["auth", "db", "api", "cache"])
        }
    elif schema == "event":
        data = {
            "event_id": f"evt_{random.randint(100000, 999999)}",
            "type": random.choice(["click", "purchase", "login", "logout"]),
            "user_id": random.randint(1, 10000),
            "properties": {
                "page": f"/page/{random.randint(1, 20)}",
                "device": random.choice(["mobile", "desktop", "tablet"])
            }
        }
    else:
        data = {"error": "unknown schema", "schema": schema}
    return json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")


def generate_image_content(width: int = 100, height: int = 100, fmt: str = "PNG") -> bytes:
    """Генерирует простое изображение (градиент или цветной блок)."""
    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Простой цветной прямоугольник
    r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    draw.rectangle([0, 0, width, height], fill=(r, g, b))

    # Добавим немного шума
    for _ in range(width * height // 100):
        x, y = random.randint(0, width - 1), random.randint(0, height - 1)
        noise = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.point((x, y), fill=noise)

    buffer = BytesIO()
    img.save(buffer, format=fmt)
    return buffer.getvalue()


def generate_csv_content(rows: int = 100) -> bytes:
    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "name", "email", "score", "active"])

    for i in range(1, rows + 1):
        writer.writerow([
            i,
            f"User_{i}",
            f"user{i}@example.com",
            round(random.uniform(0, 100), 2),
            random.choice(["true", "false"])
        ])

    return output.getvalue().encode("utf-8")


def generate_xml_content(items: int = 50) -> bytes:
    from xml.etree.ElementTree import Element, SubElement, tostring
    import xml.dom.minidom

    root = Element("users")
    for i in range(1, items + 1):
        user = SubElement(root, "user", id=str(i))
        name = SubElement(user, "name")
        name.text = f"User_{i}"
        email = SubElement(user, "email")
        email.text = f"user{i}@example.com"
        score = SubElement(user, "score")
        score.text = str(round(random.uniform(0, 100), 2))

    rough = tostring(root, "utf-8")
    reparsed = xml.dom.minidom.parseString(rough)
    return reparsed.toprettyxml(indent="  ").encode("utf-8")


def create_test_file(args):
    """Создаёт один файл. Принимает кортеж (filepath, size_kb)."""
    filepath, size_kb, content_config = args
    # ext = filepath.suffix.lower()
    ctype = content_config["type"]

    if content_config["type"] == "text":
        content = generate_text_content(
            lines=content_config["text_lines"],
            words_per_line=content_config["text_words_per_line"]
        )
    elif content_config["type"] == "json":
        content = generate_json_content(schema=content_config["json_schema"])
    elif content_config["type"] == "image":
        w, h = content_config["image_size"]
        fmt = content_config["image_format"].upper()
        content = generate_image_content(width=w, height=h, fmt=fmt)
        # Расширение должно соответствовать формату
        # if fmt == "JPEG":
        #     new_path = filepath.with_suffix(".jpg")
        #     new_path.write_bytes(content)
        #     return str(new_path)
    elif ctype == "csv":
        content = generate_csv_content(rows=content_config["csv_rows"])
        filepath = filepath.with_suffix(".csv")
    elif ctype == "xml":
        content = generate_xml_content(items=content_config["xml_items"])
        filepath = filepath.with_suffix(".xml")
    else:
        # Старый режим: бинарные данные
        size_bytes = size_kb * 1024
        if size_bytes == 0:
            content = b""
        else:
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


def parse_outer_args():
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
    parser.add_argument("--content-type", choices=["text", "json", "image", "csv", "xml"],
                        default="binary",
                        help="Тип содержимого: text, json, image (по умолчанию: бинарные данные)")
    parser.add_argument("--text-lines", type=int, default=10,
                        help="Количество строк в текстовом файле")
    parser.add_argument("--text-words-per-line", type=int, default=8, help="Слов в строке")
    parser.add_argument("--json-schema", choices=["user", "log", "event"], default="user",
                        help="Схема JSON")
    parser.add_argument("--image-format", choices=["png", "jpg"], default="png",
                        help="Формат изображения")
    parser.add_argument("--image-size", type=str, default="100x100",
                        help="Размер изображения WxH")
    parser.add_argument("--csv-rows", type=int, default=100, help="Количество строк в CSV")
    parser.add_argument("--xml-items", type=int, default=50, help="Количество элементов в XML")
    args = parser.parse_args()
    return args


def validate_common_args(args):
    """Проверяет общие входящие аргументы."""
    if args.count <= 0:
        raise ValueError("Количество файлов должно быть > 0")
    if args.workers < 1:
        raise ValueError("Число потоков должно быть >= 1")


def validate_size_range(max_size, min_size):
    """Проверяет диапазон размеров."""
    if min_size < 0 or max_size < 0:
        raise ValueError("Размеры не могут быть отрицательными")
    if min_size > max_size:
        raise ValueError("MIN не может быть больше MAX")


def define_bin_size(args, sizes):
    """Определяет размер бинарных файлов."""
    skew = 0.0

    if args.size is not None:
        if args.size < 0:
            raise ValueError("Размер не может быть отрицательным")
        sizes = [args.size] * args.count

    elif args.size_range is not None:
        min_size, max_size = args.size_range
        validate_size_range(max_size, min_size)

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
    return sizes, skew


def main():
    """Основной запуск"""
    args = parse_outer_args()

    validate_common_args(args)

    sizes = []

    # Конфигурация контента
    content_config = {"type": args.content_type}
    if args.content_type == "text":
        content_config.update({
            "text_lines": args.text_lines,
            "text_words_per_line": args.text_words_per_line
        })
        # Для текста игнорируем размер в КБ — используем реальный объём
        sizes = [0] * args.count  # будет перезаписано
    elif args.content_type == "json":
        content_config["json_schema"] = args.json_schema
        sizes = [0] * args.count
    elif args.content_type == "image":
        content_config["image_format"] = args.image_format
        try:
            w, h = map(int, args.image_size.split("x"))
            content_config["image_size"] = (w, h)
        except ValueError as exc:
            raise ValueError(
                "Неверный формат --image-size. Используйте WxH, например: 1920x1080"
            ) from exc
        sizes = [0] * args.count
    elif args.content_type == "csv":
        content_config.update({"csv_rows": args.csv_rows})
        sizes = [0] * args.count
    elif args.content_type == "xml":
        content_config.update({"xml_items": args.xml_items})
        sizes = [0] * args.count
    else:
        # binary — используем sizes как раньше
        pass

    # Определяем список размеров
    sizes, skew = define_bin_size(args, sizes)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    width = len(str(args.count))
    file_tasks = []

    for i in range(1, args.count + 1):
        filename = f"{args.prefix}_{str(i).zfill(width)}{args.extension}"
        filepath = output_dir / filename
        file_tasks.append((filepath, sizes[i - 1], content_config))

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


if __name__ == "__main__":
    main()
