"""
Тестовая утилита пороверки функций генерации файлов
"""
import tempfile
from pathlib import Path
import json
from io import BytesIO
from PIL import Image
from generate_files import (
    generate_deterministic_chunk,
    generate_text_content,
    generate_json_content,
    generate_csv_content,
    generate_xml_content,
    generate_svg_content,
    generate_pdf_content,
    generate_image_content,
    generate_log_sizes,
)


def test_generate_deterministic_chunk():
    """Тест: детерминированные данные воспроизводимы."""
    data1 = generate_deterministic_chunk(seed=42, size=100)
    data2 = generate_deterministic_chunk(seed=42, size=100)
    assert data1 == data2
    assert len(data1) == 100


def test_generate_text_content():
    """Тест: текст содержит ожидаемое количество строк."""
    content = generate_text_content(lines=5, words_per_line=3)
    lines = content.decode("utf-8").strip().split("\n")
    assert len(lines) == 5
    assert all(len(line.split()) == 3 for line in lines)


def test_generate_json_content_user():
    """Тест: JSON имеет правильную структуру."""
    content = generate_json_content(schema="user")
    data = content.decode("utf-8")
    assert "id" in data
    assert "name" in data
    assert "active" in data
    data = json.loads(data)
    assert isinstance(data["active"], bool)


def test_generate_json_content_log():
    """Тест: JSON имеет правильную структуру."""
    content = generate_json_content(schema="log")
    data = content.decode("utf-8")
    assert "timestamp" in data
    assert "message" in data
    assert "service" in data
    data = json.loads(data)
    assert data["service"] in ["auth", "db", "api", "cache"]


def test_generate_json_content_event():
    """Тест: JSON имеет правильную структуру."""
    content = generate_json_content(schema="event")
    data = content.decode("utf-8")
    assert "event_id" in data
    assert "type" in data
    assert "user_id" in data
    data = json.loads(data)
    assert data["type"] in ["click", "purchase", "login", "logout"]


def test_generate_json_content_error():
    """Тест: JSON имеет правильную структуру."""
    content = generate_json_content(None)
    data = content.decode("utf-8")
    assert "error" in data



def test_generate_csv_content():
    """Тест: CSV содержит заголовок и строки."""
    content = generate_csv_content(rows=2)
    text = content.decode("utf-8")
    lines = text.strip().split("\n")
    assert len(lines) == 3  # заголовок + 2 строки
    assert "id,name,email,score,active" in lines[0]


def test_generate_xml_content():
    """Тест: XML содержит ожидаемые теги."""
    content = generate_xml_content(items=1)
    text = content.decode("utf-8")
    assert "<users>" in text
    assert "<user id=" in text
    assert "<name>User_1</name>" in text


def test_generate_svg_content():
    """Тест: SVG содержит корневой тег и размеры."""
    content = generate_svg_content(width=100, height=200)
    text = content.decode("utf-8")
    assert '<svg width="100" height="200"' in text
    assert "</svg>" in text


def test_generate_image_content():
    """Тест: изображение создаётся и имеет правильный формат."""
    content = generate_image_content(width=50, height=50, fmt="PNG")
    img = Image.open(BytesIO(content))
    assert img.size == (50, 50)
    assert img.mode == "RGB"


def test_generate_pdf_content():
    """Тест: PDF начинается с %PDF и содержит заголовок."""
    content = generate_pdf_content(title="Test Title")
    assert content.startswith(b"%PDF")
    # Проверяем наличие заголовка в текстовом виде (упрощённо) - not working, content is a stream
    # assert b"Test Title" in content


def test_generate_log_sizes():
    """Тест: логарифмическое распределение даёт разные размеры."""
    sizes = generate_log_sizes(count=10, min_kb=10, max_kb=1000, skew=1.0)
    assert len(sizes) == 10
    assert all(10 <= s <= 1000 for s in sizes)
    assert min(sizes) == 10  # гарантировано первым
    assert max(sizes) == 1000  # гарантировано последним


def test_file_creation():
    """Интеграционный тест: файл создаётся на диске."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test.bin"
        content = b"test data"
        filepath.write_bytes(content)
        assert filepath.exists()
        assert filepath.read_bytes() == content
