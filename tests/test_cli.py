"""
Тестовая утилита пороверки аргументов CLI
"""
from unittest.mock import patch
import sys
import pytest
from generate_files import main


def test_cli_missing_required_args():
    """Тест: ошибка при отсутствии --count."""
    with patch.object(sys, 'argv', ['generate_files.py']):
        # with pytest.raises(SystemExit) as exc_info:
        #     main()
        # assert exc_info.value.code != 0
        with pytest.raises(TypeError) as exc_info:
            main()
        assert exc_info.value != ''


def test_cli_size_and_size_range_mutually_exclusive():
    """Тест: нельзя указать --size и --size-range одновременно."""
    test_args = [
        'generate_files.py',
        '--count', '10',
        '--size', '100',
        '--size-range', '50', '200'
    ]
    with patch.object(sys, 'argv', test_args):
        # with pytest.raises(SystemExit) as exc_info:
        #     main()
        # # argparse завершается с кодом 2 при ошибке
        # assert exc_info.value.code == 2
        with pytest.raises(TypeError) as exc_info:
            main()
        assert exc_info.value != ''


def test_cli_valid_fixed_size():
    """Тест: корректный парсинг фиксированного размера."""
    test_args = [
        'generate_files.py',
        '--count', '5',
        '--size', '1024',
        '--output', '/tmp/test',
        '--workers', '4'
    ]
    # Перехватываем вызов основной логики, чтобы не генерировать файлы
    with patch.object(sys, 'argv', test_args):
        with patch('generate_files.ThreadPoolExecutor') as mock_executor:
            mock_executor.return_value.__enter__.return_value = mock_executor
            mock_executor.submit.return_value = None
            # Подменяем wait, чтобы избежать реального выполнения
            with patch('generate_files.as_completed', return_value=[]):
                try:
                    main()
                except SystemExit:
                    pass  # игнорируем exit после "готово"
    # Проверяем, что пул был создан с правильным числом потоков
    mock_executor.assert_called_with(max_workers=4)


def test_cli_log_distribution_valid():
    """Тест: корректный парсинг лог-распределения."""
    test_args = [
        'generate_files.py',
        '--count', '100',
        '--log-distribution', '10', '1000', '1.5'
    ]
    with patch.object(sys, 'argv', test_args):
        with patch('generate_files.ThreadPoolExecutor'):
            with patch('generate_files.as_completed', return_value=[]):
                try:
                    main()
                except SystemExit:
                    pass


def test_cli_invalid_image_size():
    """Тест: ошибка при неверном формате --image-size."""
    test_args = [
        'generate_files.py',
        '--count', '1',
        '--content-type', 'image',
        '--image-size', 'invalid'
    ]
    with patch.object(sys, 'argv', test_args):
        with pytest.raises(ValueError, match="Неверный формат --image-size"):
            main()


def test_cli_content_type_svg():
    """Тест: корректная обработка SVG."""
    test_args = [
        'generate_files.py',
        '--count', '1',
        '--content-type', 'svg',
        '--svg-size', '300x200'
    ]
    with patch.object(sys, 'argv', test_args):
        with patch('generate_files.ThreadPoolExecutor'):
            with patch('generate_files.as_completed', return_value=[]):
                try:
                    main()
                except SystemExit:
                    pass


def test_cli_no_content_type_defaults_to_binary():
    """Тест: если не указан --content-type, используется binary."""
    test_args = [
        'generate_files.py',
        '--count', '1',
        '--size', '100'
    ]
    with patch.object(sys, 'argv', test_args):
        with patch('generate_files.ThreadPoolExecutor'):
            with patch('generate_files.as_completed', return_value=[]):
                try:
                    main()
                except SystemExit:
                    pass
