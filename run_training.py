#!/usr/bin/env python
"""
Скрипт для запуска обучения модели
"""

import sys
from pathlib import Path

# Добавляем путь к src
sys.path.append(str(Path(__file__).parent / "src"))

from src.train import main

if __name__ == "__main__":
    main()