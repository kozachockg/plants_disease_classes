#!/usr/bin/env python
"""
Скрипт для запуска инференса
"""

import sys
from pathlib import Path

# Добавляем путь к src
sys.path.append(str(Path(__file__).parent / "src"))

from src.predict import main

if __name__ == "__main__":
    main()