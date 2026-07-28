# -*- coding: utf-8 -*-
"""Полный прогон: сборка датасета -> анализ -> отчёт.

Запуск:  python scripts/run_all.py
"""

from __future__ import annotations

import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    t0 = time.time()
    for step in ("01_build_dataset.py", "02_analyze.py"):
        print(f"\n>>> {step}")
        runpy.run_path(str(HERE / step), run_name="__main__")
    print(f"\nВесь пайплайн выполнен за {time.time() - t0:.1f} с")
    print(f"Отчёт: {HERE.parent / 'reports' / 'index.html'}")


if __name__ == "__main__":
    sys.exit(main())
