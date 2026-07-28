# -*- coding: utf-8 -*-
"""Единый стиль графиков и вспомогательные функции сохранения."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import FIGURES

# Палитра: один акцентный цвет + нейтральные, чтобы графики читались как система
INK = "#1b1b1f"
MUTED = "#8a8f98"
GRID = "#e3e5e8"
ACCENT = "#c2410c"        # основной акцент (терракота)
ACCENT_2 = "#0369a1"      # сравнение / «после»
ACCENT_3 = "#15803d"      # положительный эффект
ACCENT_4 = "#b91c1c"      # отрицательный эффект
SERIES = ["#c2410c", "#0369a1", "#15803d", "#a16207", "#7c3aed",
          "#be185d", "#0f766e", "#b91c1c", "#4d7c0f", "#9333ea"]


def setup() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.titleweight": "600",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def clean_axes(ax, spines=("top", "right")) -> None:
    for s in spines:
        ax.spines[s].set_visible(False)


def save(fig, name: str, folder: Path = FIGURES) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"    [фигура] {path.name}")
    return path
