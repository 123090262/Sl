"""Generate a Chinese blue confusion matrix for patent documents."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

STAGE_LABELS = ("W", "N1", "N2", "N3", "REM")
CONFUSION_MATRIX = np.array(
    [
        [89.82, 7.23, 1.03, 0.24, 1.68],
        [10.07, 61.57, 16.30, 0.50, 11.55],
        [0.64, 3.67, 87.53, 4.52, 3.64],
        [0.11, 0.16, 11.12, 88.62, 0.00],
        [1.30, 3.73, 4.60, 0.04, 90.33],
    ],
    dtype=float,
)
PREFERRED_FONTS = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC")


def _configure_fonts() -> None:
    """Select an installed Chinese font and preserve it in vector output."""
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = next(
        (name for name in PREFERRED_FONTS if name in installed_fonts),
        None,
    )
    if selected_font is None:
        warnings.warn(
            "未找到微软雅黑、黑体或思源黑体，中文可能显示不完整。",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        plt.rcParams["font.sans-serif"] = [selected_font]

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "path"


def create_chart(output_dir: Path) -> tuple[Path, Path]:
    """Render the chart and return the generated PNG and SVG paths."""
    _configure_fonts()
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "sleepgat_confusion_matrix_cn_patent.png"
    svg_path = output_dir / "sleepgat_confusion_matrix_cn_patent.svg"

    figure, axes = plt.subplots(figsize=(7.2, 6.2))
    image = axes.imshow(CONFUSION_MATRIX, cmap="Blues", vmin=0, vmax=100)

    axes.set_title("总体混淆矩阵（20名受试者）[%]", fontsize=15, pad=12)
    axes.set_xlabel("预测睡眠阶段", fontsize=13)
    axes.set_ylabel("真实睡眠阶段", fontsize=13)
    axes.set_xticks(np.arange(len(STAGE_LABELS)), labels=STAGE_LABELS)
    axes.set_yticks(np.arange(len(STAGE_LABELS)), labels=STAGE_LABELS)
    axes.tick_params(axis="both", labelsize=12)

    threshold = 55.0
    for row_index in range(CONFUSION_MATRIX.shape[0]):
        for col_index in range(CONFUSION_MATRIX.shape[1]):
            value = CONFUSION_MATRIX[row_index, col_index]
            text_color = "white" if value >= threshold else "#303030"
            axes.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
            )

    colorbar = figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=11)

    figure.tight_layout()
    figure.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png_path, svg_path


if __name__ == "__main__":
    create_chart(Path(__file__).resolve().parent / "outputs" / "figures")
