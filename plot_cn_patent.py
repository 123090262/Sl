"""Generate a Chinese grayscale metric chart for patent documents."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

METRIC_LABELS = ("准确率", "宏平均F1分数", "科恩κ系数")
METRIC_VALUES = (85.34, 83.37, 80.47)
METRIC_ERRORS = (1.20, 1.80, 1.50)
BAR_COLORS = ("#595959", "#969696", "#D0D0D0")
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
    png_path = output_dir / "sleepgat_metrics_cn_patent.png"
    svg_path = output_dir / "sleepgat_metrics_cn_patent.svg"

    figure, axes = plt.subplots(figsize=(6.5, 5.8))
    axes.bar(
        METRIC_LABELS,
        METRIC_VALUES,
        yerr=METRIC_ERRORS,
        color=BAR_COLORS,
        edgecolor="black",
        linewidth=1.2,
        capsize=8,
        error_kw={
            "ecolor": "black",
            "elinewidth": 1.4,
            "capthick": 1.4,
        },
    )
    axes.set_ylabel("性能指标（%）")
    axes.set_ylim(40, 100)
    axes.grid(axis="y", linestyle="--", color="#BFBFBF", alpha=0.55)
    axes.set_axisbelow(True)

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
