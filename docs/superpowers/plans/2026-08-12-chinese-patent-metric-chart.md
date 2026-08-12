# Chinese Patent Metric Chart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible Python script that renders the existing three evaluation metrics as a Chinese grayscale patent figure in 600 dpi PNG and SVG formats.

**Architecture:** Keep the existing English `plot.py` unchanged. Add one focused plotting module with a callable `create_chart` function and a command-line entry point, plus one small pytest file that verifies both artifacts and their basic structure.

**Tech Stack:** Python, Matplotlib, pytest, pathlib, standard-library warnings

## Global Constraints

- Use the exact values `85.34`, `83.37`, and `80.47`, with errors `1.20`, `1.80`, and `1.50` percentage points.
- Use labels `准确率`, `宏平均F1分数`, `科恩κ系数`, and y-axis label `性能指标（%）`.
- Keep the y-axis range at `40` through `100`.
- Do not modify `plot.py`, training code, model code, or experiment data.
- Render three distinct grayscale bars with black outlines and black capped error bars.
- Save a 600 dpi PNG and an SVG whose text is converted to paths.
- Use a non-interactive Matplotlib backend and warn clearly when no preferred Chinese font is installed.

---

### Task 1: Chinese grayscale chart generator

**Files:**
- Create: `plot_cn_patent.py`
- Create: `tests/test_plot_cn_patent.py`
- Generate: `outputs/figures/sleepgat_metrics_cn_patent.png`
- Generate: `outputs/figures/sleepgat_metrics_cn_patent.svg`

**Interfaces:**
- Consumes: metric values and errors defined as immutable module-level tuples in `plot_cn_patent.py`.
- Produces: `create_chart(output_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]`, returning the PNG path followed by the SVG path.

- [ ] **Step 1: Write the output regression test**

```python
from pathlib import Path

from plot_cn_patent import create_chart


def test_create_chart_writes_png_and_svg(tmp_path: Path) -> None:
    png_path, svg_path = create_chart(tmp_path)

    assert png_path == tmp_path / "sleepgat_metrics_cn_patent.png"
    assert svg_path == tmp_path / "sleepgat_metrics_cn_patent.svg"
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert png_path.stat().st_size > 10_000

    svg_text = svg_path.read_text(encoding="utf-8")
    assert "<svg" in svg_text
    assert "<path" in svg_text
    assert svg_path.stat().st_size > 5_000
```

- [ ] **Step 2: Run the regression test and verify the missing module failure**

Run: `python -m pytest tests/test_plot_cn_patent.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'plot_cn_patent'`.

- [ ] **Step 3: Implement the plotting module**

Implement `plot_cn_patent.py` with these concrete elements:

```python
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
    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in PREFERRED_FONTS if name in installed), None)
    if selected is None:
        warnings.warn(
            "未找到微软雅黑、黑体或思源黑体，中文可能显示不完整。",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        plt.rcParams["font.sans-serif"] = [selected]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "path"


def create_chart(output_dir: Path) -> tuple[Path, Path]:
    _configure_fonts()
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "sleepgat_metrics_cn_patent.png"
    svg_path = output_dir / "sleepgat_metrics_cn_patent.svg"

    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    ax.bar(
        METRIC_LABELS,
        METRIC_VALUES,
        yerr=METRIC_ERRORS,
        color=BAR_COLORS,
        edgecolor="black",
        linewidth=1.2,
        capsize=8,
        error_kw={"ecolor": "black", "elinewidth": 1.4, "capthick": 1.4},
    )
    ax.set_ylabel("性能指标（%）")
    ax.set_ylim(40, 100)
    ax.grid(axis="y", linestyle="--", color="#BFBFBF", alpha=0.55)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, svg_path


if __name__ == "__main__":
    create_chart(Path(__file__).resolve().parent / "outputs" / "figures")
```

- [ ] **Step 4: Run the focused test**

Run: `python -m pytest tests/test_plot_cn_patent.py -v`

Expected: `1 passed`; a font warning is acceptable only when none of the three preferred fonts is installed.

- [ ] **Step 5: Generate the final artifacts and inspect their metadata**

Run: `python plot_cn_patent.py`

Expected: both files appear under `outputs/figures/` and the process exits with code `0`.

Run:

```powershell
Get-Item outputs\figures\sleepgat_metrics_cn_patent.png, outputs\figures\sleepgat_metrics_cn_patent.svg |
    Select-Object Name, Length
```

Expected: PNG is larger than `10,000` bytes and SVG is larger than `5,000` bytes.

- [ ] **Step 6: Perform visual verification**

Open `outputs/figures/sleepgat_metrics_cn_patent.png` and verify all of the following:

- Chinese labels render without empty squares or mojibake.
- No x-axis or y-axis label is clipped.
- The three bars remain distinguishable when viewed in grayscale.
- Error bars are black, capped, and aligned with their bars.
- Values, y-axis limits, and overall ordering match the original chart.

- [ ] **Step 7: Commit the implementation**

```powershell
git add -- plot_cn_patent.py tests/test_plot_cn_patent.py
git commit -m "feat: add Chinese patent metric chart"
```

Do not add generated PNG or SVG files to Git unless the repository later establishes an explicit policy for versioning generated patent assets.
