# Chinese Confusion Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python generator for the Chinese blue global confusion matrix.

**Architecture:** Add one focused plotting module that owns the matrix constants, Chinese labels, font setup, rendering, and artifact export. Add one focused pytest file that verifies the public output contract and key source values.

**Tech Stack:** Python, pathlib, matplotlib, pytest.

## Global Constraints

- Keep stage labels as `W`, `N1`, `N2`, `N3`, `REM`.
- Use title `总体混淆矩阵（20名受试者）[%]`.
- Use x-axis label `预测睡眠阶段` and y-axis label `真实睡眠阶段`.
- Use a blue heatmap.
- Write PNG and SVG files under `outputs/figures`.
- Do not modify unrelated documents or legacy scripts.

---

### Task 1: Confusion Matrix Plot Generator

**Files:**
- Create: `plot_confusion_matrix_cn_patent.py`
- Create: `tests/test_plot_confusion_matrix_cn_patent.py`

**Interfaces:**
- Produces: `CONFUSION_MATRIX`, `STAGE_LABELS`, and `create_chart(output_dir: Path) -> tuple[Path, Path]`
- Consumes: Matplotlib with Agg backend and local Chinese fonts if installed.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from plot_confusion_matrix_cn_patent import (
    CONFUSION_MATRIX,
    STAGE_LABELS,
    create_chart,
)


def test_confusion_matrix_values_and_labels_are_preserved() -> None:
    assert STAGE_LABELS == ("W", "N1", "N2", "N3", "REM")
    assert CONFUSION_MATRIX.shape == (5, 5)
    assert CONFUSION_MATRIX[0, 0] == 89.82
    assert CONFUSION_MATRIX[1, 2] == 16.30
    assert CONFUSION_MATRIX[3, 4] == 0.00
    assert CONFUSION_MATRIX[4, 4] == 90.33


def test_create_chart_writes_png_and_svg(tmp_path: Path) -> None:
    png_path, svg_path = create_chart(tmp_path)

    assert png_path == tmp_path / "sleepgat_confusion_matrix_cn_patent.png"
    assert svg_path == tmp_path / "sleepgat_confusion_matrix_cn_patent.svg"
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert png_path.stat().st_size > 20_000

    svg_text = svg_path.read_text(encoding="utf-8")
    assert "<svg" in svg_text
    assert "<path" in svg_text
    assert svg_path.stat().st_size > 10_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `D:\Users\32120\anaconda3\python.exe -m pytest tests/test_plot_confusion_matrix_cn_patent.py -v`

Expected: fail because `plot_confusion_matrix_cn_patent` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create a script that defines the matrix constants, configures Chinese fonts, draws a blue confusion matrix with larger annotations, and writes:

- `sleepgat_confusion_matrix_cn_patent.png`
- `sleepgat_confusion_matrix_cn_patent.svg`

- [ ] **Step 4: Run test to verify it passes**

Run: `D:\Users\32120\anaconda3\python.exe -m pytest tests/test_plot_confusion_matrix_cn_patent.py -v`

Expected: both tests pass.

- [ ] **Step 5: Render final artifacts**

Run: `D:\Users\32120\anaconda3\python.exe plot_confusion_matrix_cn_patent.py`

Expected: PNG and SVG are written to `outputs/figures`.
