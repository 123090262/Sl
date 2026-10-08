from pathlib import Path

from plot_confusion_matrix_cn_patent import (
    CONFUSION_MATRIX,
    STAGE_LABELS,
    create_chart,
)


def test_confusion_matrix_values_and_labels_are_preserved() -> None:
    """Catch accidental changes to the source matrix or stage order."""
    assert STAGE_LABELS == ("W", "N1", "N2", "N3", "REM")
    assert CONFUSION_MATRIX.shape == (5, 5)
    assert CONFUSION_MATRIX[0, 0] == 89.82
    assert CONFUSION_MATRIX[1, 2] == 16.30
    assert CONFUSION_MATRIX[3, 4] == 0.00
    assert CONFUSION_MATRIX[4, 4] == 90.33


def test_create_chart_writes_png_and_svg(tmp_path: Path) -> None:
    """Catch missing or malformed Chinese confusion-matrix artifacts."""
    png_path, svg_path = create_chart(tmp_path)

    assert png_path == tmp_path / "sleepgat_confusion_matrix_cn_patent.png"
    assert svg_path == tmp_path / "sleepgat_confusion_matrix_cn_patent.svg"
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert png_path.stat().st_size > 20_000

    svg_text = svg_path.read_text(encoding="utf-8")
    assert "<svg" in svg_text
    assert "<path" in svg_text
    assert svg_path.stat().st_size > 10_000
