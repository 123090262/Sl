from pathlib import Path

from plot_cn_patent import AXIS_LABEL_SIZE, TICK_LABEL_SIZE, create_chart


def test_coordinate_fonts_are_larger_for_document_readability() -> None:
    """Catch regressions that make chart coordinates too small in Word."""
    assert AXIS_LABEL_SIZE >= 16
    assert TICK_LABEL_SIZE >= 15


def test_create_chart_writes_png_and_svg(tmp_path: Path) -> None:
    """Catch missing or malformed patent-chart output artifacts."""
    png_path, svg_path = create_chart(tmp_path)

    assert png_path == tmp_path / "sleepgat_metrics_cn_patent.png"
    assert svg_path == tmp_path / "sleepgat_metrics_cn_patent.svg"
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert png_path.stat().st_size > 10_000

    svg_text = svg_path.read_text(encoding="utf-8")
    assert "<svg" in svg_text
    assert "<path" in svg_text
    assert svg_path.stat().st_size > 5_000
