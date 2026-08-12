from pathlib import Path

from plot_cn_patent import create_chart


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
