# Chinese Confusion Matrix Design

## Goal

Generate a Chinese version of the global sleep-stage confusion matrix for patent or report use, with larger numeric annotations and blue heatmap coloring.

## Source Values

Rows are true sleep stages and columns are predicted sleep stages. Values are percentages:

```text
          W     N1     N2     N3    REM
W     89.82   7.23   1.03   0.24   1.68
N1    10.07  61.57  16.30   0.50  11.55
N2     0.64   3.67  87.53   4.52   3.64
N3     0.11   0.16  11.12  88.62   0.00
REM    1.30   3.73   4.60   0.04  90.33
```

## Labels

- Title: `总体混淆矩阵（20名受试者）[%]`
- X axis: `预测睡眠阶段`
- Y axis: `真实睡眠阶段`
- Stage labels: `W`, `N1`, `N2`, `N3`, `REM`

The sleep-stage abbreviations remain in English because they are standard labels and keep the matrix compact.

## Visual Requirements

- Use a blue heatmap color scale.
- Show each cell value with two decimal places.
- Increase annotation font size relative to the provided image.
- Preserve readable contrast by using white text on dark cells and dark text on light cells.
- Export both 600 dpi PNG and SVG to `outputs/figures`.
