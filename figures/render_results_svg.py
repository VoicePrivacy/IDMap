"""Render the IDMap paper result tables as resolution-independent SVG.

The numbers below are transcribed from the three original paper tables in
figures/*.png. Keep those PNGs as provenance when updating these SVGs.
"""

from html import escape
from pathlib import Path


HERE = Path(__file__).resolve().parent
FONT = "Georgia, 'Times New Roman', serif"


def svg_table(filename, title, columns, rows, widths, *, notes="", emphasis=()):
    margin = 22
    title_y = 37
    top = 65
    header_h = 51
    row_h = 38
    footer_h = 38 if notes else 13
    width = sum(widths) + 2 * margin
    height = top + header_h + len(rows) * row_h + footer_h
    edges = [margin]
    for column_width in widths:
        edges.append(edges[-1] + column_width)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title, quote=True)}">',
        f'<title>{escape(title)}</title>',
        f'<desc>Vector recreation of the corresponding IDMap paper table. '
        f'All values remain text and can be selected at any zoom level.</desc>',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{margin}" y="{title_y}" font-family="{FONT}" font-size="23" '
        f'font-weight="700" fill="#172633">{escape(title)}</text>',
        f'<rect x="{margin}" y="{top}" width="{sum(widths)}" height="{header_h}" '
        f'fill="#eaf0f5"/>',
        f'<path d="M {margin} {top} H {edges[-1]} M {margin} {top + header_h} '
        f'H {edges[-1]}" fill="none" stroke="#34475a" stroke-width="1.3"/>',
    ]
    for i, name in enumerate(columns):
        x = (edges[i] + edges[i + 1]) / 2
        parts.append(
            f'<text x="{x}" y="{top + 32}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="15.5" font-weight="700" '
            f'fill="#172633">{escape(name)}</text>'
        )
    for row_i, row in enumerate(rows):
        y = top + header_h + row_i * row_h
        if row_i % 2:
            parts.append(
                f'<rect x="{margin}" y="{y}" width="{sum(widths)}" '
                f'height="{row_h}" fill="#f8fafb"/>'
            )
        if row_i and row[0] != rows[row_i - 1][0]:
            parts.append(
                f'<path d="M {margin} {y} H {edges[-1]}" '
                f'stroke="#8495a4" stroke-width="1.2"/>'
            )
        for col_i, value in enumerate(row):
            x = (edges[col_i] + edges[col_i + 1]) / 2
            bold = ' font-weight="700"' if (row_i, col_i) in emphasis else ""
            parts.append(
                f'<text x="{x}" y="{y + 25}" text-anchor="middle" '
                f'font-family="{FONT}" font-size="16" fill="#172633"{bold}>'
                f'{escape(value)}</text>'
            )
    bottom = top + header_h + len(rows) * row_h
    parts.append(
        f'<path d="M {margin} {bottom} H {edges[-1]}" '
        f'stroke="#34475a" stroke-width="1.3"/>'
    )
    if notes:
        parts.append(
            f'<text x="{margin}" y="{bottom + 25}" font-family="{FONT}" '
            f'font-size="13.5" fill="#435568">{escape(notes)}</text>'
        )
    parts.append("</svg>\n")
    (HERE / filename).write_text("\n".join(parts), encoding="utf-8")


def main():
    systems = ["RS", "Average", "PSD", "GAN N", "GAN U", "MLP N", "MLP U", "Diff N", "Diff U"]
    svg_table(
        "EER_WER_UAR.svg",
        "Privacy and speech utility (IDMap paper, Table I)",
        ["Metric", "Dataset", "Gender", *systems],
        [
            ["EER", "libri-dev", "f", "41.47", "41.79", "45.49", "42.47", "43.59", "46.16", "46.32", "48.47", "48.21"],
            ["EER", "libri-dev", "m", "43.22", "39.60", "44.31", "44.22", "40.77", "46.60", "44.24", "47.86", "47.36"],
            ["EER", "libri-test", "f", "41.54", "39.43", "44.33", "43.54", "44.62", "43.61", "43.81", "48.18", "47.66"],
            ["EER", "libri-test", "m", "40.58", "39.87", "42.29", "40.60", "41.99", "45.83", "46.32", "48.46", "47.34"],
            ["EER", "average", "–", "41.70", "40.17", "43.88", "42.70", "42.74", "45.54", "45.17", "48.24", "47.64"],
            ["WER", "libri-dev", "–", "3.31", "3.37", "3.39", "3.38", "3.45", "3.37", "3.38", "3.38", "3.41"],
            ["WER", "libri-test", "–", "3.21", "3.23", "3.23", "3.28", "3.23", "3.23", "3.25", "3.22", "3.28"],
            ["UAR", "IEMOCAP-dev", "–", "53.47", "52.25", "53.42", "53.48", "52.74", "53.67", "52.79", "53.78", "54.23"],
            ["UAR", "IEMOCAP-test", "–", "52.21", "52.11", "53.88", "53.43", "53.23", "52.63", "53.88", "52.85", "53.01"],
        ],
        [75, 126, 75] + [95] * 9,
        notes="All values are percentages. N and U denote normal and uniform sampling; UAR is measured in SER.",
        emphasis={(4, 10)},
    )
    svg_table(
        "Gvd.svg",
        "Voice distinctness and de-identification (IDMap paper, Table II)",
        ["Metric", *systems],
        [
            ["Gvd (dB)", "−1.684", "−3.056", "0.387", "−0.137", "−0.144", "0.412", "0.406", "0.521", "0.513"],
            ["DeID (%)", "98.43", "98.23", "99.24", "98.93", "98.75", "99.49", "99.31", "99.76", "99.96"],
        ],
        [125] + [98] * 9,
        notes="Pooled LibriSpeech development and test subsets; N and U denote normal and uniform sampling.",
    )
    svg_table(
        "RTFs.svg",
        "Real-time factor (IDMap paper, Table III)",
        ["Average", "PSD", "GAN", "IDMap-MLP", "IDMap-Diff"],
        [["2.113 × 10⁻⁴", "3.384", "0.0419", "6.417 × 10⁻⁴", "2.43 × 10⁻³"]],
        [172] * 5,
        notes="RTF measures pseudo-speaker vector generation, not end-to-end audio synthesis.",
    )


if __name__ == "__main__":
    main()
