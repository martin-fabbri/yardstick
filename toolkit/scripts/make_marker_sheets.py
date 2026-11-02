#!/usr/bin/env python3
# Copyright 2026 Martin Fabbri
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Printable Yardstick marker sheets: ArUco DICT_5X5_250, one ID per tabloid page.

Each page: the marker (black square 180 mm by default, measured at the outer edge of the black border), its ID and
printing instructions, and a 100 mm bar with 10 mm ticks to check the print at the counter. The layout is the one of
the first printed set (IDs 0-3, 2026-10-31); this script reproduces that set cell for cell.

Bit patterns come from OpenCV's own DICT_5X5_250 (the dictionary the app and the toolkit detect with).

--verify renders every page and checks, with OpenCV's detector, that it decodes as its ID and that the black square
measures the stated size within 1 mm. --check-against compares the black cells of each page with an earlier PDF.

Run from toolkit/:
    uv run --with reportlab --with pypdf --with pypdfium2 python scripts/make_marker_sheets.py \
        --ids 4 5 6 7 --out lotfit-markers-4-7-180mm-tabloid.pdf --verify
"""
from __future__ import annotations

import argparse
import re
import sys

import cv2
import numpy as np
from reportlab.lib.pagesizes import TABLOID
from reportlab.pdfgen import canvas

PT_PER_MM = 72 / 25.4
X0 = 140.8819          # left edge of the marker on the page, pt (first printed set)
TOP = 981.0304         # lower edge of the marker's top row of cells, pt
ROLE = {0: "ID 0 = START: flat on the ground at the back door. The walk starts and ends here."}


def cells(marker_id: int) -> list[list[bool]]:
    """7 x 7 grid (5 x 5 bits plus the black border), True = black."""
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    img = cv2.aruco.generateImageMarker(d, marker_id, 7)
    return [[bool(img[r, c] == 0) for c in range(7)] for r in range(7)]


def draw_page(c: canvas.Canvas, marker_id: int, side_mm: float) -> None:
    cell = side_mm * PT_PER_MM / 7
    c.setFillColorRGB(0, 0, 0)
    c.setStrokeColorRGB(0, 0, 0)
    for r, row in enumerate(cells(marker_id)):
        for k, black in enumerate(row):
            if black:
                c.rect(X0 + k * cell, TOP - r * cell, cell, cell, stroke=0, fill=1)
    size = f"{side_mm:g} mm"
    c.setFont("Helvetica-Bold", 28)
    c.drawString(X0, 472.8189, f"LotFit marker  ID {marker_id}")
    c.setFont("Helvetica", 11)
    c.drawString(X0, 450.1417, f"ArUco DICT_5X5_250. Black square = {size} (outer edge of the black border).")
    c.drawString(X0, 433.1339, f"Print at 100% / ACTUAL SIZE. Matte. Keep all white margins. In the app set this ID to {size}.")
    if marker_id in ROLE:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(X0, 413.2913, ROLE[marker_id])
    c.setLineWidth(1.2)
    y = 373.6063
    c.line(X0, y, X0 + 100 * PT_PER_MM, y)
    for i in range(11):
        x = X0 + i * 10 * PT_PER_MM
        c.line(x, y, x, y + (22.6772 if i in (0, 5, 10) else 14.1732))
    c.setFont("Helvetica", 10)
    c.drawString(X0, 356.5984, "100 mm: check with a ruler at the counter. Not exactly 100 mm -> reprint at actual size.")
    c.showPage()


def write_pdf(path: str, ids: list[int], side_mm: float) -> None:
    c = canvas.Canvas(path, pagesize=TABLOID)
    c.setTitle("LotFit markers " + ", ".join(map(str, ids)))
    for i in ids:
        draw_page(c, i, side_mm)
    c.save()


def verify(path: str, ids: list[int], side_mm: float, dpi: int = 300) -> bool:
    import pypdfium2 as pdfium

    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250))
    pdf = pdfium.PdfDocument(path)
    ok = len(pdf) == len(ids)
    for n, want in enumerate(ids):
        img = np.array(pdf[n].render(scale=dpi / 72).to_pil().convert("L"))
        corners, found, _ = det.detectMarkers(img)
        got = [] if found is None else found.ravel().tolist()
        side = None
        if got == [want]:
            q = corners[0][0]
            side = float(np.mean([np.linalg.norm(q[k] - q[(k + 1) % 4]) for k in range(4)])) / dpi * 25.4
        good = got == [want] and side is not None and abs(side - side_mm) <= 1.0
        ok &= good
        print(f"page {n + 1}: decoded {got}, black square {side and round(side, 1)} mm -> {'ok' if good else 'FAIL'}")
    return ok


def black_cells(pdf_page) -> list[tuple[float, float]]:
    s = pdf_page.get_contents().get_data().decode("latin1")
    return sorted((round(float(a), 1), round(float(b), 1)) for a, b in re.findall(r"n ([\d.]+) ([\d.]+) [\d.]+ [\d.]+ re f\*", s))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ids", type=int, nargs="+", required=True, help="marker IDs, one page each")
    ap.add_argument("--out", required=True, help="output PDF")
    ap.add_argument("--size-mm", type=float, default=180.0, help="black square side (default 180)")
    ap.add_argument("--verify", action="store_true", help="render and decode every page")
    ap.add_argument("--check-against", help="an earlier PDF with the same IDs in the same order")
    a = ap.parse_args()
    write_pdf(a.out, a.ids, a.size_mm)
    print("wrote", a.out)
    ok = True
    if a.verify:
        ok &= verify(a.out, a.ids, a.size_mm)
    if a.check_against:
        import pypdf

        old, new = pypdf.PdfReader(a.check_against), pypdf.PdfReader(a.out)
        for n, i in enumerate(a.ids):
            same = black_cells(old.pages[n]) == black_cells(new.pages[n])
            ok &= same
            print(f"ID {i}: black cells identical to {a.check_against}: {same}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
