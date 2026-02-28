#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tagStandard52h13 AprilTag sheet PDF generator (A4)
- Tag size: 2.0 cm (each side)
- Gap between tags: 0.6 cm
- Fills the page with as many tags as fit (centered), continuing to next pages if requested.

Images source (raw PNGs):
https://raw.githubusercontent.com/AprilRobotics/apriltag-imgs/master/tagStandard52h13/tag52_13_00007.png
"""

import os
import math
import argparse
from io import BytesIO

import requests
from PIL import Image

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader


RAW_BASE = (
    "https://raw.githubusercontent.com/AprilRobotics/apriltag-imgs/master/"
    "tagStandard52h13/tag52_13_{:05d}.png"
)

def download_tag_png(tag_id: int, cache_dir: str) -> str:
    """Download tag PNG to cache_dir (if not exists). Return file path."""
    os.makedirs(cache_dir, exist_ok=True)
    fname = f"tag52_13_{tag_id:05d}.png"
    fpath = os.path.join(cache_dir, fname)

    if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
        return fpath

    url = RAW_BASE.format(tag_id)
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Download failed for id={tag_id} (HTTP {r.status_code}) -> {url}")

    # Validate it's an image (basic check)
    try:
        Image.open(BytesIO(r.content)).convert("L")
    except Exception as e:
        raise RuntimeError(f"Downloaded file is not a valid image for id={tag_id}: {e}")

    with open(fpath, "wb") as f:
        f.write(r.content)

    return fpath


def compute_grid(page_w, page_h, tag_size_pt, gap_pt):
    """
    Compute how many cols/rows fit on page, then margins to center the grid.
    Using: total = n*tag + (n-1)*gap
    """
    cols = int((page_w + gap_pt) // (tag_size_pt + gap_pt))
    rows = int((page_h + gap_pt) // (tag_size_pt + gap_pt))
    if cols < 1 or rows < 1:
        raise RuntimeError("Page too small for the given tag size/gap.")

    grid_w = cols * tag_size_pt + (cols - 1) * gap_pt
    grid_h = rows * tag_size_pt + (rows - 1) * gap_pt

    margin_x = (page_w - grid_w) / 2.0
    margin_y = (page_h - grid_h) / 2.0
    return cols, rows, margin_x, margin_y


def generate_pdf(output_pdf: str, start_id: int, pages: int, cache_dir: str):
    page_w, page_h = A4

    tag_size_pt = 2.0 * cm      # 2 cm
    gap_pt = 0.6 * cm           # 0.6 cm

    cols, rows, mx, my = compute_grid(page_w, page_h, tag_size_pt, gap_pt)
    tags_per_page = cols * rows

    c = canvas.Canvas(output_pdf, pagesize=A4)

    current_id = start_id
    for page_idx in range(pages):
        # place from top-left to bottom-right
        for r in range(rows):
            for col in range(cols):
                # PDF origin is bottom-left. We want row 0 at top.
                x = mx + col * (tag_size_pt + gap_pt)
                y = page_h - (my + (r + 1) * tag_size_pt + r * gap_pt)

                png_path = download_tag_png(current_id, cache_dir)
                img = ImageReader(png_path)

                c.drawImage(
                    img,
                    x, y,
                    width=tag_size_pt,
                    height=tag_size_pt,
                    preserveAspectRatio=True,
                    mask='auto'
                )
                current_id += 1

        c.showPage()

    c.save()

    print("Done.")
    print(f"Output: {output_pdf}")
    print(f"A4 grid: {cols} x {rows} = {tags_per_page} tags/page")
    print(f"IDs used: {start_id} .. {current_id - 1}")
    print(f"Cache: {os.path.abspath(cache_dir)}")


def main():
    ap = argparse.ArgumentParser(description="Generate A4 PDF of tagStandard52h13 AprilTags (2cm, 0.6cm gap).")
    ap.add_argument("--output", "-o", default="tagStandard52h13_sheet.pdf", help="Output PDF filename")
    ap.add_argument("--start-id", type=int, default=0, help="First tag ID to place")
    ap.add_argument("--pages", type=int, default=1, help="Number of pages to generate")
    ap.add_argument("--cache-dir", default="apriltag_cache_tagStandard52h13", help="Folder to cache downloaded PNGs")
    args = ap.parse_args()

    generate_pdf(args.output, args.start_id, args.pages, args.cache_dir)


if __name__ == "__main__":
    main()