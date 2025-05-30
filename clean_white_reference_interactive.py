#!/usr/bin/env python3
"""
Interactive script to choose a white reference file and crop to only clean rows with no contamination.

Usage:
    python clean_white_reference_interactive.py
"""
import os
from pathlib import Path

import numpy as np
import cv2
import spectral as sp
from scipy.stats import zscore
import tkinter as tk
from tkinter import filedialog, simpledialog


def select_reference_file():
    """Open a dialog to select a hyperspectral ENVI reference file."""
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select Reference File",
        filetypes=[("ENVI Header Files", "*.hdr")]
    )
    return path


def get_output_prefix():
    """Prompt for an output filename prefix."""
    root = tk.Tk()
    root.withdraw()
    prefix = simpledialog.askstring(
        "Filename Prefix",
        "Enter prefix for output file:"
    )
    return prefix.strip() if prefix else ""


def normalize_slice(image, width=1000):
    """
    Normalize the first `width` columns of a multi-band image slice via z-score,
    then rescale to uint8 for display.
    """
    patch = image[:, :width, :].astype(np.float32).copy()
    for c in range(patch.shape[2]):
        flat = zscore(patch[..., c].ravel()).reshape(patch[..., c].shape)
        mn, mx = flat.min(), flat.max()
        if mx > mn:
            flat = np.clip((flat - mn) / (mx - mn) * 255, 0, 255)
        patch[..., c] = flat
    return patch.astype(np.uint8)


def interactive_range_selection(file_path):
    """
    Display a slice of the hyperspectral file and let the user select
    horizontal row ranges to include.

    Returns:
        List of (start_row, end_row) tuples.
    """
    hsi = sp.envi.open(file_path)
    bands = [int(b) for b in hsi.metadata.get("default bands", [])]
    data = hsi.read_bands(bands).copy()
    display = normalize_slice(data)

    selections = []
    temp = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal temp
        if event == cv2.EVENT_LBUTTONDOWN:
            temp = [y]
        elif event == cv2.EVENT_LBUTTONUP and temp:
            temp.append(y)
            y1, y2 = sorted(temp)
            selections.append((y1, y2))
            print(f"Selected rows: {y1}–{y2}")

    cv2.namedWindow("Select Ranges")
    cv2.setMouseCallback("Select Ranges", mouse_callback)

    print("Drag to select regions. 'd' to undo last, 'f' to finish.")
    while True:
        vis = display.copy()
        for start, end in selections:
            cv2.rectangle(vis, (0, start), (display.shape[1] - 1, end), (0, 255, 0), 2)
        cv2.imshow("Select Ranges", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('d') and selections:
            removed = selections.pop()
            print(f"Removed selection: {removed}")
        elif key == ord('f'):
            break

    cv2.destroyAllWindows()
    return selections


def crop_and_save(file_path, selections, prefix, width=1000):
    """
    Crop the selected row ranges and save as a new ENVI file.
    """
    hsi = sp.envi.open(file_path)
    metadata = hsi.metadata.copy()
    data = hsi.load().copy()

    cropped = np.concatenate([
        data[start:end, :width, :]
        for start, end in selections
    ], axis=0)

    metadata['lines'] = cropped.shape[0]
    metadata['samples'] = cropped.shape[1]

    out_dir = Path(file_path).parent / "cropped"
    out_dir.mkdir(parents=True, exist_ok=True)

    name = f"{prefix}_reference_cropped.hdr" if prefix else "reference_cropped.hdr"
    out_path = out_dir / name

    new_img = sp.envi.create_image(
        str(out_path), metadata=metadata,
        dtype=np.float32, force=True
    )
    memmap = new_img.open_memmap(writable=True)
    memmap[...] = cropped.astype(np.float32)
    memmap.flush()
    memmap._mmap.close()

    print(f"Saved cropped reference to: {out_path}")


def main():
    file_path = select_reference_file()
    if not file_path:
        print("No file selected. Exiting.")
        return

    prefix = get_output_prefix()
    selections = interactive_range_selection(file_path)

    if selections:
        crop_and_save(file_path, selections, prefix)
    else:
        print("No regions selected; nothing to save.")


if __name__ == "__main__":
    main()
