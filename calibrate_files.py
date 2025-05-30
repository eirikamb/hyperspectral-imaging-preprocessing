#!/usr/bin/env python3
"""
Batch radiometric calibration of hyperspectral ENVI files using a white reference.

Usage:
    python calibrate_files_using_white_reference.py
"""
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import numpy as np
import spectral as sp


def prompt_files(file_types, title, multiple=True):
    """
    Open a file dialog to select one or more files.
    """
    root = tk.Tk()
    root.withdraw()
    if multiple:
        paths = filedialog.askopenfilenames(title=title, filetypes=file_types)
    else:
        paths = [filedialog.askopenfilename(title=title, filetypes=file_types)]
    return list(paths) if paths else []


def prompt_directory(title):
    """
    Open a dialog to select an output directory.
    """
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title)
    return folder or ''


def compute_white_reference(hdr_path, sample_width=1000):
    """
    Load a white reference ENVI file and compute its mean spectrum over columns.
    """
    img = sp.envi.open(hdr_path)
    data = img.load().astype(np.float32)
    # average over rows, then take specified columns
    mean_spectrum = np.mean(data, axis=0, keepdims=True)
    return mean_spectrum[:, :sample_width, :]


def calibrate_cube(raw_img, white_ref, sample_width=1000):
    """
    Apply radiometric calibration to a hyperspectral cube using white reference.
    """
    raw_data = raw_img.load().astype(np.float32)
    # limit to sample_width columns
    raw_data = raw_data[:, :sample_width, :]
    return raw_data / white_ref


def write_envi_image(output_path, data_cube, metadata, interleave='bil'):
    """
    Save a NumPy hyperspectral cube to an ENVI file with updated metadata.
    """
    metadata = metadata.copy()
    metadata['lines'] = data_cube.shape[0]
    metadata['samples'] = data_cube.shape[1]

    img = sp.envi.create_image(
        hdr_file=str(output_path),
        metadata=metadata,
        interleave=interleave,
        dtype=np.float32,
        force=True
    )
    memmap = img.open_memmap(writable=True)
    memmap[:, :, :] = data_cube
    memmap.flush()
    memmap._mmap.close()


def main():
    # Prompt user selections
    hdr_filter = [("ENVI Headers", "*.hdr")]
    raw_paths = prompt_files(hdr_filter, title="Select Raw ENVI Files to Calibrate")
    if not raw_paths:
        print("No raw files selected. Exiting.")
        return

    white_path = prompt_files(hdr_filter, title="Select White Reference", multiple=False)
    if not white_path:
        print("No white reference selected. Exiting.")
        return
    white_path = white_path[0]

    out_dir = prompt_directory(title="Select Output Directory")
    if not out_dir:
        print("No output directory selected. Exiting.")
        return
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Compute white reference
    print(f"Loading white reference: {Path(white_path).name}")
    white_ref = compute_white_reference(white_path)

    # Process each raw file
    for raw_path in raw_paths:
        raw_name = Path(raw_path).name
        print(f"\nCalibrating {raw_name}...")

        raw_img = sp.envi.open(raw_path)
        calibrated = calibrate_cube(raw_img, white_ref)

        # Construct output filename
        base = raw_name.replace('_raw_rad_float32.hdr', '')
        out_name = f"{base}_calibrated.hdr"
        out_path = Path(out_dir) / out_name

        # Write calibrated cube
        write_envi_image(out_path, calibrated, raw_img.metadata)
        print(f"Saved calibrated image: {out_path}")


if __name__ == '__main__':
    main()
