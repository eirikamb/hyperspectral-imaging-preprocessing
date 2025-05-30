import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
import spectral as sp
import os

# ---- Utility Functions ----

def extract_marker_mask_rule_based(hsi, wavelengths,
                                   low_thresh=0.6, high_thresh=0.9,
                                   low_max=600, high_min=850):
    h, w, b = hsi.shape
    low_idx  = np.where(wavelengths < low_max)[0]
    high_idx = np.where(wavelengths > high_min)[0]

    pix = hsi.reshape(-1, b)
    low_ok  = np.all(pix[:, low_idx]  < low_thresh, axis=1)
    high_ok = np.all(pix[:, high_idx] > high_thresh, axis=1)

    mask = (low_ok & high_ok).reshape(h, w)
    return mask.astype(np.uint8)


def estimate_shift_from_masks(mask1, mask2,
                              x_range=(0,150), y_range=(-300,300),
                              visualize=False):
    h, w = mask1.shape
    best_score = -1
    best_shift = (0, 0)

    for dx in range(*x_range):
        for dy in range(y_range[0], y_range[1]):
            y1, y2 = max(0, dy), max(0, -dy)
            height = min(h - y1, h - y2)
            if height <= 0 or dx >= w:
                continue

            p1 = mask1[y1:y1+height, -dx:]
            p2 = mask2[y2:y2+height, :dx]
            if p1.shape != p2.shape:
                continue

            ov = np.sum((p1>0)&(p2>0))
            if ov > best_score:
                best_score = ov
                best_shift = (dx, dy)

    if visualize:
        print(f"[Marker] Best shift: x={best_shift[0]}, y={best_shift[1]}, overlap={best_score}")
    return best_shift, best_score


def linear_5_percent_normalize(img):
    out = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[2]):
        band = img[:,:,i].astype(np.float32)
        lo, hi = np.percentile(band, 5), np.percentile(band, 95)
        norm = np.clip((band - lo)/(hi - lo + 1e-8), 0,1)
        out[:,:,i] = (norm*255).astype(np.uint8)
    return out


def create_half_overlap_mosaic(hsi1, hsi2, shift_x, shift_y):
    y1, y2 = max(0, shift_y), max(0, -shift_y)
    h = min(hsi1.shape[0]-y1, hsi2.shape[0]-y2)
    left_crop  = hsi1[y1:y1+h, :, :]
    right_crop = hsi2[y2:y2+h, :, :]

    dx = shift_x
    if dx <= 0 or dx > left_crop.shape[1]:
        raise ValueError("Invalid horizontal shift")

    half = dx // 2
    W1 = left_crop.shape[1]
    left_sec  = left_crop[:, :W1-half, :]
    right_sec = right_crop[:, half:, :]

    return np.hstack((left_sec, right_sec))

# ---- GUI Class ----

class HyperspectralAlignerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Hyperspectral Aligner")

        self.left_hsi = self.right_hsi = None
        self.left_rgb = self.right_rgb = None
        self.left_bands = self.right_bands = None
        self.wavelengths = None
        self.mosaic_hsi  = None
        self.mosaic_rgb  = None
        self.left_path   = None

        self.canvas1 = tk.Label(master); self.canvas1.grid(row=0, column=0)
        self.canvas2 = tk.Label(master); self.canvas2.grid(row=0, column=1)

        tk.Button(master, text="Load Left",      command=self.load_left).grid(row=1, column=0)
        tk.Button(master, text="Load Right",     command=self.load_right).grid(row=1, column=1)
        tk.Button(master, text="Rotate Left",    command=lambda: self.rot_flip("left","rot")).grid(row=2, column=0)
        tk.Button(master, text="Rotate Right",   command=lambda: self.rot_flip("right","rot")).grid(row=2, column=1)
        tk.Button(master, text="Flip Left",      command=lambda: self.rot_flip("left","flip")).grid(row=3, column=0)
        tk.Button(master, text="Flip Right",     command=lambda: self.rot_flip("right","flip")).grid(row=3, column=1)
        tk.Button(master, text="Swap Images",    command=self.swap_images).grid(row=4, column=0, columnspan=2)
        tk.Button(master, text="Align & Mosaic", command=self.align_and_mosaic).grid(row=5, column=0, columnspan=2)
        tk.Button(master, text="Save Mosaic",    command=self.save_mosaic).grid(row=6, column=0, columnspan=2)

    def load_left(self):
        p = filedialog.askopenfilename(title="Select Left .hdr",
                                    filetypes=[("ENVI Header Files", "*.hdr")],
        )
        if not p: return
        self.left_path = p
        self.left_hdr = sp.envi.open(p)
        self.left_hsi = self.left_hdr.load().copy()
        self._load_waves_and_bands(self.left_hdr, side="left")
        self.update_preview("left")

    def load_right(self):
        p = filedialog.askopenfilename(title="Select Right .hdr",
                                    filetypes=[("ENVI Header Files", "*.hdr")],
        )
        if not p: return
        self.right_hdr = sp.envi.open(p)
        self.right_hsi = self.right_hdr.load().copy()
        self._load_waves_and_bands(self.right_hdr, side="right")
        self.update_preview("right")

    def _load_waves_and_bands(self, hdr, side):
        if self.wavelengths is None:
            self.wavelengths = np.array([float(w) for w in hdr.metadata['wavelength']])
        bands = [int(b) for b in hdr.metadata.get("default bands", [30,60,90])]
        rgb = linear_5_percent_normalize(hdr.load()[:, :, bands])
        setattr(self, f"{side}_rgb", rgb)
        setattr(self, f"{side}_bands", bands)

    def rot_flip(self, side, action):
        hsi = getattr(self, f"{side}_hsi")
        rgb = getattr(self, f"{side}_rgb")
        if hsi is None: return
        if action=="rot":
            hsi = np.rot90(hsi, k=2, axes=(0,1))
            rgb = np.rot90(rgb, k=2)
        else:
            hsi = np.fliplr(hsi)
            rgb = np.fliplr(rgb)
        setattr(self, f"{side}_hsi", hsi)
        setattr(self, f"{side}_rgb", rgb)
        self.update_preview(side)

    def swap_images(self):
        for attr in ("hsi","rgb","bands"):
            l = getattr(self, f"left_{attr}")
            r = getattr(self, f"right_{attr}")
            setattr(self, f"left_{attr}", r)
            setattr(self, f"right_{attr}", l)
        self.update_preview("left")
        self.update_preview("right")

    def update_preview(self, side):
        rgb = getattr(self, f"{side}_rgb")
        if rgb is None: return
        img = Image.fromarray(rgb).resize((256,256))
        tkimg = ImageTk.PhotoImage(img)
        setattr(self, f"{side}_imgtk", tkimg)
        getattr(self, f"canvas{1 if side=='left' else 2}").config(image=tkimg)

    def align_and_mosaic(self):
        if self.left_hsi is None or self.right_hsi is None:
            return messagebox.showwarning("Missing", "Load both HSI volumes first.")
        m1 = extract_marker_mask_rule_based(self.left_hsi, self.wavelengths)
        m2 = extract_marker_mask_rule_based(self.right_hsi, self.wavelengths)
        (dx, dy), score = estimate_shift_from_masks(m1,m2, visualize=True)
        try:
            self.mosaic_hsi = create_half_overlap_mosaic(self.left_hsi,
                                                         self.right_hsi,
                                                         dx, dy)
        except ValueError as e:
            return messagebox.showerror("Alignment Error", str(e))
        bands = self.left_bands
        self.mosaic_rgb = linear_5_percent_normalize(self.mosaic_hsi[:,:,bands])
        plt.figure(figsize=(10,5))
        plt.imshow(self.mosaic_rgb); plt.axis('off')
        plt.title("Mosaic RGB Preview")
        plt.show()

    def save_mosaic(self):
        if self.mosaic_hsi is None:
            return messagebox.showwarning("Nothing to save", "Please run Align & Mosaic first.")
        if not self.left_path:
            return messagebox.showerror("Error", "Left image path not set.")

        base_dir = os.path.dirname(self.left_path)
        mosaic_dir = os.path.join(base_dir, "mosaic")
        os.makedirs(mosaic_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(self.left_path))[0]
        prefix    = f"mosaic_{base_name}"

        # rebuild metadata, preserving original but updating dimensions
        metadata = dict(self.left_hdr.metadata)
        metadata['lines']   = str(self.mosaic_hsi.shape[0])
        metadata['samples'] = str(self.mosaic_hsi.shape[1])

        # save HSI (ENVI .hdr/.dat)
        hdr_path = os.path.join(mosaic_dir, prefix + ".hdr")
        sp.envi.save_image(hdr_path,
                           self.mosaic_hsi.astype(np.float32),
                           metadata=metadata,
                           force=True)

        # save RGB preview
        rgb_path = os.path.join(mosaic_dir, prefix + "_rgb.png")
        Image.fromarray(self.mosaic_rgb).save(rgb_path)

        messagebox.showinfo("Saved",
            f"Saved hyperspectral mosaic →\n  {hdr_path}\n\n"
            f"Saved RGB preview →\n  {rgb_path}"
        )
        print("Saved",
            f"Saved hyperspectral mosaic →\n  {hdr_path}\n\n"
            f"Saved RGB preview →\n  {rgb_path}"
        )

# ---- Run ----

if __name__ == "__main__":
    root = tk.Tk()
    HyperspectralAlignerGUI(root)
    root.mainloop()
