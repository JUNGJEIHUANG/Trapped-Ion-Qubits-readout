from __future__ import annotations

import argparse
from pathlib import Path
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".npy"}


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    array = np.squeeze(array)

    if array.ndim == 3 and array.shape[-1] in (3, 4):
        arr = array.astype(np.float32)
        if arr.dtype != np.uint8:
            lo = float(np.nanmin(arr))
            hi = float(np.nanmax(arr))
            if hi > lo:
                arr = (arr - lo) / (hi - lo) * 255.0
            else:
                arr = np.zeros_like(arr)
        return np.clip(arr, 0, 255).astype(np.uint8)

    arr = array.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr = np.zeros_like(arr)
    return np.clip(arr, 0, 255).astype(np.uint8)


def load_frames(path: Path) -> list[Image.Image]:
    if path.suffix.lower() == ".npy":
        data = np.load(path, allow_pickle=False)
        data = np.asarray(data)

        if data.ndim == 2:
            return [Image.fromarray(normalize_to_uint8(data)).convert("RGB")]

        if data.ndim == 3:
            if data.shape[-1] in (3, 4):
                return [Image.fromarray(normalize_to_uint8(data)).convert("RGB")]
            return [Image.fromarray(normalize_to_uint8(data[i])).convert("RGB") for i in range(data.shape[0])]

        if data.ndim == 4 and data.shape[-1] in (3, 4):
            return [Image.fromarray(normalize_to_uint8(data[i])).convert("RGB") for i in range(data.shape[0])]

        raise ValueError(f"Unsupported npy shape: {data.shape}")

    return [Image.open(path).convert("RGB")]


class ImageSlideshow:
    def __init__(self, root: tk.Tk, files: list[Path], interval_ms: int, fullscreen: bool):
        self.root = root
        self.files = files
        self.interval_ms = interval_ms
        self.file_index = 0
        self.frame_index = 0
        self.frames: list[Image.Image] = []
        self.paused = False
        self.photo = None
        self.after_id = None

        self.root.title("Image / NPY Slideshow")
        self.root.configure(bg="black")
        self.root.attributes("-fullscreen", fullscreen)

        self.label = tk.Label(root, bg="black")
        self.label.pack(fill=tk.BOTH, expand=True)

        self.status = tk.Label(root, bg="black", fg="white", anchor="w", font=("Arial", 12))
        self.status.pack(fill=tk.X)

        self.root.bind("<Escape>", lambda event: self.root.destroy())
        self.root.bind("<space>", lambda event: self.toggle_pause())
        self.root.bind("<Right>", lambda event: self.next_frame())
        self.root.bind("<Left>", lambda event: self.prev_frame())
        self.root.bind("<Down>", lambda event: self.next_file())
        self.root.bind("<Up>", lambda event: self.prev_file())
        self.root.bind("f", lambda event: self.toggle_fullscreen())
        self.root.bind("F", lambda event: self.toggle_fullscreen())
        self.root.bind("q", lambda event: self.root.destroy())
        self.root.bind("Q", lambda event: self.root.destroy())
        self.root.bind("<Configure>", lambda event: self.show_current())

        self.load_current_file()
        self.show_current()
        self.schedule_next()

    def load_current_file(self):
        self.frames = load_frames(self.files[self.file_index])
        self.frame_index = min(self.frame_index, len(self.frames) - 1)

    def toggle_fullscreen(self):
        self.root.attributes("-fullscreen", not bool(self.root.attributes("-fullscreen")))

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused and self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if not self.paused:
            self.schedule_next()
        self.update_status()

    def schedule_next(self):
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
        if not self.paused:
            self.after_id = self.root.after(self.interval_ms, self.next_frame)

    def next_file(self):
        self.file_index = (self.file_index + 1) % len(self.files)
        self.frame_index = 0
        self.load_current_file()
        self.show_current()
        self.schedule_next()

    def prev_file(self):
        self.file_index = (self.file_index - 1) % len(self.files)
        self.frame_index = 0
        self.load_current_file()
        self.show_current()
        self.schedule_next()

    def next_frame(self):
        self.frame_index += 1
        if self.frame_index >= len(self.frames):
            self.file_index = (self.file_index + 1) % len(self.files)
            self.frame_index = 0
            self.load_current_file()
        self.show_current()
        self.schedule_next()

    def prev_frame(self):
        self.frame_index -= 1
        if self.frame_index < 0:
            self.file_index = (self.file_index - 1) % len(self.files)
            self.load_current_file()
            self.frame_index = len(self.frames) - 1
        self.show_current()
        self.schedule_next()

    def show_current(self):
        if not self.frames:
            return
        image = self.frames[self.frame_index].copy()
        width = max(self.label.winfo_width(), 1)
        height = max(self.label.winfo_height(), 1)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), "black")
        x = (width - image.width) // 2
        y = (height - image.height) // 2
        canvas.paste(image, (x, y))
        self.photo = ImageTk.PhotoImage(canvas)
        self.label.configure(image=self.photo)
        self.update_status()

    def update_status(self):
        state = "Paused" if self.paused else "Playing"
        path = self.files[self.file_index]
        self.status.configure(
            text=f"{state} | file {self.file_index + 1}/{len(self.files)} | frame {self.frame_index + 1}/{len(self.frames)} | {path} | Space pause | ←/→ frame | ↑/↓ file | F fullscreen | Esc/Q quit"
        )


def collect_files(folder: Path, recursive: bool) -> list[Path]:
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def main():
    parser = argparse.ArgumentParser(description="Automatically display image and npy files in a folder.")
    parser.add_argument("--folder", type=Path, default=Path("."))
    parser.add_argument("--file", type=Path, default=None, help="Display one specific file, such as a .npy file.")
    parser.add_argument("--interval", type=float, default=0.2, help="Seconds between frames/images.")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--windowed", action="store_true")
    args = parser.parse_args()

    if args.file is not None:
        files = [args.file.expanduser().resolve()]
    else:
        folder = args.folder.expanduser().resolve()
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")
        files = collect_files(folder, recursive=not args.no_recursive)

    files = [path for path in files if path.exists() and path.is_file()]
    if not files:
        raise RuntimeError("No supported image or npy files found.")

    root = tk.Tk()
    ImageSlideshow(root, files, interval_ms=max(20, int(args.interval * 1000)), fullscreen=not args.windowed)
    root.mainloop()


if __name__ == "__main__":
    main()
