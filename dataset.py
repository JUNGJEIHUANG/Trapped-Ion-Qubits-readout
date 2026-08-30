import os

from glob import glob

import random


import cv2

import numpy as np

from PIL import Image

import torch

from torch.utils.data import Dataset

from torchvision.transforms import ToTensor


from config import IMG_DIR


SITE_DIA_LABEL_DIR = os.environ.get("SITE_DIA_LABEL_DIR", "")


MAX_IONS = 300

MASK_THRESHOLD = 127

MIN_COMPONENT_AREA = 1


def _mask_to_img(mask_file):

    mask_dir, mask_filename = os.path.split(mask_file)

    img_dir = mask_dir.replace("masks", "images")

    img_file = os.path.splitext(mask_filename)[0] + ".png"

    return os.path.join(img_dir, img_file)


def _img_to_mask(img_file):

    img_dir, img_filename = os.path.split(img_file)

    mask_dir = img_dir.replace("images", "masks")

    mask_file = os.path.splitext(img_filename)[0] + ".png"

    return os.path.join(mask_dir, mask_file)


def get_img_files(require_masks=True):

    if require_masks:

        mask_files = sorted(glob(os.path.join(IMG_DIR, "masks", "*.png")))

        img_files = [_mask_to_img(f) for f in mask_files]

    else:

        img_files = sorted(glob(os.path.join(IMG_DIR, "images", "*.png")))


    valid_img_files = []

    for img_file in img_files:

        mask_file = _img_to_mask(img_file)

        if os.path.exists(img_file) and (not require_masks or os.path.exists(mask_file)):

            valid_img_files.append(img_file)


    return np.array(valid_img_files)


def extract_centers_from_mask(mask_np, max_ions=MAX_IONS, min_area=MIN_COMPONENT_AREA):


    binary = (mask_np > MASK_THRESHOLD).astype(np.uint8)


    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(

        binary, connectivity=8

    )


    centers_list = []

    for label_id in range(1, num_labels):

        area = stats[label_id, cv2.CC_STAT_AREA]

        if area < min_area:

            continue

        cx, cy = centroids[label_id]

        centers_list.append((float(cx), float(cy)))


    centers_list = sorted(centers_list, key=lambda p: p[0])


    if len(centers_list) > max_ions:

        centers_list = centers_list[:max_ions]


    centers = np.zeros((max_ions, 2), dtype=np.float32)

    valid = np.zeros((max_ions,), dtype=np.float32)


    for i, (cx, cy) in enumerate(centers_list):

        centers[i, 0] = cx

        centers[i, 1] = cy

        valid[i] = 1.0


    return centers, valid


class MaskDataset(Dataset):

    def __init__(

        self,

        img_files,

        transform=None,

        mask_transform=None,

        max_ions=MAX_IONS,

        site_dia_label_dir=None,

        require_mask=True,

    ):

        self.img_files = list(img_files)

        self.mask_files = [_img_to_mask(f) for f in self.img_files]

        self.transform = transform if transform is not None else ToTensor()

        self.mask_transform = (

            mask_transform if mask_transform is not None else ToTensor()

        )

        self.max_ions = max_ions

        self.site_dia_label_dir = site_dia_label_dir or SITE_DIA_LABEL_DIR or None

        self.require_mask = bool(require_mask)

        self.site_coords = None

        self.site_coords_int = None

        if self.site_dia_label_dir:

            coords_path = os.path.join(self.site_dia_label_dir, "site_coords.npy")

            coords_int_path = os.path.join(self.site_dia_label_dir, "site_coords_int.npy")

            if not os.path.exists(coords_path):

                raise FileNotFoundError(f"Site-DIA coordinates not found: {coords_path}")

            self.site_coords = np.load(coords_path).astype(np.float32)

            if os.path.exists(coords_int_path):

                self.site_coords_int = np.load(coords_int_path).astype(np.int16)


    def _load_site_dia_label(self, sample_path):

        if not self.site_dia_label_dir:

            return None

        stem = os.path.splitext(os.path.basename(sample_path))[0]

        label_path = os.path.join(self.site_dia_label_dir, "per_sample_npz", stem + ".npz")

        if not os.path.exists(label_path):

            raise FileNotFoundError(f"Site-DIA per-sample label not found: {label_path}")

        data = np.load(label_path)

        state = data["state_hard"].astype(np.float32)

        valid = data["valid"].astype(np.float32)

        coords = data["coords"].astype(np.float32) if "coords" in data else self.site_coords

        return coords, state, valid, label_path


    def __len__(self):

        return len(self.img_files)


    def __getitem__(self, idx):

        img_path = self.img_files[idx]

        mask_path = self.mask_files[idx]


        if not os.path.exists(img_path):

            raise FileNotFoundError(f"Image not found: {img_path}")

        mask_exists = os.path.exists(mask_path)

        if self.require_mask and not mask_exists:

            raise FileNotFoundError(f"Mask not found: {mask_path}")


        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


        if img is None:

            raise ValueError(f"Failed to read image: {img_path}")


        if mask_exists:

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:

                raise ValueError(f"Failed to read mask: {mask_path}")

            centers, centers_valid = extract_centers_from_mask(mask, max_ions=self.max_ions)

        else:


            mask = np.zeros_like(img, dtype=np.uint8)

            centers = np.zeros((self.max_ions, 2), dtype=np.float32)

            centers_valid = np.zeros((self.max_ions,), dtype=np.float32)


        seed = random.randint(0, 2**32)


        random.seed(seed)

        img_pil = Image.fromarray(img)

        img_tensor = self.transform(img_pil)


        random.seed(seed)

        mask_pil = Image.fromarray(mask)

        mask_tensor = self.mask_transform(mask_pil)


        mask_tensor = (mask_tensor > 0.5).float()


        sample = {

            "image": img_tensor,

            "mask": mask_tensor,

            "centers_gt": torch.from_numpy(centers),

            "centers_valid": torch.from_numpy(centers_valid),

            "img_path": img_path,

            "mask_path": mask_path,

        }


        site_label = self._load_site_dia_label(img_path)

        if site_label is not None:

            site_coords, state_hard, valid, label_path = site_label

            sample.update(

                {

                    "site_coords": torch.from_numpy(site_coords),

                    "state_hard": torch.from_numpy(state_hard),

                    "valid": torch.from_numpy(valid),

                    "site_label_path": label_path,

                }

            )

        return sample


if __name__ == "__main__":

    pass

