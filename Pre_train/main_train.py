import argparse

import os

import glob

import random


import cv2

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader


import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from dataset import IonUnlabeledDataset

from model_denoise import DWNetV2DenoiseUNet

from ssim_utils import ssim_loss


DATA_DIR = os.environ.get("PRETRAIN_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pretrain"))

SAVE_DIR = os.environ.get(

    "PRETRAIN_SAVE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "Run_pretrain")

)


EPOCHS = 100

BATCH_SIZE = 16

LR = 1e-3

NUM_WORKERS = 0

RESIZE_W = 456

RESIZE_H = 88


L1_WEIGHT = 0.8

SSIM_WEIGHT = 0.2


BLUR_PROB = 0.3

MASK_PROB = 0.3

MAX_MASK_FRACTION = 0.12


DEVICE = "cuda"

print(f"Using device: {DEVICE}")


def add_noise(x):

    std = torch.empty(1, device=x.device).uniform_(0.01, 0.08).item()

    noisy = x + torch.randn_like(x) * std


    scale = torch.empty(1, device=x.device).uniform_(20.0, 60.0).item()

    noisy = torch.poisson(torch.clamp(noisy, 0.0, 1.0) * scale) / scale


    offset = torch.empty((x.size(0), 1, 1, 1), device=x.device).uniform_(0.0, 0.06)

    noisy = noisy + offset


    noisy_np = noisy.detach().cpu().numpy()

    blurred = []

    for idx in range(noisy_np.shape[0]):

        img = noisy_np[idx, 0]

        if random.random() < BLUR_PROB:

            img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.8)

        blurred.append(img[None, ...])

    noisy = torch.from_numpy(np.stack(blurred, axis=0)).to(

        device=x.device, dtype=x.dtype

    )


    batch_size, _, height, width = noisy.shape

    for idx in range(batch_size):

        if random.random() < MASK_PROB:

            mask_h = random.randint(

                max(1, int(height * 0.03)), max(1, int(height * MAX_MASK_FRACTION))

            )

            mask_w = random.randint(

                max(1, int(width * 0.03)), max(1, int(width * MAX_MASK_FRACTION))

            )

            top = random.randint(0, max(0, height - mask_h))

            left = random.randint(0, max(0, width - mask_w))

            noisy[idx, :, top : top + mask_h, left : left + mask_w] = 0.0


    noisy = torch.clamp(noisy, 0.0, 1.0)

    return noisy


def main(args):

    data_dir = args.data_dir

    save_dir = args.save_dir

    resize_w = args.resize_w

    resize_h = args.resize_h


    os.makedirs(save_dir, exist_ok=True)


    image_paths = sorted(glob.glob(os.path.join(data_dir, "*.*")))

    assert len(image_paths) > 0, f"No images found in {data_dir}"


    print(f"Found {len(image_paths)} images")

    print(f"Resize target: {resize_w}x{resize_h}")


    dataset = IonUnlabeledDataset(

        image_paths,

        resize=(resize_w, resize_h),

    )


    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=True,

        drop_last=True,

    )


    model = DWNetV2DenoiseUNet(in_channels=1, out_channels=1).to(DEVICE)


    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


    criterion_l1 = nn.L1Loss()


    best_loss = float("inf")


    for epoch in range(EPOCHS):

        model.train()

        epoch_loss = 0.0


        for batch in loader:

            clean = batch["image"].to(DEVICE)

            noisy = add_noise(clean)

            pred = model(noisy)

            loss_l1 = criterion_l1(pred, clean)

            loss_ssim = ssim_loss(pred, clean)

            loss = L1_WEIGHT * loss_l1 + SSIM_WEIGHT * loss_ssim


            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            epoch_loss += loss.item()


        scheduler.step()


        avg_loss = epoch_loss / len(loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.6f}")


        torch.save(

            {

                "epoch": epoch + 1,

                "model_state_dict": model.state_dict(),

            },

            os.path.join(save_dir, "last.pth"),

        )


        if avg_loss < best_loss:

            best_loss = avg_loss

            torch.save(

                {

                    "epoch": epoch + 1,

                    "model_state_dict": model.state_dict(),

                },

                os.path.join(save_dir, "best.pth"),

            )

            print("Saved best.pth")


    print("Pretraining finished")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(

        description="Self-supervised pretraining for DWNetV2 denoising autoencoder"

    )

    parser.add_argument(

        "--data_dir",

        type=str,

        default=DATA_DIR,

        help="Directory containing unlabeled pretraining images",

    )

    parser.add_argument(

        "--save_dir",

        type=str,

        default=SAVE_DIR,

        help="Directory to save pretraining checkpoints",

    )

    parser.add_argument("--resize_w", type=int, default=RESIZE_W, help="Resize width")

    parser.add_argument("--resize_h", type=int, default=RESIZE_H, help="Resize height")

    args = parser.parse_args()

    main(args)

