import sys, os, argparse, json, glob, logging
import numpy as np
from PIL import Image
import cv2

def crop_image(img, default_size=None):
    old_im = img.convert("L")
    img_data = np.asarray(old_im, dtype=np.uint8)  # height, width
    nnz_inds = np.where(img_data != 255)
    if len(nnz_inds[0]) == 0:
        if not default_size:
            return old_im
        else:
            assert len(default_size) == 2, default_size
            x_min, y_min, x_max, y_max = 0, 0, default_size[0], default_size[1]
            old_im = old_im.crop((x_min, y_min, x_max + 1, y_max + 1))
            return old_im
    y_min = np.min(nnz_inds[0])
    y_max = np.max(nnz_inds[0])
    x_min = np.min(nnz_inds[1])
    x_max = np.max(nnz_inds[1])
    old_im = old_im.crop((x_min, y_min, x_max + 1, y_max + 1))
    return old_im


def pad_group_image(img, pad_size, buckets):
    PAD_TOP, PAD_LEFT, PAD_BOTTOM, PAD_RIGHT = pad_size
    old_im = img
    old_size = (
        old_im.size[0] + PAD_LEFT + PAD_RIGHT,
        old_im.size[1] + PAD_TOP + PAD_BOTTOM,
    )
    j = -1
    for i in range(len(buckets)):
        if old_size[0] <= buckets[i][0] and old_size[1] <= buckets[i][1]:
            j = i
            break
    if j < 0:
        new_size = old_size
        new_im = Image.new("RGB", new_size, (255, 255, 255))
        new_im.paste(old_im, (PAD_LEFT, PAD_TOP))
        return new_im
    new_size = buckets[j]
    new_im = Image.new("RGB", new_size, (255, 255, 255))
    new_im.paste(old_im, (PAD_LEFT, PAD_TOP))
    return new_im


def downsample_image(img, ratio):
    assert ratio > 0
    old_im = img
    width, height = old_im.size
    if ratio == 1:
        return old_im
    old_size = old_im.size
    new_size = (int(old_size[0] / ratio), int(old_size[1] / ratio))
    new_im = old_im.resize(new_size, Image.LANCZOS)
    return new_im

def preprocess_image(
    img,
    crop_blank_default_size=[600, 60],
    pad_size=[8, 8, 8, 8],
    buckets=[
        [240, 100],
        [320, 80],
        [400, 80],
        [400, 100],
        [480, 80],
        [480, 100],
        [560, 80],
        [560, 100],
        [640, 80],
        [640, 100],
        [720, 80],
        [720, 100],
        [720, 120],
        [720, 200],
        [800, 100],
        [800, 320],
        [1000, 200],
        [1000, 400],
        [1200, 200],
        [1600, 200],
        [1600, 1600],
    ],
    downsample_ratio=4,
):
    if isinstance(img, str):
        img = Image.fromarray(cv2.imread(img, 0))
    else:
        img = Image.fromarray(img)

    img = crop_image(img, crop_blank_default_size)
    img = pad_group_image(img, pad_size, buckets)
    img = downsample_image(img, downsample_ratio)
    
    return img


if __name__ == "__main__":

    img = preprocess_image(sys.argv[1])
    img.save(sys.argv[1])
