import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset import center_crop, center_crop_v2
from vis_utils import create_images_grid


def prepare_crop_img(img, target_size_yx, crop_mode):
    assert crop_mode in [1, 2, 3]
    if crop_mode == 1:
        img = center_crop(img, target_size_yx)
    elif crop_mode == 2:
        img = center_crop_v2(img, target_size_yx)
    else:
        if np.random.choice([True, False]):
            img = center_crop(img, target_size_yx)
        else:
            img = center_crop_v2(img, target_size_yx)
    return img

def show_preview_collage(df, idxs, target_size_yx, n_rows, n_cols, target_path=None):
    assert len(idxs) == n_rows * n_cols
    crop_mode = 3
    imgs = [cv2.imread(df.iloc[idx]['path']) for idx in idxs]
    imgs_grid = []
    for img_idx in range(len(imgs)):
        img = imgs[img_idx]
        size_yx = img.shape[:2]
        if target_size_yx is not None:
            if size_yx != target_size_yx:
                img = prepare_crop_img(img, target_size_yx, crop_mode)
        imgs_grid.append(img)
    imgs_grid = create_images_grid(imgs_grid, n_rows=n_rows, n_cols=n_cols)
    if target_path is not None:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        cv2.imwrite(target_path, imgs_grid)
        print(f'Saved preview imgs to {target_path}')
    plt.imshow(imgs_grid[..., ::-1])
    plt.show()


if __name__ == '__main__':
    csv_path = os.path.join('.', 'dataset_csvs', 'Landscape.csv')
    imgs_grid_path = os.path.join('.', 'preview_samples', 'Landscape_512x512.png')
    df = pd.read_csv(csv_path)
    target_size_yx = (512, 512)
    filter_column = 'thr_512'
    if filter_column is not None:
        src_size = len(df)
        df = df[df[filter_column]].reset_index(drop=True, inplace=False)
        upd_size = len(df)
        print(f'Filtered df: src_size={src_size}, upd_size={upd_size}, column={filter_column}')
    idxs = [0, 1, 2, 3, 4, 5, 6, 17, 8, 9, 10, 11, 12, 13]
    n_rows = 2
    n_cols = 7
    show_preview_collage(df, idxs, target_size_yx=target_size_yx, n_rows=n_rows, n_cols=n_cols,
                         target_path=imgs_grid_path)
