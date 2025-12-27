import os
from glob import glob

import cv2
import numpy as np
import torch

from wavelets import WAVELETS_LIST
from wavelets.utils import extract_coeffs_from_channels, merge_coeffs_into_spatial, scale_into_range, \
    DEFAULT_DATA_FORMAT, NCHW_FORMAT, NHWC_FORMAT, \
    COEFFS_SCALES_2D
from vis_utils import create_images_grid


PREVIEW_DIR = os.path.join(".", "preview_samples")


def preview_dataset():
    n_cols = 7
    n_rows = 4
    n_images = n_cols * n_rows
    target_size_xy = (256, 256)
    k_idx = 15
    images_dir = f"./FFHQ/{str(1000 * k_idx).zfill(5)}/"
    images_paths = sorted(glob(os.path.join(images_dir, "*.png")))[:n_images]
    images = [cv2.resize(cv2.imread(p), target_size_xy, interpolation=cv2.INTER_AREA) for p in images_paths]
    grid = create_images_grid(images, n_cols=n_cols, n_rows=n_rows)
    preview_image_path = os.path.join(PREVIEW_DIR, "FFHQ_grid.png")
    os.makedirs(os.path.dirname(preview_image_path), exist_ok=True)
    cv2.imwrite(preview_image_path, grid)
    print(f"Saved preview image to {preview_image_path} from dir {images_dir}")


def preview_wavelet_decomposition(wavelet_name):
    normalize_input = True
    data_format = NCHW_FORMAT
    image_path = "image.png"
    image = cv2.imread(image_path)

    wavelet = None
    for w in WAVELETS_LIST:
        if w[0].lower() == wavelet_name.lower():
            wavelet = w
            break
    assert wavelet is not None, f'wavelet_name={wavelet_name} is not supported'
    forward_2d_op = wavelet[1]
    kernel = wavelet[3]

    if data_format == NCHW_FORMAT:
        image = np.transpose(image, (2, 0, 1))

    input_image = image[None, ...].astype(np.float32)
    if normalize_input:
        input_image = (input_image / 127.5) - 1.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_image = torch.from_numpy(input_image).to(device)

    anz_image = forward_2d_op(input_image, kernel,
                              scale_1d_coeffs=True,
                              scale_2d_coeffs=True,
                              coeffs_scales_2d=COEFFS_SCALES_2D,
                              data_format=data_format)
    anz_image_coeffs = extract_coeffs_from_channels(anz_image, data_format=data_format)
    scaled_anz_image_coeffs = []
    coeffs_names = ['x_LL', 'x_LH', 'X_HL', 'X_HH']
    for idx, c in enumerate(anz_image_coeffs):
        name = coeffs_names[idx]
        vis_c = scale_into_range(c, (0, 1))
        scaled_anz_image_coeffs.append(vis_c)
    vis_anz_image = merge_coeffs_into_spatial(scaled_anz_image_coeffs, data_format=data_format)
    vis_anz_image = vis_anz_image[0].detach().cpu().numpy()
    if data_format == NCHW_FORMAT:
        vis_anz_image = np.transpose(vis_anz_image, [1, 2, 0])
    vis_anz_image = (255 * vis_anz_image).astype(np.uint8)
    image_idx = image_path.rsplit("/", 1)[1].split(".")[0]
    fixed_wavelet_name = wavelet_name.replace("/", "")
    save_path = os.path.join(PREVIEW_DIR, f"FFHQ-{image_idx}_{fixed_wavelet_name}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, vis_anz_image)
    print(f"Saved image to {save_path}")


if __name__ == "__main__":
    # preview_dataset()
    wavelet_name = "cdf-9/7"
    wavelet_name = "cdf-5/3"
    # wavelet_name = "bior_spline-3/7"
    wavelet_name = "haar"
    preview_wavelet_decomposition(wavelet_name)
