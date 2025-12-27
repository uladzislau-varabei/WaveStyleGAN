import cv2
import pandas as pd
import torch
import matplotlib.pyplot as plt

from wavelets.utils import scale_into_range, extract_coeffs_from_channels, merge_coeffs_into_spatial
from wavelets import WAVELETS_DICT_V2
from vis_utils import add_title_to_image, create_images_grid


if __name__ == '__main__':
    df = pd.read_csv('../dataset_csvs/FFHQ_v1.csv')
    idx = 0
    images_path = df.iloc[idx]['path']
    image = cv2.imread(images_path)
    input_image = torch.from_numpy(image[None, ...]).permute(0, 3, 1, 2).to(dtype=torch.float32) / 127.5 - 1
    data_format = 'NCHW'
    max_cols = 6

    target_wavelets = [
        'CDF-9/7', 'CDF-5/3', 'Haar', 'Haar_v2',
        'Daubechies-4', 'Coiflet-12',
        'Bior_spline-3/3', 'Bior_spline-3/5', 'Bior_spline-3/7', 'Bior_spline-3/9', 'Bior_spline-4/8',
        'Rev_bior_spline-3/3', 'Rev_bior_spline-3/5',  'Rev_bior_spline-3/7', 'Rev_bior_spline-3/9', 'Rev_bior_spline-4/8',
    ]
    target_wavelets = [x.lower() for x in target_wavelets]
    all_images = []
    for k, v in WAVELETS_DICT_V2.items():
        if (target_wavelets is not None) and (k.lower() in target_wavelets):
            forward_2d_op = v[0]
            kernel = v[2]
            anz_image = forward_2d_op(input_image, kernel, True, False, None, data_format)
            coeffs = extract_coeffs_from_channels(anz_image, data_format)
            coeffs_names = ['LL', 'LH', 'HL', 'HH']
            for c, n in zip(coeffs, coeffs_names):
                min_value = round(c.min().item(), 2)
                max_value = round(c.max().item(), 2)
                mean_value = round(c.mean().item(), 2)
                print(f'{k}_{n}: mean={mean_value}, min={min_value}, max={max_value}')
            print('---')
            coeffs = [scale_into_range(c, (0, 255)).clip(0, 255).to(dtype=torch.uint8) for c in coeffs]
            anz_scaled_image = merge_coeffs_into_spatial(coeffs, data_format)
            anz_scaled_image = anz_scaled_image.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
            anz_scaled_image = add_title_to_image(anz_scaled_image, k)
            all_images.append(anz_scaled_image)

    n_cols = min(max_cols, len(all_images))
    n_rows = len(all_images) // n_cols
    grid = create_images_grid(all_images, n_cols=n_cols, n_rows=n_rows)
    plt.imshow(grid[..., ::-1])
    plt.show()
