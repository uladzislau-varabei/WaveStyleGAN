import cv2
import numpy as np
import pandas as pd
import torch

from wavelets import WAVELETS_DICT_V2
from wavelets.utils import extract_coeffs_v2, extract_coeffs_from_channels, COEFFS_SCALES_2D
from wavelets.dwt2d import DWTForward


def eval_stats(x, thr=0.02):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    mean = x.mean()
    std = x.std()
    min1 = mean - 3 * std
    max1 = mean + 3 * std
    med = np.median(x)
    min0 = x.min()
    max0 = x.max()
    q = 0.1
    q1 = np.quantile(x.flatten(), q)
    q2 = np.quantile(x.flatten(), 1 - q)
    zero_ratio = np.where(abs(x) < thr, 1, 0).sum() / np.prod(x.shape)
    return {
        'mean': mean, 'std': std, 'med': med,
        'min': round(min0, 6), 'max': round(max0, 6),
        'min_std': round(min1, 6), 'max_std': round(max1, 6),
        f'min_q{q}': round(q1, 6), f'max_q{q}': round(q2, 6),
        'zero_ratio': round(zero_ratio, 3)
    }


if __name__ == '__main__':
    df = pd.read_csv('../dataset_csvs/FFHQ_v1.csv')
    batch_size = 32
    images_paths = df['path'][:batch_size]
    images = np.array([cv2.imread(p) for p in images_paths], dtype=np.float32)
    images = np.transpose(images / 127.5 - 1, (0, 3, 1, 2))  # [NHWC => NCHW]
    print(f'Input images.shape: {images.shape}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images = torch.from_numpy(images).to(device)

    modes = ['zero', 'symmetric', 'periodization', 'constant', 'reflect', 'replicate', 'periodic']

    wavelet = DWTForward(J=1, wave='bior4.4', mode='base').to(device)
    res1 = wavelet(images)
    res1_LL, res1_LH, res1_HL, res1_HH = extract_coeffs_v2(res1)
    print(f'LL={tuple(res1_LL.shape)}, LH={tuple(res1_LH.shape)}, HL={tuple(res1_HL.shape)}, HH={tuple(res1_HH.shape)}')
    # print(f'res1 LL: {eval_stats(res1_LL)}')
    # print(f'res1 LH: {eval_stats(res1_LH)}')
    # print(f'res1 HL: {eval_stats(res1_HL)}')
    # print(f'res1 HH: {eval_stats(res1_HH)}')

    wave = 'cdf-9/7'
    wave_data = WAVELETS_DICT_V2[wave]
    forward_2d_op = wave_data[0]
    kernel = wave_data[2]
    res2 = forward_2d_op(images, kernel, True, True, COEFFS_SCALES_2D, 'NCHW')
    res2_LL, res2_LH, res2_HL, res2_HH = extract_coeffs_from_channels(res2)
    print(f'images: {eval_stats(images)}')
    print(f'res2 LL: {eval_stats(res2_LL)}')
    print(f'res2 LH: {eval_stats(res2_LH)}')
    print(f'res2 HL: {eval_stats(res2_HL)}')
    print(f'res2 HH: {eval_stats(res2_HH)}')

    import matplotlib.pyplot as plt
    bins = 100
    plt.figure(1)
    plt.hist(res2_LH.detach().cpu().numpy().flatten(), bins=bins, density=True)
    plt.title('LH')
    plt.figure(2)
    plt.hist(res2_HL.detach().cpu().numpy().flatten(), bins=bins, density=True)
    plt.title('HL')
    plt.figure(3)
    plt.hist(res2_HH.detach().cpu().numpy().flatten(), bins=bins, density=True)
    plt.title('HH')
    plt.show()
