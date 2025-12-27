from wavelets.bior_spline_33 import (
    fast_biorspline33_2d_op, fast_inv_biorspline33_2d_op, BIOR_SPLINE_33_KERNEL
)
from wavelets.bior_spline_35 import (
    fast_biorspline35_2d_op, fast_inv_biorspline35_2d_op, BIOR_SPLINE_35_KERNEL
)
from wavelets.bior_spline_37 import (
    fast_biorspline37_2d_op, fast_inv_biorspline37_2d_op, BIOR_SPLINE_37_KERNEL
)
from wavelets.bior_spline_39 import (
    fast_biorspline39_2d_op, fast_inv_biorspline39_2d_op, BIOR_SPLINE_39_KERNEL
)
from wavelets.bior_spline_48 import (
    fast_biorspline48_2d_op, fast_inv_biorspline48_2d_op, BIOR_SPLINE_48_KERNEL
)
from wavelets.rev_bior_spline_33 import (
    fast_revbiorspline33_2d_op, fast_inv_revbiorspline33_2d_op, REV_BIOR_SPLINE_33_KERNEL
)
from wavelets.rev_bior_spline_35 import (
    fast_revbiorspline35_2d_op, fast_inv_revbiorspline35_2d_op, REV_BIOR_SPLINE_35_KERNEL
)
from wavelets.rev_bior_spline_37 import (
    fast_revbiorspline37_2d_op, fast_inv_revbiorspline37_2d_op, REV_BIOR_SPLINE_37_KERNEL
)
from wavelets.rev_bior_spline_39 import (
    fast_revbiorspline39_2d_op, fast_inv_revbiorspline39_2d_op, REV_BIOR_SPLINE_39_KERNEL
)
from wavelets.rev_bior_spline_48 import (
    fast_revbiorspline48_2d_op, fast_inv_revbiorspline48_2d_op, REV_BIOR_SPLINE_48_KERNEL
)
from wavelets.cdf_53 import fast_cdf53_2d_op, fast_inv_cdf53_2d_op, CDF_53_KERNEL
from wavelets.cdf_97 import fast_cdf97_2d_op, fast_inv_cdf97_2d_op, CDF_97_KERNEL
from wavelets.haar import fast_haar_2d_op, fast_inv_haar_2d_op, HAAR_KERNEL
from wavelets.haar_v2 import fast_haar_2d_op_v2, fast_inv_haar_2d_op_v2, HAAR_KERNEL_V2
from wavelets.daub_4 import fast_daub4_2d_op, fast_inv_daub4_2d_op, DAUB4_KERNEL
from wavelets.coif_12 import fast_coif12_2d_op, fast_inv_coif12_2d_op, COIF12_KERNEL


WAVELETS_LIST = [
    ['CDF-9/7', fast_cdf97_2d_op, fast_inv_cdf97_2d_op, CDF_97_KERNEL],
    ['CDF-5/3', fast_cdf53_2d_op, fast_inv_cdf53_2d_op, CDF_53_KERNEL],
    ['Haar', fast_haar_2d_op, fast_inv_haar_2d_op, HAAR_KERNEL],
    ['Haar_v2', fast_haar_2d_op_v2, fast_inv_haar_2d_op_v2, HAAR_KERNEL_V2],
    ['Daubechies-4', fast_daub4_2d_op, fast_inv_daub4_2d_op, DAUB4_KERNEL],
    ['Coiflet-12', fast_coif12_2d_op, fast_inv_coif12_2d_op, COIF12_KERNEL],
    ['Bior_spline-3/3', fast_biorspline33_2d_op, fast_inv_biorspline33_2d_op, BIOR_SPLINE_33_KERNEL],
    ['Bior_spline-3/5', fast_biorspline35_2d_op, fast_inv_biorspline35_2d_op, BIOR_SPLINE_35_KERNEL],
    ['Bior_spline-3/7', fast_biorspline37_2d_op, fast_inv_biorspline37_2d_op, BIOR_SPLINE_37_KERNEL],
    ['Bior_spline-3/9', fast_biorspline39_2d_op, fast_inv_biorspline39_2d_op, BIOR_SPLINE_39_KERNEL],
    ['Bior_spline-4/8', fast_biorspline48_2d_op, fast_inv_biorspline48_2d_op, BIOR_SPLINE_48_KERNEL],
    ['Rev_bior_spline-3/3', fast_revbiorspline33_2d_op, fast_inv_revbiorspline33_2d_op, REV_BIOR_SPLINE_33_KERNEL],
    ['Rev_bior_spline-3/5', fast_revbiorspline35_2d_op, fast_inv_revbiorspline35_2d_op, REV_BIOR_SPLINE_35_KERNEL],
    ['Rev_bior_spline-3/7', fast_revbiorspline37_2d_op, fast_inv_revbiorspline37_2d_op, REV_BIOR_SPLINE_37_KERNEL],
    ['Rev_bior_spline-3/9', fast_revbiorspline39_2d_op, fast_inv_revbiorspline39_2d_op, REV_BIOR_SPLINE_39_KERNEL],
    ['Rev_bior_spline-4/8', fast_revbiorspline48_2d_op, fast_inv_revbiorspline48_2d_op, REV_BIOR_SPLINE_48_KERNEL],
]

FW_KEY = "forward_2d_op"
BW_KEY = "backward_2d_op"
KERNEL_KEY = "kernel"

WAVELETS_DICT = {
    w[0].lower(): {
        FW_KEY: w[1],
        BW_KEY: w[2],
        KERNEL_KEY: w[3]
    } for w in WAVELETS_LIST
}

WAVELETS_DICT_V2 = {w[0].lower(): [w[1], w[2], w[3]] for w in WAVELETS_LIST}


if __name__ == '__main__':
    import cv2
    import pandas as pd
    from tqdm import tqdm
    from wavelets.utils import find_scales

    df = pd.read_csv('../dataset_csvs/FFHQ_v1.csv')
    total_images = 1000
    images_paths = df['path'][:total_images]
    batch_size = 100
    data_format = 'NCHW'

    target_wavelets = [
        #'CDF-9/7', 'CDF-5/3', 'Haar', #'Haar_v2',
        'Haar',
        # 'Daubechies-4', 'Coiflet-12',
        # 'Bior_spline-3/3', 'Bior_spline-3/5', 'Bior_spline-3/7', 'Bior_spline-3/9', 'Bior_spline-4/8',
        # 'Rev_bior_spline-3/3', 'Rev_bior_spline-3/5',  'Rev_bior_spline-3/7', 'Rev_bior_spline-3/9', 'Rev_bior_spline-4/8',
    ]
    images = [cv2.imread(p) for p in tqdm(images_paths)]
    target_wavelets = [x.lower() for x in target_wavelets]
    for k, v in WAVELETS_DICT_V2.items():
        if (target_wavelets is not None) and (k.lower() in target_wavelets):
            forward_2d_op = v[0]
            kernel = v[2]
            scales = find_scales(images, name=k, kernel=kernel, forward_2d_op=forward_2d_op,
                                 batch_size=batch_size, data_format=data_format)

    """
    1. Haar: 1e-5, 10k: 100%, 87%, 87%, 77%
    mean=0.477, std=0.010, med=0.476
    x_LL: mean=0.951, std=0.020, med=0.951, scale_mean=0.501, scale_med=0.501
    x_LH: mean=0.027, std=0.001, med=0.027, scale_mean=17.665, scale_med=17.664
    x_HL: mean=0.029, std=0.001, med=0.029, scale_mean=16.704, scale_med=16.661
    x_HH: mean=0.013, std=0.000, med=0.013, scale_mean=36.951, scale_med=37.014
    """
