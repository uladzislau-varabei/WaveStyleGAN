from wavelets.utils import NCHW_FORMAT, NHWC_FORMAT, DEFAULT_DATA_FORMAT, \
    extract_coeffs_from_channels, extract_coeffs_from_spatial, \
    merge_coeffs_into_channels, merge_coeffs_into_spatial, \
    test_lifting_scheme, test_grad
from vis_utils import prepare_input_image, show_lifting_results


HAAR_KERNEL_V2 = [0.]  # for compatibility with other wavelets

# Note: implementation is based on https://github.com/rinongal/swagan/blob/main/training/wavelets.py
# Thanks to original article: https://arxiv.org/abs/1907.03128
# TODO: fix error with inverse op

# ----- New vectorized versions -----

def fast_haar_2d_op_v2(x, kernel, scale_1d_coeffs, scale_2d_coeffs, coeffs_scales_2d, data_format=DEFAULT_DATA_FORMAT):
    # Unused args for compatibility with other wavelets
    _ = kernel
    _ = scale_1d_coeffs
    _ = scale_2d_coeffs
    _ = coeffs_scales_2d
    # 1. Extract patches
    if data_format == NCHW_FORMAT:
        x1 = x[:, :, 0::2, 0::2] #x(2i−1, 2j−1)
        x2 = x[:, :, 1::2, 0::2] #x(2i, 2j-1)
        x3 = x[:, :, 0::2, 1::2] #x(2i−1, 2j)
        x4 = x[:, :, 1::2, 1::2] #x(2i, 2j)
    else: # if data_format == NHWC_FORMAT:
        x1 = x[:, 0::2, 0::2, :] #x(2i−1, 2j−1)
        x2 = x[:, 1::2, 0::2, :] #x(2i, 2j-1)
        x3 = x[:, 0::2, 1::2, :] #x(2i−1, 2j)
        x4 = x[:, 1::2, 1::2, :] #x(2i, 2j)
    # 2. Apply op
    x_LL = x1 + x2 + x3 + x4
    x_LH = -x1 - x3 + x2 + x4
    x_HL = -x1 + x3 - x2 + x4
    x_HH = x1 - x3 - x2 + x4
    # 3. Merge coeffs
    x_output = merge_coeffs_into_channels([x_LL, x_LH, x_HL, x_HH], data_format=data_format)
    return x_output


def fast_inv_haar_2d_op_v2(x, kernel, scale_1d_coeffs, scale_2d_coeffs, coeffs_scales_2d, data_format=DEFAULT_DATA_FORMAT):
    # Unused args for compatibility with other wavelets
    _ = kernel
    _ = scale_1d_coeffs
    _ = scale_2d_coeffs
    _ = coeffs_scales_2d
    # x_LL, x_LH, x_HL, x_HH = x_coeffs
    # 1. Extract coeffs
    x_LL, x_LH, x_HL, x_HH = extract_coeffs_from_channels(x, data_format=data_format)
    # 2. Apply op
    x1 = (x_LL - x_LH - x_HL + x_HH) / 4
    x2 = (x_LL - x_LH + x_HL - x_HH) / 4
    x3 = (x_LL + x_LH - x_HL - x_HH) / 4
    x4 = (x_LL + x_LH + x_HL + x_HH) / 4
    # 3. Convert to spatial
    x = merge_coeffs_into_spatial([x1, x2, x3, x4], data_format=data_format)
    return x


# ----- Main -----

if __name__ == '__main__':
    image, _ = prepare_input_image()
    data_format = NHWC_FORMAT
    #data_format = NCHW_FORMAT
    vis_anz_image, error, _ = test_lifting_scheme(image,
                                                  kernel=HAAR_KERNEL_V2,
                                                  forward_2d_op=fast_haar_2d_op_v2,
                                                  backward_2d_op=fast_inv_haar_2d_op_v2,
                                                  data_format=data_format)

    show_lifting_results(src_image=image, anz_image=vis_anz_image, wavelet_name='Haar_v2')

    grad_vis_image, grad_diff = test_grad(image,
                                          kernel=HAAR_KERNEL_V2,
                                          forward_2d_op=fast_haar_2d_op_v2,
                                          backward_2d_op=fast_inv_haar_2d_op_v2,
                                          data_format=data_format)
    import matplotlib.pyplot as plt
    plt.imshow(grad_vis_image)
    plt.title(f"Grad mean diff: {grad_diff}")
    plt.show()
