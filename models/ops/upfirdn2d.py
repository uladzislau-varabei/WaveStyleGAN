
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Custom PyTorch ops for efficient resampling of 2D images."""

import os
import platform

import numpy as np
import torch

from models import misc
from models.ops import conv2d_gradfix, custom_ops
from shared_utils import ALLOW_CUDA_KERNEL_USAGE, is_running_on_linux

#----------------------------------------------------------------------------

_plugin = None

# Help:
# 1. pip install ninja
# 2. conda install cuda-nvcc -c nvidia (thanks to https://github.com/conda/conda/issues/7757 by iyume)
# 3. export CFLAGS="-I/home/user/miniconda3/envs/torch-2.6/lib/python3.10/site-packages/nvidia/cuda_runtime/include $CFLAGS"

"""
Help:
export CUDA_HOME="/home/user/miniconda3/envs/torch-2.6" (which nvcc)
 locate cuda_runtime_api.h
 
 /home/user/miniconda3/envs/torch-2.6/lib/python3.10/site-packages/torch/include/ATen/cuda/CUDAContextLight.h:6:10:


/home/user/miniconda3/envs/torch-2.6/lib/python3.10/site-packages/nvidia/cuda_cupti/lib

export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CFLAGS="-I$CUDA_HOME/include $CFLAGS"
 
"""

def _init():
    global _plugin
    # return False # disable for now
    if _plugin is None:
        _plugin = custom_ops.get_plugin(
            module_name='upfirdn2d_plugin',
            sources=['upfirdn2d.cpp', 'upfirdn2d.cu'],
            headers=['upfirdn2d.h'],
            source_dir=os.path.dirname(__file__),
            extra_cuda_cflags=['--use_fast_math', '--allow-unsupported-compiler'],
        )
    return True

def _parse_scaling(scaling):
    if isinstance(scaling, int):
        scaling = [scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, sy = scaling
    assert sx >= 1 and sy >= 1
    return sx, sy

def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 2:
        padx, pady = padding
        padding = [padx, padx, pady, pady]
    padx0, padx1, pady0, pady1 = padding
    return padx0, padx1, pady0, pady1

def _get_filter_size(f):
    if f is None:
        return 1, 1
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2], f'f.ndim={f.ndim}, f.shape={f.shape}'
    fw = f.shape[-1]
    fh = f.shape[0]
    with misc.suppress_tracer_warnings():
        fw = int(fw)
        fh = int(fh)
    misc.assert_shape(f, [fh, fw][:f.ndim])
    assert fw >= 1 and fh >= 1
    return fw, fh

#----------------------------------------------------------------------------

def setup_filter(f, device=torch.device('cpu'), normalize=True, flip_filter=False, gain=1, separable=None):
    r"""Convenience function to setup 2D FIR filter for `upfirdn2d()`.

    Args:
        f:           Torch tensor, numpy array, or python list of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable),
                     `[]` (impulse), or
                     `None` (identity).
        device:      Result device (default: cpu).
        normalize:   Normalize the filter so that it retains the magnitude
                     for constant input signal (DC)? (default: True).
        flip_filter: Flip the filter? (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        separable:   Return a separable filter? (default: select automatically).

    Returns:
        Float32 tensor of the shape
        `[filter_height, filter_width]` (non-separable) or
        `[filter_taps]` (separable).
    """
    # Validate.
    if f is None:
        f = 1
    f = torch.as_tensor(f, dtype=torch.float32)
    assert f.ndim in [0, 1, 2]
    assert f.numel() > 0
    if f.ndim == 0:
        f = f[np.newaxis]

    # Separable?
    if separable is None:
        separable = (f.ndim == 1 and f.numel() >= 8)
    if f.ndim == 1 and not separable:
        f = f.ger(f)
    assert f.ndim == (1 if separable else 2)

    # Apply normalize, flip, gain, and device.
    if normalize:
        f /= f.sum()
    if flip_filter:
        f = f.flip(list(range(f.ndim)))
    f = f * (gain ** (f.ndim / 2))
    f = f.to(device=device)
    return f

#----------------------------------------------------------------------------

def upfirdn2d(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1, impl='cuda',
    use_custom_conv2d_op=True):
    r"""Pad, upsample, filter, and downsample a batch of 2D images.

    Performs the following sequence of operations for each channel:

    1. Upsample the image by inserting N-1 zeros after each pixel (`up`).

    2. Pad the image with the specified number of zeros on each side (`padding`).
       Negative padding corresponds to cropping the image.

    3. Convolve the image with the specified 2D FIR filter (`f`), shrinking it
       so that the footprint of all output pixels lies within the input image.

    4. Downsample the image by keeping every Nth pixel (`down`).

    This sequence of operations bears close resemblance to scipy.signal.upfirdn().
    The fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports gradients of arbitrary order.

    Args:
        x:                     Float32/float64/float16 input tensor of the shape
                               `[batch_size, num_channels, in_height, in_width]`.
        f:                     Float32 FIR filter of the shape
                              `[filter_height, filter_width]` (non-separable),
                              `[filter_taps]` (separable), or
                              `None` (identity).
        up:                   Integer upsampling factor. Can be a single int or a list/tuple
                              `[x, y]` (default: 1).
        down:                 Integer downsampling factor. Can be a single int or a list/tuple
                              `[x, y]` (default: 1).
        padding:              Padding with respect to the upsampled image. Can be a single number
                              or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                              (default: 0).
        flip_filter:          False = convolution, True = correlation (default: False).
        gain:                 Overall scaling factor for signal magnitude (default: 1).
        impl:                 Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).
        use_custom_conv2d_op: Enable conv2d_gradfix (default: True)?

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    assert isinstance(x, torch.Tensor)
    assert impl in ['ref', 'cuda', 'custom_grad']
    if impl == 'cuda' and x.device.type == 'cuda' and _init():
        # print("Using CUDA upfirdn2d")
        return _upfirdn2d_cuda(up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain).apply(x, f)
    elif impl == 'custom_grad':
        # print(f"Using custom grad upfirdn2d")
        return _upfirdn2d_custom_grad(up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain,
            use_custom_conv2d_op=use_custom_conv2d_op).apply(x, f)
    # print(f"Using ref upfirdn2d")
    return _upfirdn2d_ref(x, f, up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain,
                use_custom_conv2d_op=use_custom_conv2d_op)

#----------------------------------------------------------------------------

@misc.profiled_function
def _upfirdn2d_ref(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1, use_custom_conv2d_op=True):
    """Slow reference implementation of `upfirdn2d()` using standard PyTorch ops.
    """
    # Validate arguments.
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    if f is None:
        f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    assert f.dtype == torch.float32 and not f.requires_grad
    batch_size, num_channels, in_height, in_width = x.shape
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)

    # Check that upsampled buffer is not smaller than the filter.
    upW = in_width * upx + padx0 + padx1
    upH = in_height * upy + pady0 + pady1
    assert upW >= f.shape[-1] and upH >= f.shape[0]

    # Upsample by inserting zeros.
    x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
    x = torch.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])

    # Pad or crop.
    x = torch.nn.functional.pad(x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)])
    x = x[:, :, max(-pady0, 0) : x.shape[2] - max(-pady1, 0), max(-padx0, 0) : x.shape[3] - max(-padx1, 0)]

    # Setup filter.
    f = f * (gain ** (f.ndim / 2))
    f = f.to(x.dtype)
    if not flip_filter:
        f = f.flip(list(range(f.ndim)))

    # Convolve with the filter.
    f = f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)
    if f.ndim == 4:
        x = conv2d_gradfix.conv2d(input=x, weight=f, groups=num_channels, use_custom_op=use_custom_conv2d_op)
    else:
        x = conv2d_gradfix.conv2d(input=x, weight=f.unsqueeze(2), groups=num_channels, use_custom_op=use_custom_conv2d_op)
        x = conv2d_gradfix.conv2d(input=x, weight=f.unsqueeze(3), groups=num_channels, use_custom_op=use_custom_conv2d_op)

    # Downsample by throwing away pixels.
    x = x[:, :, ::downy, ::downx]
    return x


#----------------------------------------------------------------------------

_upfirdn2d_custom_grad_cache = dict()


def _upfirdn2d_custom_grad(up=1, down=1, padding=0, flip_filter=False, gain=1, use_custom_conv2d_op=True):
    """Reference implementation of `upfirdn2d()` with custom gradient.
    """
    # Parse arguments.
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)

    # Lookup from cache.
    key = (upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain)
    if key in _upfirdn2d_custom_grad_cache:
        return _upfirdn2d_custom_grad_cache[key]

    # Forward op.
    class Upfirdn2dCustomGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, f): # pylint: disable=arguments-differ
            assert isinstance(x, torch.Tensor) and x.ndim == 4
            if f is None:
                f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            if f.ndim == 1 and f.shape[0] == 1:
                f = f.square().unsqueeze(0)  # Convert separable-1 into full-1x1.
            assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
            y = x
            if f.ndim == 2:
                y = upfirdn2d(y, f, up=[upx, upy], down=[downx, downy], padding=[padx0, padx1, pady0, pady1],
                    flip_filter=flip_filter, gain=gain, impl='ref', use_custom_conv2d_op=use_custom_conv2d_op)
            else:
                y = upfirdn2d(y, f.unsqueeze(0), up=[upx, 1], down=[downx, 1], padding=[padx0, padx1, 0, 0],
                    flip_filter=flip_filter, gain=1.0, impl='ref', use_custom_conv2d_op=use_custom_conv2d_op)
                y = upfirdn2d(y, f.unsqueeze(1), up=[1, upy], down=[1, downy], padding=[0, 0, pady0, pady1],
                    flip_filter=flip_filter, gain=gain, impl='ref', use_custom_conv2d_op=use_custom_conv2d_op)
            ctx.save_for_backward(f)
            ctx.x_shape = x.shape
            return y

        @staticmethod
        def backward(ctx, dy): # pylint: disable=arguments-differ
            f, = ctx.saved_tensors
            _, _, ih, iw = ctx.x_shape
            _, _, oh, ow = dy.shape
            fw, fh = _get_filter_size(f)
            p = [
                fw - padx0 - 1,
                iw * upx - ow * downx + padx0 - upx + 1,
                fh - pady0 - 1,
                ih * upy - oh * downy + pady0 - upy + 1,
            ]
            dx = None
            df = None

            if ctx.needs_input_grad[0]:
                dx = _upfirdn2d_custom_grad(up=down, down=up, padding=p, flip_filter=(not flip_filter), gain=gain).apply(dy, f)

            assert not ctx.needs_input_grad[1]
            return dx, df

    # Add to cache.
    _upfirdn2d_custom_grad_cache[key] = Upfirdn2dCustomGrad
    return Upfirdn2dCustomGrad


#----------------------------------------------------------------------------

_upfirdn2d_cuda_cache = dict()

def _upfirdn2d_cuda(up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Fast CUDA implementation of `upfirdn2d()` using custom ops.
    """
    # Parse arguments.
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)

    # Lookup from cache.
    key = (upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain)
    if key in _upfirdn2d_cuda_cache:
        return _upfirdn2d_cuda_cache[key]

    # Forward op.
    class Upfirdn2dCuda(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, f): # pylint: disable=arguments-differ
            assert isinstance(x, torch.Tensor) and x.ndim == 4
            if f is None:
                f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            if f.ndim == 1 and f.shape[0] == 1:
                f = f.square().unsqueeze(0) # Convert separable-1 into full-1x1.
            assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
            y = x
            if f.ndim == 2:
                y = _plugin.upfirdn2d(y, f, upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter, gain)
            else:
                y = _plugin.upfirdn2d(y, f.unsqueeze(0), upx, 1, downx, 1, padx0, padx1, 0, 0, flip_filter, 1.0)
                y = _plugin.upfirdn2d(y, f.unsqueeze(1), 1, upy, 1, downy, 0, 0, pady0, pady1, flip_filter, gain)
            ctx.save_for_backward(f)
            ctx.x_shape = x.shape
            return y

        @staticmethod
        def backward(ctx, dy): # pylint: disable=arguments-differ
            f, = ctx.saved_tensors
            _, _, ih, iw = ctx.x_shape
            _, _, oh, ow = dy.shape
            fw, fh = _get_filter_size(f)
            p = [
                fw - padx0 - 1,
                iw * upx - ow * downx + padx0 - upx + 1,
                fh - pady0 - 1,
                ih * upy - oh * downy + pady0 - upy + 1,
            ]
            dx = None
            df = None

            if ctx.needs_input_grad[0]:
                dx = _upfirdn2d_cuda(up=down, down=up, padding=p, flip_filter=(not flip_filter), gain=gain).apply(dy, f)

            assert not ctx.needs_input_grad[1]
            return dx, df

    # Add to cache.
    _upfirdn2d_cuda_cache[key] = Upfirdn2dCuda
    return Upfirdn2dCuda

#----------------------------------------------------------------------------

def filter2d(x, f, padding=0, flip_filter=False, gain=1, impl='cuda'):
    r"""Filter a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape matches the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        padding:     Padding with respect to the output. Can be a single number or a
                     list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + fw // 2,
        padx1 + (fw - 1) // 2,
        pady0 + fh // 2,
        pady1 + (fh - 1) // 2,
    ]
    return upfirdn2d(x, f, padding=p, flip_filter=flip_filter, gain=gain, impl=impl)

#----------------------------------------------------------------------------

def upsample2d(x, f, up=2, padding=0, flip_filter=False, gain=1, impl='cuda', use_custom_conv2d_op=True):
    r"""Upsample a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape is a multiple of the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x:                    Float32/float64/float16 input tensor of the shape
                              `[batch_size, num_channels, in_height, in_width]`.
        f:                    Float32 FIR filter of the shape
                              `[filter_height, filter_width]` (non-separable),
                              `[filter_taps]` (separable), or
                              `None` (identity).
        up:                   Integer upsampling factor. Can be a single int or a list/tuple
                              `[x, y]` (default: 1).
        padding:              Padding with respect to the output. Can be a single number or a
                              list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                              (default: 0).
        flip_filter:          False = convolution, True = correlation (default: False).
        gain:                 Overall scaling factor for signal magnitude (default: 1).
        impl:                 Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).
        use_custom_conv2d_op: Enable conv2d_gradfix (default: True)?

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    upx, upy = _parse_scaling(up)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw + upx - 1) // 2,
        padx1 + (fw - upx) // 2,
        pady0 + (fh + upy - 1) // 2,
        pady1 + (fh - upy) // 2,
    ]
    return upfirdn2d(x, f, up=up, padding=p, flip_filter=flip_filter, gain=gain*upx*upy, impl=impl,
        use_custom_conv2d_op=use_custom_conv2d_op)

#----------------------------------------------------------------------------

def downsample2d(x, f, down=2, padding=0, flip_filter=False, gain=1, impl='cuda', use_custom_conv2d_op=True):
    r"""Downsample a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape is a fraction of the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x:                    Float32/float64/float16 input tensor of the shape
                              `[batch_size, num_channels, in_height, in_width]`.
        f:                    Float32 FIR filter of the shape
                              `[filter_height, filter_width]` (non-separable),
                              `[filter_taps]` (separable), or
                              `None` (identity).
        down:                 Integer downsampling factor. Can be a single int or a list/tuple
                              `[x, y]` (default: 1).
        padding:              Padding with respect to the input. Can be a single number or a
                              list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                              (default: 0).
        flip_filter:          False = convolution, True = correlation (default: False).
        gain:                 Overall scaling factor for signal magnitude (default: 1).
        impl:                 Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).
        use_custom_conv2d_op: Enable conv2d_gradfix (default: True)?

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw - downx + 1) // 2,
        padx1 + (fw - downx) // 2,
        pady0 + (fh - downy + 1) // 2,
        pady1 + (fh - downy) // 2,
    ]
    return upfirdn2d(x, f, down=down, padding=p, flip_filter=flip_filter, gain=gain, impl=impl,
        use_custom_conv2d_op=use_custom_conv2d_op)

#----------------------------------------------------------------------------


# from kornia.filters import filter2d as kornia_filter2d


def _upfirdn2d_kornia(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1, use_custom_conv2d_op=True):
    _ = use_custom_conv2d_op # unused

    assert isinstance(x, torch.Tensor) and x.ndim == 4
    if f is None:
        f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    assert f.dtype == torch.float32 and not f.requires_grad
    batch_size, num_channels, in_height, in_width = x.shape
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)

    # Check that upsampled buffer is not smaller than the filter.
    upW = in_width * upx + padx0 + padx1
    upH = in_height * upy + pady0 + pady1
    assert upW >= f.shape[-1] and upH >= f.shape[0]

    # Upsample by inserting zeros.
    x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
    x = torch.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])
    print(f'kornia shape after inserted zeroes: {x.shape}')

    # Setup filter.
    # f = f * (gain ** (f.ndim / 2))
    f = f.to(x.dtype)
    if not flip_filter:
        f = f.flip(list(range(f.ndim)))

    # Convolve with the filter.
    f = f[None, None, :] * f[None, :, None]
    x = kornia_filter2d(x, f, border_type='constant', padding='same', normalized=True)

    # Downsample by throwing away pixels.
    x = x[:, :, ::downy, ::downx]
    return x


if __name__ == '__main__':
    # Test different upfirdn2d implementations to find the one which is supported by ONNX
    x = torch.tensor([
        [1, 1, 1],
        [2, 3, 2],
        [4, 4, 4]
    ], dtype=torch.float32)[None, None, ..., ...]

    f = torch.tensor([1, 3, 3, 1], dtype=torch.float32)

    up = 2
    down = 1
    padding = [1, 1, 1, 1]
    gain = 1
    y2 = _upfirdn2d_kornia(x, f, up=up, down=down, padding=padding, gain=gain)

    y1 = _upfirdn2d_ref(x, f, up=up, down=down, padding=padding, flip_filter=False, gain=gain, use_custom_conv2d_op=False)

    print(f'ref shape: {y1.shape}, kornia shape: {y2.shape}')