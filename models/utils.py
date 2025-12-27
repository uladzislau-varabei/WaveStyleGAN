import numpy as np
import torch.nn as nn
import torch.optim

from models import misc
from logger import log_message
from shared_utils import DEBUG_MODE


def scale_conv_weight_for_wavelet(layer, use_wavelet, init_wavelet_scales, forward_mode=False, inverse_mode=False,
    desc=None):
    if use_wavelet and init_wavelet_scales is not None:
        assert (forward_mode or inverse_mode) and not (forward_mode and inverse_mode)
        assert len(init_wavelet_scales) == 4, 'Scales for LL, LH, HL, HH must be provided'
        if forward_mode:
            # Forward mode => conv after DWT => scale input channels
            C = layer.weight.shape[1]
        elif inverse_mode:
            # Inverse mode => IDWT after conv => scale output channels
            C = layer.weight.shape[0]
        else:
            assert False
        assert len(init_wavelet_scales) == 4, 'Scales for LL, LH, HL, HH must be provided'
        s1, s2, s3, s4 = init_wavelet_scales
        assert C % 4 == 0
        C = C // 4
        scales_LL = torch.tensor([s1], dtype=torch.float32).repeat(C)
        scales_LH = torch.tensor([s2], dtype=torch.float32).repeat(C)
        scales_HL = torch.tensor([s3], dtype=torch.float32).repeat(C)
        scales_HH = torch.tensor([s4], dtype=torch.float32).repeat(C)
        scales = torch.cat([scales_LL, scales_LH, scales_HL, scales_HH], dim=0).to(layer.weight.device)
        if forward_mode:
            # Forward mode => conv after DWT => use original scales for input channels
            scales = scales[None, :, None, None]
        elif inverse_mode:
            # Inverse mode => IDWT after conv => use inverse scales for output channels
            scales = scales[:, None, None, None]
            scales = 1. / scales
        else:
            assert False
        with torch.no_grad():
            # Note: no_grad is needed
            layer.weight.data.copy_(scales * layer.weight.data)
            if DEBUG_MODE:
                print(f'Scaled weights for {desc}')
    else:
        if DEBUG_MODE:
            print(f'Skipped scaling weights for {desc}')
    return layer


def get_activation(activation, inplace=True):
    # Format: [activation, gain]
    activation_data = {
        'linear': [nn.Identity(), 1],
        'relu'  : [nn.ReLU(inplace=inplace), np.sqrt(2)],
        'lrelu' : [nn.LeakyReLU(0.2, inplace=inplace), np.sqrt(2)],
        'selu'  : [nn.SELU(inplace=inplace), 1],
        'silu'  : [nn.SiLU(inplace=inplace), np.sqrt(2)],
        # Note: this function is not used in the original implementation, so gain is taken based on other functions
        'gelu'  : [nn.GELU(), np.sqrt(2)]
    }
    assert activation in activation_data.keys(), f'activation={activation} is not supported'
    return activation_data[activation]


def get_normalization(norm, in_channels, groups=None, virtual_bs=None):
    if groups is None:
        groups = 4 if (in_channels // 4) == 0 else 2
    if virtual_bs is None:
        virtual_bs = 8  # auto param for BatchNormLocal2d
    norm = norm.lower()
    norm_data = {
        'batch_norm': BatchNorm2dFP32(in_channels),
        'batch_norm_local': BatchNormLocal2d(in_channels, virtual_bs, affine=True),
        'group_norm': GroupNormFP32(groups, in_channels),
    }
    assert norm in norm_data.keys()
    return norm_data[norm]


class BatchNorm2dFP32(nn.BatchNorm2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class GroupNormFP32(nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class BatchNormLocal1d(torch.nn.Module):
    # Thanks to: https://github.com/autonomousvision/stylegan-t/blob/main/networks/discriminator.py#L35
    # Some refactoring and fix for Torch 2.x
    def __init__(self, num_features, virtual_bs, affine=True, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = torch.nn.Parameter(torch.ones(self.num_features))
            self.bias = torch.nn.Parameter(torch.zeros(self.num_features))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x):
        orig_shape = x.shape
        B, C, L = orig_shape # B - batch, C - channels, L - length

        # Reshape batch into groups.
        G = int(np.ceil(B / self.virtual_bs))
        x = x.view(G, -1, C, L)

        # Calculate stats.
        mean = x.mean(dim=[1, 3], keepdim=True)
        var = x.var(dim=[1, 3], keepdim=True, correction=0)
        x = (x - mean) / (torch.sqrt(var + self.eps))

        # Optionally apply affine transform.
        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x.view(orig_shape)

    def extra_repr(self):
        return f'virtual_bs={self.virtual_bs}, affine={self.affine}'


class BatchNormLocal2d(torch.nn.Module):
    # Thanks to: https://github.com/autonomousvision/stylegan-t/blob/main/networks/discriminator.py#L35
    # Updated to 2d and fix for Torch 2.x
    # For axis refer to https://d2l.ai/chapter_convolutional-modern/batch-norm.html for
    def __init__(self, num_features, virtual_bs, affine=True, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = torch.nn.Parameter(torch.ones(self.num_features))
            self.bias = torch.nn.Parameter(torch.zeros(self.num_features))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x):
        orig_shape = x.shape
        orig_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        B, C, H, W = orig_shape
        assert B % self.virtual_bs == 0, \
            f'Batch size must be divisible by virtual batch size, got B={B}, VBS={self.virtual_bs}'

        # Reshape batch into groups.
        G = B // self.virtual_bs
        x = x.view(G, -1, C, H, W)

        # Calculate and apply stats.
        mean = x.mean(dim=[1, 3, 4], keepdim=True)
        var = x.var(dim=[1, 3, 4], keepdim=True, correction=0)
        x = (x - mean) / (torch.sqrt(var + self.eps))

        # Restore original shape.
        x = x.view(orig_shape)

        # Optionally apply affine transform.
        if self.affine:
            # x.shape = [B, C, H, W]
            x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        x = x.to(dtype=orig_dtype)
        return x

    def extra_repr(self):
        return f'virtual_bs={self.virtual_bs}, affine={self.affine}'


@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


def prepare_optimizer(module, optimizer_config, reg_interval, rank=0, logger=None):
    opt_name = optimizer_config['type'].lower()
    opt_lr = optimizer_config['lr']
    ratio = None
    message = ''
    if reg_interval is not None:
        assert reg_interval > 0
        ratio = reg_interval / (reg_interval + 1)
        message = f'{opt_name.upper()} opt - src lr={opt_lr:.6f}, '
        opt_lr = opt_lr * ratio
        message += f'upd lr={opt_lr:.6f}, reg interval={reg_interval}'
    module_params = module.parameters()
    if opt_name == 'adam':
        opt_eps = float(optimizer_config['eps'])
        betas = optimizer_config['betas']
        if ratio is not None:
            message += f'; src betas={betas}, '
            betas = [beta ** ratio for beta in betas]
            message += f'upd betas={betas}'
        assert 0.0 <= opt_eps < 0.1, f'Adam eps={opt_eps}'
        opt = torch.optim.Adam(module_params, lr=opt_lr, betas=betas, eps=opt_eps)
    else:
        assert False, f'Optimizer={opt_name} is not supported'
    if len(message) > 0:
        log_message(message, rank, logger)
    return opt


class GenInputsSampler(torch.nn.Module):
    def __init__(self, device, config):
        super().__init__()
        general_params = config['general_params']
        self.z_dim = general_params['z_dim']
        self.distribution = general_params['z_distribution'].lower()
        self.num_classes = general_params['num_classes']
        assert self.distribution in ['normal'], \
            f'Distribution {self.distribution} is not supported for GenInputsSampler'
        self.device = device

    def forward(self, batch_size):
        # 1. Generate z
        target_shape = [batch_size, self.z_dim]
        if self.distribution == 'normal':
            gen_z = torch.randn(target_shape, device=self.device)
        else:
            assert False
        # 2. Generate c
        if self.num_classes > 1:
            gen_c = torch.randint(0, self.num_classes, (batch_size, 1), device=self.device)
            gen_c = torch.nn.functional.one_hot(gen_c, num_classes=self.num_classes)
        else:
            gen_c = torch.zeros(batch_size, device=self.device)  # just some tensor
        return gen_z, gen_c
