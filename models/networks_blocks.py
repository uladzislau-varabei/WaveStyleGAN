from collections import OrderedDict

import numpy as np
import torch

from models.layers import modulated_conv2d_v1, \
    FlattenLayer, ResidualBlock, FourierSynthesisInput, BiasActivationLayer, \
    Conv2dLayer, SMConv2dLayer, SpectralConv2d, FastFourierConv2dLayer, \
    FullyConnectedLayer, SpectralFullyConnectedLayer, \
    MinibatchStdLayer, ToRGBLayer, WaveletLayer, WaveletUpFIRDn2dLayer
from models.ops import bias_act, upfirdn2d
from models.utils import scale_conv_weight_for_wavelet
from models import misc
from shared_utils import USE_DEFAULT_DATA_FORMAT_BEHAVIOUR, DEBUG_MODE, \
    USE_NEW_BIAS_ACT, log_ffc_output_info, USE_LEGACY_BEHAVIOUR


# ----- Generator -----

class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                         # Number of input channels.
        out_channels,                        # Number of output channels.
        w_dim,                               # Intermediate latent (W) dimensionality.
        resolution,                          # Resolution of this layer.
        kernel_size          = 3,            # Convolution kernel size.
        up                   = 1,            # Integer upsampling factor.
        use_noise            = True,         # Enable noise input?
        activation           = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter      = [1, 3, 3, 1], # Low-pass filter to apply when resampling activations.
        conv_clamp           = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        gain_mult            = 1,            # Additional gain factor to multiply by existing gain and clamp.
        channels_last        = False,        # Use channels_last format for the weights?
        use_custom_conv2d_op = True,         # Enable conv2d_gradfix (default: True)?
        upfirdn2d_impl       = 'cuda',       # Implementation of upfirdn2d op. One of ['ref', 'cuda', 'custom_grad'] (default: 'ref').
        bias_act_impl        = 'cuda',       # Implementation of bias with activation. One of ['ref', 'cuda'].
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.use_custom_conv2d_op = use_custom_conv2d_op
        self.upfirdn2d_impl = upfirdn2d_impl
        if USE_NEW_BIAS_ACT:
            self.bias_act_impl = bias_act_impl
            self.act_gain = bias_act.activation_funcs[activation]['def_gain']
        else:
            self.gain_mult = gain_mult
            self.bias_act = BiasActivationLayer(self.activation, clamp=self.conv_clamp, gain_mult=self.gain_mult)

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size]
        ).to(memory_format=memory_format))
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.in_channels, in_resolution, in_resolution])
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        x = modulated_conv2d_v1(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight,
            fused_modconv=fused_modconv, use_custom_conv2d_op=self.use_custom_conv2d_op,
            upfirdn2d_impl=self.upfirdn2d_impl)

        if USE_NEW_BIAS_ACT:
            act_gain = self.act_gain * gain
            act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
            x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp,
                    impl=self.bias_act_impl)
        else:
            x = self.bias_act(x, self.bias.to(x.dtype))
        return x

    def extra_repr(self):
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d},',
            f'resolution={self.resolution:d}, up={self.up}, activation={self.activation:s}'])


class SynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                                 # Number of input channels, 0 = first block.
        out_channels,                                # Number of output channels.
        w_dim,                                       # Intermediate latent (W) dimensionality.
        resolution,                                  # Resolution of this block.
        img_channels,                                # Number of output color channels.
        is_first,                                    # Is this the first block after Fourier input?
        is_last,                                     # Is this the last block?
        architecture              = 'skip',          # Architecture: 'orig', 'skip', 'resnet'.
        activation                = 'lrelu',         # Activation function: 'relu', 'lrelu', etc.
        resample_filter           = [1, 3, 3, 1],    # Low-pass filter to apply when resampling activations.
        conv_clamp                = 256,             # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16                  = False,           # Use FP16 for this block?
        fp16_channels_last        = False,           # Use channels-last memory format with FP16?
        channels_last             = False,           # Use channels-last memory format?
        use_custom_conv2d_op      = True,            # Enable conv2d_gradfix (default: True)?
        upfirdn2d_impl            = 'cuda',          # Implementation of upfirdn2d op. One of ['ref', 'cuda', 'custom_grad'] (default: 'ref').
        bias_act_impl             = 'cuda',          # Implementation of bias with activation. One of ['ref', 'cuda'].
        fused_modconv_default     = True,            # Default value of fused_modconv. 'inference_only' = True for inference, False for training.
        dwt_params                = None,            # Wavelet params.
        input_impl_idx            = None,            # Input implementation idx.
        input_type                = 'const',         # Input type. One of ['const', 'Fourier'].
        **layer_kwargs,                              # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_first = is_first
        self.is_last = is_last
        self.input_resolution = resolution if is_first else resolution // 2
        self.architecture = architecture
        self.use_fp16 = use_fp16
        if USE_DEFAULT_DATA_FORMAT_BEHAVIOUR:
            self.channels_last = (use_fp16 and fp16_channels_last)
        else:
            self.channels_last = channels_last
        self.fused_modconv_default = fused_modconv_default
        self.src_resample_filter = resample_filter
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.use_custom_conv2d_op = use_custom_conv2d_op
        self.upfirdn2d_impl = upfirdn2d_impl
        self.bias_act_impl = bias_act_impl
        assert input_impl_idx in [1, 2, 3], f'input_impl_idx={input_impl_idx} is not supported for SynthesisBlock'
        self.input_impl_idx = input_impl_idx
        input_type = input_type.lower()
        assert input_type in ['const', 'fourier'], f'input_type={input_type} is not supported for SynthesisBlock'
        self.input_type = input_type
        self.num_conv = 0
        self.num_torgb = 0
        self.use_wavelet = dwt_params['wavelet'] is not None
        if self.use_wavelet:
            if not self.is_first:
                self.img_channels = self.img_channels * 4
            self.wavelet_upfirdn2d = self.make_wavelet_upfirdn2d_layer(dwt_params)
        else:
            self.wavelet_upfirdn2d = None

        if in_channels == 0 and self.input_impl_idx in [2, 3]:
            if input_type.lower() == 'Fourier'.lower():
                # The same params as for StyleGAN-T
                # TODO: fix sampling_rate. Division by sampling_rate / 2 - bandwidth
                if USE_LEGACY_BEHAVIOUR:
                    sampling_rate_add = 0
                    bandwidth = 2
                else:
                    # sampling_rate_add = 4 if resolution == 4 else 0
                    sampling_rate_add = 0
                    bandwidth = 1 if resolution == 4 else 2
                self.fourier_input = FourierSynthesisInput(w_dim=self.w_dim,
                                                           channels=out_channels,
                                                           size=resolution,
                                                           sampling_rate=resolution + sampling_rate_add,
                                                           bandwidth=bandwidth,
                                                           bias_act_impl=self.bias_act_impl)
                self.num_conv += 1
                self.const_input = None
            elif input_type.lower() == 'const':
                self.fourier_input = None
                self.const_input = torch.nn.Parameter(
                    torch.randn([out_channels, resolution, resolution], dtype=torch.float32),
                )
            else:
                assert False
        else:
            self.fourier_input = None
            self.const_input = None

        if USE_NEW_BIAS_ACT:
            # Note: passed in forward
            gain_mult = None
        else:
            gain_mult = 1 / np.sqrt(2.) if self.architecture == 'resnet' else 1
        first_conv_up = 1 if self.is_first else 2
        common_kwargs = dict(channels_last=self.channels_last, use_custom_conv2d_op=self.use_custom_conv2d_op,
            upfirdn2d_impl=self.upfirdn2d_impl, bias_act_impl=self.bias_act_impl)
        if in_channels != 0 or self.input_impl_idx == 1:
            self.conv0 = SynthesisLayer(in_channels, out_channels,
                                        w_dim=w_dim,
                                        resolution=resolution,
                                        up=first_conv_up,
                                        activation=activation,
                                        resample_filter=resample_filter,
                                        conv_clamp=conv_clamp,
                                        **common_kwargs,
                                        **layer_kwargs)
            self.num_conv += 1
        else:
            self.conv0 = None
        self.conv1 = SynthesisLayer(out_channels, out_channels,
                                    w_dim=w_dim,
                                    resolution=resolution,
                                    activation=activation,
                                    conv_clamp=conv_clamp,
                                    gain_mult=gain_mult,
                                    **common_kwargs,
                                    **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            # For other resolutions img_channels is adjusted during wavelet initialization
            out_img_channels = self.img_channels * (4 if self.use_wavelet and is_first else 1)
            torgb = ToRGBLayer(out_channels, out_img_channels,
                               w_dim=w_dim,
                               conv_clamp=conv_clamp,
                               **common_kwargs)
            init_wavelet_scales = dwt_params['init_torgb_wavelet_scales']
            desc = f'toRGB, resolution={self.resolution}, inverse_mode'
            self.torgb = scale_conv_weight_for_wavelet(torgb,
                                                       use_wavelet=self.use_wavelet,
                                                       init_wavelet_scales=init_wavelet_scales,
                                                       inverse_mode=True,
                                                       desc=desc)
            self.num_torgb += 1

        self.skip = None
        self.skip_wavelet_upfirdn2d = None
        if in_channels != 0 and architecture == 'resnet':
            # TODO: should activation be used here?
            # TODO: check wavelets
            # conv2d_resample: if up > 1, down == 1, kernel_size == 1, then conv2d => upsample (fast path)
            if self.use_wavelet:
                self.skip_wavelet_upfirdn2d = self.make_wavelet_upfirdn2d_layer(dwt_params)
                self.skip_conv_up_factor = 1
            else:
                self.skip_conv_up_factor = 2
            self.skip = Conv2dLayer(in_channels, out_channels,
                                    kernel_size=1,
                                    bias=False,
                                    up=self.skip_conv_up_factor,
                                    resample_filter=resample_filter,
                                    gain_mult=gain_mult,
                                    **common_kwargs)

    def make_wavelet_upfirdn2d_layer(self, dwt_params):
        return WaveletUpFIRDn2dLayer(
            dwt_params['wavelet'],
            train_kernel=dwt_params['train_kernel'],
            scale_1d_coeffs=dwt_params['scale_1d_coeffs'],
            scale_2d_coeffs=dwt_params['scale_2d_coeffs'],
            coeffs_scales_2d_version=dwt_params['coeffs_scales_2d_version'],
            up=True,
            down=False,
            resample_filter=self.src_resample_filter,
            use_fp16=self.use_fp16,
            use_custom_conv2d_op=self.use_custom_conv2d_op,
            upfirdn2d_impl=self.upfirdn2d_impl
        )

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, update_emas=False, **layer_kwargs):
        # TODO: implement processing of EMAs
        _ = update_emas # unused
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        if ws.device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.training)

        # Input.
        if self.in_channels == 0 and self.input_impl_idx in [2, 3]:
            if self.input_type.lower() == 'Fourier'.lower():
                x = self.fourier_input(next(w_iter))
            elif self.input_type.lower() == 'const':
                x = self.const_input.to(dtype=dtype, memory_format=memory_format)
                # To avoid issues with gradient for G_reg loss add this very small value
                # (similar to L2 regularization term)
                x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1]) + 0.001 * ws[0, 0, 0] ** 2
            else:
                assert False
        else:
            misc.assert_shape(x, [None, self.in_channels, self.input_resolution, self.input_resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0 and self.input_impl_idx in [2, 3]:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
        elif self.architecture == 'resnet':
            # Note: in this case use gain=np.sqrt(0.5) for self.skip and self.conv1.
            # Currently, adjusted in __init__ for old bias_act implementation
            y = self.skip(x, gain=np.sqrt(0.5))
            if self.use_wavelet:
                # Skip layer: conv2d => upsampling (skip conv up factor depends on wavelet usage)
                y = self.skip_wavelet_upfirdn2d(y)
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            # x = y.add_(x) # fix custom grad error
            x = y + x
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            if self.use_wavelet:
                img = self.wavelet_upfirdn2d(img)
            else:
                img = upfirdn2d.upsample2d(img, self.resample_filter, up=2, impl=self.upfirdn2d_impl,
                    use_custom_conv2d_op=self.use_custom_conv2d_op)

        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            # img = img.add_(y) if img is not None else y
            img = img + y if img is not None else y # fix gradient for custom_grad upfirdn2d

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'


# ----- Discriminator -----

class DiscriminatorBaseBlock(torch.nn.Module):
    def __init__(self,
        resolution,                          # Resolution of this block.
        architecture         = None,         # Architecture. Only applicable to DiscriminatorBlock.
        conv_type            = None,         # Convolution layer type: ['base', 'selfmod', 'spectral'].
        resample_filter      = [1, 3, 3, 1], # Low-pass filter to apply when resampling activations.
        use_fp16             = False,        # Use FP16 for this block?
        fp16_channels_last   = False,        # Use channels-last memory format with FP16?
        channels_last        = False,        # Use channels-last memory format?
        use_custom_conv2d_op = True,         # Enable conv2d_gradfix (default: True)?
        upfirdn2d_impl       = 'cuda',       # Implementation of upfirdn2d op. One of ['ref', 'cuda', 'custom_grad'] (default: 'ref').
        bias_act_impl        = 'cuda',       # Implementation of bias with activation, One of ['ref', 'cuda'].
        dwt_params           = None,         # Wavelet params.
        ffc_params           = None,         # Fast Fourier convolution para,s.
    ):
        super().__init__()
        self.resolution = resolution
        self.architecture = architecture
        assert conv_type in ['base', 'selfmod', 'spectral']
        self.conv_type = conv_type
        self.use_fp16 = use_fp16
        if USE_DEFAULT_DATA_FORMAT_BEHAVIOUR:
            self.channels_last = (use_fp16 and fp16_channels_last)
        else:
            self.channels_last = channels_last
        self.src_resample_filter = resample_filter
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.use_custom_conv2d_op = use_custom_conv2d_op
        self.upfirdn2d_impl = upfirdn2d_impl
        self.bias_act_impl = bias_act_impl
        self.dwt_params = dwt_params
        self.use_wavelet = dwt_params['wavelet'] is not None if dwt_params is not None else False
        self.ffc_params = ffc_params
        self.use_ffc = self.ffc_params['use_ffc'] if self.ffc_params is not None else False
        # For compatibility with FFC
        self.is_first_ffc_block = None
        self.is_last_ffc_block = None

    def init_skip_wavelet_upfirdn2d(self, dwt_params):
        self.skip_wavelet_upfirdn2d = None
        self.skip_wavelet_upfirdn2d_local = None
        self.skip_wavelet_upfirdn2d_global = None
        if self.use_wavelet:
            if self.use_ffc:
                self.skip_wavelet_upfirdn2d_local = self.make_wavelet_upfirdn2d_layer(dwt_params)
                if not self.is_first_ffc_block:
                    self.skip_wavelet_upfirdn2d_global = self.make_wavelet_upfirdn2d_layer(dwt_params)
            else:
                self.skip_wavelet_upfirdn2d = self.make_wavelet_upfirdn2d_layer(dwt_params)

    def make_conv2d_layer(self, in_channels, out_channels, kernel_size, is_trainable,
        force_conv_type=None, bias=True, activation='linear', up=1, down=1, conv_clamp=None, gain_mult=1,
        is_first_block_conv=False, is_last_block_conv=False, force_default_conv=False):
        if self.use_ffc and (not force_default_conv):
            # print('FFConv2d layer')
            assert force_conv_type is None, f'FFC is enabled, but force_conv_type={force_conv_type}'
            layer = self.make_ffconv2d_layer(in_channels, out_channels, kernel_size=kernel_size, down=down,
                is_first_conv_block=is_first_block_conv, is_last_conv_block=is_last_block_conv,
                is_trainable=is_trainable)
            return layer
        conv_type = force_conv_type if force_conv_type is not None else self.conv_type
        if conv_type == 'selfmod' and (not force_default_conv):
            # print('SMConv2d layer')
            layer = SMConv2dLayer(in_channels, out_channels, kernel_size=kernel_size, bias=bias, activation=activation,
                up=up, down=down, resample_filter=self.src_resample_filter, conv_clamp=conv_clamp, gain_mult=gain_mult,
                channels_last=self.channels_last, use_custom_conv2d_op=self.use_custom_conv2d_op,
                upfirdn2d_impl=self.upfirdn2d_impl, bias_act_impl=self.bias_act_impl, trainable=is_trainable)
        elif conv_type == 'spectral' and (not force_default_conv):
            # print('SpectralConv2d layer')
            """
            # TODO: add this implementation later. For now, only cls layer in DiscriminaotrHead uses it, 
            # so no activation, up or down is required
            layer = SpectralConv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, activation=activation,
                up=up, down=down, resample_filter=self.src_resample_filter, conv_clamp=conv_clamp, gain_mult=gain_mult,
                channels_last=self.channels_last, use_custom_conv2d_op=self.use_custom_conv2d_op, 
                upfirdn2d_impl=self.upfirdn2d_impl, trainable=is_trainable)
            """
            assert up == 1 and down == 1 and activation == 'linear'
            layer = SpectralConv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, stride=1, padding=0)
        else:
            # print('Conv2d layer')
            # Note: fromrgb layer gives backward error with 'cuda' bias_act. Using 'ref' fixed error
            bias_act = 'ref' if kernel_size == 1 else self.bias_act_impl
            # bias_act = self.bias_act_impl
            layer = Conv2dLayer(in_channels, out_channels, kernel_size=kernel_size, bias=bias, activation=activation,
                up=up, down=down, resample_filter=self.src_resample_filter, conv_clamp=conv_clamp, gain_mult=gain_mult,
                channels_last=self.channels_last, use_custom_conv2d_op=self.use_custom_conv2d_op,
                upfirdn2d_impl=self.upfirdn2d_impl, bias_act_impl=bias_act, trainable=is_trainable)
        return layer

    def make_ffconv2d_layer(self, *args, **kwargs):
        raise NotImplementedError(f'{self.__class__.__name__}: ffconv2d is not implemented')
    def make_wavelet_upfirdn2d_layer(self, dwt_params, up=False, down=False):
        assert not (up and down) and (up or down), f'{self.__class__.__name__} error in WaveletUpfirdn2d init'
        return WaveletUpFIRDn2dLayer(
            dwt_params['wavelet'],
            train_kernel=dwt_params['train_kernel'],
            scale_1d_coeffs=dwt_params['scale_1d_coeffs'],
            scale_2d_coeffs=dwt_params['scale_2d_coeffs'],
            coeffs_scales_2d_version=dwt_params['coeffs_scales_2d_version'],
            up=up,
            down=down,
            resample_filter=self.src_resample_filter,
            use_fp16=self.use_fp16,
            use_custom_conv2d_op=self.use_custom_conv2d_op,
            upfirdn2d_impl=self.upfirdn2d_impl
        )

    def make_wavelet_layer(self, dwt_params, extract_and_fuse_coeffs, always_forward=False, always_inverse=False):
        return WaveletLayer(
            dwt_params['wavelet'],
            use_affine=dwt_params['use_affine'],
            init_affine_scales=dwt_params['init_affine_scales'],
            train_affine=dwt_params['train_affine'],
            affine_lr_multiplier=dwt_params['affine_lr_multiplier'],
            train_kernel=dwt_params['train_kernel'],
            scale_1d_coeffs=dwt_params['scale_1d_coeffs'],
            scale_2d_coeffs=dwt_params['scale_2d_coeffs'],
            coeffs_scales_2d_version=dwt_params['coeffs_scales_2d_version'],
            extract_and_fuse_coeffs=extract_and_fuse_coeffs,
            always_forward=always_forward,
            always_inverse=always_inverse
        )

    def split_into_local_global(self, x):
        assert self.use_ffc, 'Can only split x into local/global when FFC is enabled'
        if self.is_first_ffc_block:
            x_local, x_global = x, 0
        else:
            x_local, x_global = x[0], x[1]
        return x_local, x_global

    def check_force_fp32(self, x, img):
        if self.use_ffc and not self.is_first_ffc_block:
            if x is not None:
                force_fp32 = x[0].device.type != 'cuda'
            else:
                force_fp32 = img.device.type != 'cuda'
        elif (x if x is not None else img).device.type != 'cuda':
            force_fp32 = True
        else:
            force_fp32 = False
        return force_fp32

    def forward(self, x, img, force_fp32):
        raise NotImplementedError(f"Implement forward() for {self.__class__.__name__}")

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}, conv_type={self.conv_type:s}'


class DiscriminatorBlock(DiscriminatorBaseBlock):
    def __init__(self,
        in_channels,                         # Number of input channels, 0 = first block.
        tmp_channels,                        # Number of intermediate channels.
        out_channels,                        # Number of output channels.
        resolution,                          # Resolution of this block.
        img_channels,                        # Number of input color channels.
        first_layer_idx,                     # Index of the first layer.
        conv_type,                           # Convolution layer type: ['base', 'selfmod', 'spectral']
        architecture         = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation           = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter      = [1, 3, 3, 1], # Low-pass filter to apply when resampling activations.
        conv_clamp           = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16             = False,        # Use FP16 for this block?
        fp16_channels_last   = False,        # Use channels-last memory format with FP16?
        channels_last        = False,        # Use channels-last memory format?
        use_custom_conv2d_op = True,         # Enable conv2d_gradfix (default: True)?
        upfirdn2d_impl       = 'cuda',       # Implementation of upfirdn2d op. One of ['ref', 'cuda', 'custom_grad'] (default: 'ref').
        bias_act_impl        = 'cuda',       # Implementation of bias with activation. One of ['ref', 'cuda'].
        freeze_layers        = 0,            # Freeze-D: Number of layers to freeze.
        dwt_params           = None,         # Wavelet params.
        ffc_params           = None,         # Fast Fourier convolution params.
        is_first_ffc_block   = False,        # Is first resolution block for FFC?
        is_last_ffc_block    = False,        # Is last resolution block for FFC (before output head)?
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__(resolution=resolution, architecture=architecture, conv_type=conv_type,
            resample_filter=resample_filter, use_fp16=use_fp16, fp16_channels_last=fp16_channels_last,
            channels_last=channels_last, use_custom_conv2d_op=use_custom_conv2d_op,
            upfirdn2d_impl=upfirdn2d_impl, bias_act_impl=bias_act_impl,
            dwt_params=dwt_params, ffc_params=ffc_params)
        self.in_channels = in_channels
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.is_first_ffc_block = is_first_ffc_block
        self.is_last_ffc_block = is_last_ffc_block

        self.num_layers = 0
        if self.use_wavelet:
            self.wavelet_upfirdn2d = self.make_wavelet_upfirdn2d_layer(dwt_params, down=True)
        else:
            self.wavelet_upfirdn2d = None

        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable

        trainable_iter = trainable_gen()

        gain_mult = 1 / np.sqrt(2) if self.architecture == 'resnet' else 1

        self.init_fromrgb(in_channels, tmp_channels, activation=activation, conv_clamp=conv_clamp,
            is_trainable=next(trainable_iter))

        self.conv0 = self.make_conv2d_layer(tmp_channels, tmp_channels, is_trainable=next(trainable_iter),
            kernel_size=3, activation=activation, conv_clamp=conv_clamp,
            is_first_block_conv=True)
        self.conv1 = self.make_conv2d_layer(tmp_channels, out_channels, is_trainable=next(trainable_iter),
            kernel_size=3, activation=activation, down=2, conv_clamp=conv_clamp, gain_mult=gain_mult,
            is_last_block_conv=True)

        if architecture == 'resnet':
            # conv2d_resample: if down > 1, up == 1, kernel_size == 1, then downsample => conv2d (fast path)
            self.init_skip_wavelet_upfirdn2d(dwt_params)
            self.init_skip_conv2d(tmp_channels, out_channels, gain_mult=gain_mult, is_trainable=next(trainable_iter))

    def init_fromrgb(self, in_channels, tmp_channels, activation, conv_clamp, is_trainable):
        self.fromrgb = None
        self.fromrgb_local = None
        self.fromrgb_global = None
        shared_kwargs = dict(is_trainable=is_trainable, kernel_size=1, activation=activation, conv_clamp=conv_clamp,
            force_default_conv=True)
        scale_shared_kwargs = dict(forward_mode=True, use_wavelet=self.use_wavelet,
            init_wavelet_scales=self.dwt_params['init_fromrgb_wavelet_scales'])
        if in_channels == 0 or self.architecture == 'skip':
            if self.use_ffc:
                # Use ratio_in_global as fromrgb layer is applied before FFC inside block
                ratio_in_global = 0 if self.is_first_ffc_block else self.ffc_params['ratio_in_global']
                global_out_channels = int(ratio_in_global * tmp_channels)
                local_out_channels = tmp_channels - global_out_channels
                fromrgb_local = self.make_conv2d_layer(self.img_channels, local_out_channels, **shared_kwargs)
                desc = f'fromRGB_local, resolution={self.resolution}, forward_mode'
                self.fromrgb_local = scale_conv_weight_for_wavelet(fromrgb_local, desc=desc, **scale_shared_kwargs)
                # No global info for the first block. The last block still has global input
                if not self.is_first_ffc_block:
                    fromrgb_global = self.make_conv2d_layer(self.img_channels, global_out_channels, **shared_kwargs)
                    desc = f'fromRGB_global, resolution={self.resolution}, forward_mode'
                    self.fromrgb_global = scale_conv_weight_for_wavelet(fromrgb_global, desc=desc, **scale_shared_kwargs)
            else:
                fromrgb = self.make_conv2d_layer(self.img_channels, tmp_channels, **shared_kwargs)
                desc = f'fromRGB, resolution={self.resolution}, forward_mode'
                self.fromrgb = scale_conv_weight_for_wavelet(fromrgb, desc=desc, **scale_shared_kwargs)

    def init_skip_wavelet_upfirdn2d(self, dwt_params):
        self.skip_wavelet_upfirdn2d = None
        self.skip_wavelet_upfirdn2d_local = None
        self.skip_wavelet_upfirdn2d_global = None
        down = True  # only downsampling in this block
        if self.use_wavelet:
            if self.use_ffc:
                self.skip_wavelet_upfirdn2d_local = self.make_wavelet_upfirdn2d_layer(dwt_params, down=down)
                if not self.is_first_ffc_block:
                    self.skip_wavelet_upfirdn2d_global = self.make_wavelet_upfirdn2d_layer(dwt_params, down=down)
            else:
                self.skip_wavelet_upfirdn2d = self.make_wavelet_upfirdn2d_layer(dwt_params, down=down)

    def init_skip_conv2d(self, tmp_channels, out_channels, gain_mult, is_trainable):
        self.skip = None
        self.skip_local = None
        self.skip_global = None
        skip_conv_down_factor = 1 if self.use_wavelet else 2
        shared_kwargs = dict(kernel_size=1, is_trainable=is_trainable, bias=False, down=skip_conv_down_factor,
            gain_mult=gain_mult, force_default_conv=True)
        if self.use_ffc:
            ratio_in_global = 0 if self.is_first_ffc_block else self.ffc_params['ratio_in_global']
            ratio_out_global = 0 if self.is_last_ffc_block else self.ffc_params['ratio_out_global']
            skip_global_in_channels = int(ratio_in_global * tmp_channels)
            skip_global_out_channels = int(ratio_out_global * out_channels)
            skip_local_in_channels = tmp_channels - skip_global_in_channels
            skip_local_out_channels = out_channels - skip_global_out_channels
            self.skip_local = self.make_conv2d_layer(skip_local_in_channels, skip_local_out_channels,
                **shared_kwargs)
            # No global info for the first block. And no global info for the last block (empty output)
            if not (self.is_first_ffc_block or self.is_last_ffc_block):
                self.skip_global = self.make_conv2d_layer(skip_global_in_channels, skip_global_out_channels,
                    **shared_kwargs)
        else:
            self.skip = self.make_conv2d_layer(tmp_channels, out_channels, **shared_kwargs)

    def make_ffconv2d_layer(self, in_channels, out_channels, kernel_size, down=1,
        is_first_conv_block=False, is_last_conv_block=False, is_trainable=True):
        # Note: this layer can only be used for intermediate blocks
        ratio_in_global = 0 if self.is_first_ffc_block and is_first_conv_block else self.ffc_params['ratio_in_global']
        ratio_out_global = 0 if self.is_last_ffc_block and is_last_conv_block else self.ffc_params['ratio_out_global']
        return FastFourierConv2dLayer(in_channels, out_channels,
            impl_idx=self.ffc_params['impl_idx'],
            ratio_in_global=ratio_in_global, ratio_out_global=ratio_out_global,
            kernel_size=kernel_size,  down=down, resample_filter=self.src_resample_filter,
            fft_norm=self.ffc_params['fft_norm'], use_lfu=self.ffc_params['use_lfu'],
            lfu_mode=self.ffc_params['lfu_mode'], conv_type=self.ffc_params['conv_type'],
            conv_clamp=self.ffc_params['conv_clamp'], conv_norm=self.ffc_params['conv_norm'],
            activation=self.ffc_params['activation'], use_custom_conv2d_op=self.use_custom_conv2d_op,
            upfirdn2d_impl=self.upfirdn2d_impl, bias_act_impl=self.bias_act_impl,
            dwt_params=self.dwt_params, channels_last=self.channels_last
        )

    def split_into_local_global(self, x, post_block=False):
        assert self.use_ffc, 'Can only split x into local/global when FFC is enabled'
        if self.is_first_ffc_block and (not post_block):
            x_local, x_global = x, 0
        else:
            x_local, x_global = x[0], x[1]
        return x_local, x_global

    def forward_ffc_resnet_skip(self, x, is_local=False, is_global=False):
        assert not (is_local and is_global)
        if is_local:
            y = self.skip_wavelet_upfirdn2d_local(x) if self.use_wavelet else x
            y = self.skip_local(y)
        elif is_global:
            if self.is_first_ffc_block or self.is_last_ffc_block:
                y = 0
            else:
                y = self.skip_wavelet_upfirdn2d_global(x) if self.use_wavelet else x
                y = self.skip_global(y)
        else:
            assert False
        return y

    def forward_resnet_skip(self, x):
        if self.use_ffc:
            x_local, x_global = self.split_into_local_global(x)
            y_local = self.forward_ffc_resnet_skip(x_local, is_local=True)
            y_global = self.forward_ffc_resnet_skip(x_global, is_global=True)
            y = (y_local, y_global)
        else:
            y = self.skip_wavelet_upfirdn2d(x) if self.use_wavelet else x
            y = self.skip(y)
        return y

    def forward_fromrgb(self, x, img):
        if self.use_ffc:
            y_local = self.fromrgb_local(img)
            if self.is_first_ffc_block:
                y_global = 0
            else:
                y_global = self.fromrgb_global(img)
            if x is not None:
                x_local, x_global = self.split_into_local_global(x)
                x_local = x_local + y_local
                x_global = x_global + y_global
            else:
                x_local = y_local
                x_global = y_global
            x = (x_local, x_global)
        else:
            y = self.fromrgb(img)
            x = x + y if x is not None else y
        return x

    def check_force_fp32(self, x, img):
        if self.use_ffc and not self.is_first_ffc_block:
            if x is not None:
                force_fp32 = x[0].device.type != 'cuda'
            else:
                force_fp32 = img.device.type != 'cuda'
        elif (x if x is not None else img).device.type != 'cuda':
            force_fp32 = True
        else:
            force_fp32 = False
        return force_fp32

    def forward(self, x, img, force_fp32=False):
        force_fp32 = True if force_fp32 else self.check_force_fp32(x, img)
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            if self.use_ffc and not self.is_first_ffc_block:
                x_local, x_global = self.split_into_local_global(x)
                misc.assert_shape(x_local, [None, None, self.resolution, self.resolution])
                misc.assert_shape(x_global, [None, None, self.resolution, self.resolution])
                x_local = x_local.to(dtype=dtype, memory_format=memory_format)
                x_global = x_global.to(dtype=dtype, memory_format=memory_format)
                x = (x_local, x_global)
            else:
                misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
                x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = self.forward_fromrgb(x, img)
            if self.architecture == 'skip':
                if self.use_wavelet:
                    img = self.wavelet_upfirdn2d(img)
                else:
                    img = upfirdn2d.downsample2d(img, self.resample_filter, down=2,
                        impl=self.upfirdn2d_impl, use_custom_conv2d_op=self.use_custom_conv2d_op)
            else:
                img = None

        # Main layers.
        if self.architecture == 'resnet':
            # Note: in this case use gain=np.sqrt(0.5) for self.skip and self.conv1. Currently, adjusted in __init__
            y = self.forward_resnet_skip(x)
            x = self.conv0(x)
            x = self.conv1(x)
            if self.use_ffc:
                if DEBUG_MODE:
                    log_ffc_output_info(x, 'x')
                    log_ffc_output_info(y, 'y')
                x_local, x_global = self.split_into_local_global(x, post_block=True)
                y_local, y_global = self.split_into_local_global(y, post_block=True)
                x_local = x_local + y_local
                x_global = x_global + y_global if not self.is_last_ffc_block else None
                x = (x_local, x_global)
            else:
                # x = y.add_(x) # fix custom grad error
                x = y + x
        else:
            x = self.conv0(x)
            x = self.conv1(x)
            if self.use_ffc:
                if DEBUG_MODE:
                    log_ffc_output_info(x, 'x')
                x_local, x_global = self.split_into_local_global(x, post_block=True)
                if self.is_last_ffc_block:
                    assert x_global == 0
                    x_global = None
                x = (x_local, x_global)

        if self.use_ffc:
            x_global_cond = x[1] is None if self.is_last_ffc_block else x[1].dtype == dtype
            assert x[0].dtype == dtype and x_global_cond, \
                f'x[0].dtype={x[0].dtype}, x[1].dtype={-1 if self.is_last_ffc_block else x[1].dtype}, dtype={dtype}'
        else:
            assert x.dtype == dtype
        return x, img


class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                     # Number of input channels.
        c_map_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                      # Resolution of this block.
        img_channels,                    # Number of input color channels.
        conv_type,                       # Convolution layer type: ['base', 'selfmod', 'spectral']
        architecture         = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size     = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels   = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation           = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp           = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_custom_conv2d_op = True,     # Enable conv2d_gradfix (default: True)?
        upfirdn2d_impl       = 'cuda',   # Implementation of upfirdn2d op. One of ['ref', 'cuda', 'custom_grad'] (default: 'ref').
        bias_act_impl        = 'cuda',   # Implementation of bias with activation. One of ['ref', 'cuda'].
        dwt_params           = None,     # Wavelet params.
        ffc_params           = None,     # Fast Fourier convolution params.
    ):
        # Note: Usage of Fast Fourier convolution is limited here as MBSTD layer adds one channel and
        # there are no two consecutive conv layers
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.c_map_dim = c_map_dim
        self.resolution = resolution
        self.img_channels = img_channels
        assert conv_type in ['base', 'selfmod', 'spectral']
        self.conv_type = conv_type
        self.architecture = architecture
        self.activation = activation
        self.use_custom_conv2d_op = use_custom_conv2d_op
        self.bias_act_impl = bias_act_impl
        self.use_wavelet = dwt_params['wavelet'] is not None
        self.use_ffc = ffc_params['use_ffc']

        if architecture == 'skip':
            fromrgb = self.make_conv2d_layer(img_channels, in_channels, kernel_size=1, force_default_conv=True)
            init_fromrgb_wavelet_scales = dwt_params['init_fromrgb_wavelet_scales']
            desc = f'fromRGB, resolution={self.resolution} (DiscriminatorEpilogue), forward_mode'
            self.fromrgb = scale_conv_weight_for_wavelet(fromrgb,
                                                         use_wavelet=self.use_wavelet,
                                                         init_wavelet_scales=init_fromrgb_wavelet_scales,
                                                         forward_mode=True,
                                                         desc=desc)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) \
            if mbstd_num_channels > 0 else None
        self.conv = self.make_conv2d_layer(in_channels + mbstd_num_channels, in_channels, kernel_size=3,
            conv_clamp=conv_clamp)
        # Note: original models use FC layers. Replace with SpectralFC layer
        self.fc = SpectralFullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation,
            bias_act_impl=bias_act_impl)
        self.out = SpectralFullyConnectedLayer(in_channels, 1 if c_map_dim == 0 else c_map_dim,
            bias_act_impl=bias_act_impl)

    def make_conv2d_layer(self, in_channels, out_channels, kernel_size, conv_clamp=None, force_default_conv=False):
        if self.conv_type == 'selfmod' and (not force_default_conv):
            layer = SMConv2dLayer(in_channels, out_channels, kernel_size=kernel_size, activation=self.activation,
                conv_clamp=conv_clamp, use_custom_conv2d_op=self.use_custom_conv2d_op,
                bias_act_impl=self.bias_act_impl)
        elif self.conv_type == 'spectral':
            assert self.activation == 'linear', \
                'SpectralConv2d layer is implemented only for up=1, down1, activation=linear'
            layer = SpectralConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0)
        else:
            layer = Conv2dLayer(in_channels, out_channels, kernel_size=kernel_size, activation=self.activation,
                conv_clamp=conv_clamp, use_custom_conv2d_op=self.use_custom_conv2d_op,
                bias_act_impl=self.bias_act_impl)
        return layer

    def forward(self, x, img, cmap, force_fp32=False):
        # Process FFC output.
        if self.use_ffc:
            # FFC: x_global must be empty
            x_local, x_global = x
            x = x_local
            assert x_global is None

        # Check input shape and set params.
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning.
        if self.c_map_dim > 0:
            misc.assert_shape(cmap, [None, self.c_map_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.c_map_dim))

        assert x.dtype == dtype
        return x

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'


# ----- Discriminator projection -----

class CCMBlock(DiscriminatorBaseBlock):
    # Cross channel mixing block
    def __init__(self,
        in_channels,                         # Number of input channels, 0 = first block.
        out_channels,                        # Number of output channels.
        resolution,                          # Resolution of this block.
        conv_type,                           # Convolution layer type: ['base', 'selfmod', 'spectral']
        architecture         = 'base',       # Architecture: 'base' (for compatibility).
        activation           = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        conv_clamp           = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16             = False,        # Use FP16 for this block?
        fp16_channels_last   = False,        # Use channels-last memory format with FP16?
        channels_last        = False,        # Use channels-last memory format?
        use_custom_conv2d_op = True,         # Enable conv2d_gradfix (default: True)?
        bias_act_impl        = 'cuda',       # Implementation of bias with activation. One of ['ref', 'cuda'].
    ):
        assert architecture in ['base']  # only base architecture is available for CCM
        super().__init__(resolution=resolution, architecture=architecture, conv_type=conv_type,
            use_fp16=use_fp16, fp16_channels_last=fp16_channels_last, channels_last=channels_last,
            use_custom_conv2d_op=use_custom_conv2d_op, bias_act_impl=bias_act_impl)
        self.in_channels = in_channels
        self.architecture = architecture

        self.conv = self.make_conv2d_layer(in_channels, out_channels, is_trainable=True,
            kernel_size=1, activation=activation, conv_clamp=conv_clamp)

    def check_force_fp32(self, x, img):
        if self.use_ffc and not self.is_first_ffc_block:
            if x is not None:
                force_fp32 = x[0].device.type != 'cuda'
            else:
                force_fp32 = img.device.type != 'cuda'
        elif (x if x is not None else img).device.type != 'cuda':
            force_fp32 = True
        else:
            force_fp32 = False
        return force_fp32

    def forward(self, x, img, force_fp32=False):
        # force_fp32 = True if force_fp32 else self.check_force_fp32(x, img)
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if type(x) is tuple:
            # Concat FFC output
            x_local, x_global = x
            x = torch.cat([x_local, x_global], dim=1)

        # Note: for src blocks res is referred to input res, output res is reduced by 2
        misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
        x = x.to(dtype=dtype, memory_format=memory_format)

        x = self.conv(x)

        assert x.dtype == dtype
        return x


class CSMBlock(DiscriminatorBaseBlock):
    # Cross stage mixing block
    def __init__(self,
        csm_in_channels,                     # Number of input channels of CSM output.
        ccm_in_channels,                     # Number of input channels of CCM output.
        out_channels,                        # Number of output channels.
        resolution,                          # Resolution of this block.
        conv_type,                           # Convolution layer type: ['base', 'selfmod', 'spectral']
        architecture         = 'base',       # Architecture: 'base' (for compatibility).
        activation           = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter      = [1, 3, 3, 1], # Low-pass filter to apply when resampling activations.
        conv_clamp           = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16             = False,        # Use FP16 for this block?
        fp16_channels_last   = False,        # Use channels-last memory format with FP16?
        channels_last        = False,        # Use channels-last memory format?
        use_custom_conv2d_op = True,         # Enable conv2d_gradfix (default: True)?
        upfirdn2d_impl       = 'cuda',       # Implementation of upfirdn2d op. One of ['ref', 'cuda', 'custom_grad'] (default: 'ref').
        bias_act_impl        = 'cuda',       # Implementation of bias with activation. One of ['ref', 'cuda'].
        dwt_params           = None,         # Wavelet params.
    ):
        assert architecture in ['base']  # only base architecture is available for CCM
        super().__init__(resolution=resolution, architecture=architecture, conv_type=conv_type,
            resample_filter=resample_filter, use_fp16=use_fp16, fp16_channels_last=fp16_channels_last,
            channels_last=channels_last, use_custom_conv2d_op=use_custom_conv2d_op,
            upfirdn2d_impl=upfirdn2d_impl, bias_act_impl=bias_act_impl, dwt_params=dwt_params)
        self.csm_in_channels = csm_in_channels
        self.ccm_in_channels = ccm_in_channels
        self.architecture = architecture
        self.gain = 1 / np.sqrt(2.)  # scale for addition of CCM and CSM outputs

        self.conv0 = self.make_conv2d_layer(csm_in_channels, ccm_in_channels, is_trainable=True,
            kernel_size=3, activation=activation, up=2, conv_clamp=conv_clamp)
        self.conv1 = self.make_conv2d_layer(ccm_in_channels, out_channels, is_trainable=True,
            kernel_size=1, activation=activation, conv_clamp=conv_clamp)

    def check_force_fp32(self, x, img):
        if self.use_ffc and not self.is_first_ffc_block:
            if x is not None:
                force_fp32 = x[0].device.type != 'cuda'
            else:
                force_fp32 = img.device.type != 'cuda'
        elif (x if x is not None else img).device.type != 'cuda':
            force_fp32 = True
        else:
            force_fp32 = False
        return force_fp32

    def forward(self, x, img, force_fp32=False):
        # force_fp32 = True if force_fp32 else self.check_force_fp32(x, img)
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        # Note: this layer is not applied to output from min projected resolution
        # Note: for src blocks res is referred to input res, output res is reduced by 2
        x_csm, x_ccm = x
        misc.assert_shape(x_csm, [None, self.csm_in_channels, self.resolution // 4, self.resolution // 4])
        misc.assert_shape(x_ccm, [None, self.ccm_in_channels, self.resolution // 2, self.resolution // 2])
        x_csm = x_csm.to(dtype=dtype, memory_format=memory_format)
        x_ccm = x_ccm.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        x_csm = self.conv0(x_csm)
        x = self.gain * (x_csm + x_ccm)
        x = self.conv1(x)

        assert x.dtype == dtype
        return x


class DiscriminatorHead(DiscriminatorBaseBlock):
    def __init__(self,
        in_channels,                         # Number of input channels, 0 = first block.
        main_channels,                       # Number of channels in main block, None to usein)channels
        c_dim,                               # Number of classes, 0 = no label.
        c_map_dim,                           # Dimensionality of mapped conditioning label
                                             # (differs from param in default epilogue).
        resolution,                          # Resolution of this block.
        target_resolution,                   # Target resolution for downsampling inputs.
                                             # If input has lower resolution, then no upsampling is applied.
        conv_type,                           # Convolution layer type: ['base', 'selfmod', 'spectral']
        architecture         = 'sg2',        # Architecture: 'sg2', 'sg-t'.
        mbstd_group_size     = 4,            # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels   = 1,            # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation           = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter      = [1, 3, 3, 1], # Low-pass filter to apply when resampling activations.
        conv_clamp           = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16             = False,        # Use FP16 for this block?
        fp16_channels_last   = False,        # Use channels-last memory format with FP16?
        channels_last        = False,        # Use channels-last memory format?
        use_custom_conv2d_op = True,         # Enable conv2d_gradfix (default: True)?
        upfirdn2d_impl       = 'cuda',       # Implementation of upfirdn2d op. One of ['ref', 'cuda', 'custom_grad'] (default: 'ref').
        bias_act_impl        = 'cuda',       # Implementation of bias with activation. One of ['ref', 'cuda'].
        dwt_params           = None,         # Wavelet params.
    ):
        assert architecture in ['sg2', 'sg-t', 'swg', 'swg-v2']
        super().__init__(resolution=resolution, architecture=architecture, conv_type=conv_type,
            resample_filter=resample_filter, use_fp16=use_fp16, fp16_channels_last=fp16_channels_last,
            channels_last=channels_last, use_custom_conv2d_op=use_custom_conv2d_op,
            upfirdn2d_impl=upfirdn2d_impl, bias_act_impl=bias_act_impl,
            dwt_params=dwt_params, ffc_params=None)
        self.in_channels = in_channels
        self.c_dim = c_dim
        self.c_map_dim = c_map_dim
        self.resolution = resolution
        self.target_resolution = target_resolution
        self.architecture = architecture
        self.mbstd_group_size = mbstd_group_size
        self.mbstd_num_channels = mbstd_num_channels
        self.activation = activation

        # Resolution refers to block input size, output size is reduced by 2,
        # so projection for resolution 16 has input size 8
        input_res = resolution // 2
        layers = []
        names = []
        if main_channels is None:
            main_channels = self.in_channels
        while input_res > target_resolution:
            input_res = input_res // 2
            in_channels = self.in_channels if len(layers) == 0 else main_channels
            layers.append(self.make_down_block(conv_clamp=conv_clamp, in_channels=in_channels, main_channels=main_channels))
            names.append(f'down{input_res}')
        # No downsampling layers => keep original number of channels
        main_channels = main_channels if len(layers) > 0 else self.in_channels
        layers.append(self.make_main_block(conv_clamp=conv_clamp, input_res=input_res, in_channels=main_channels, main_channels=main_channels))
        names.append('epilogue')
        self.main = torch.nn.Sequential(OrderedDict(zip(names, layers)))

        cls_kernel_size = {
            'sg2': 1,
            'sg-t': 4,
            'swg': 4,
            'swg-v2': 1
        }[self.architecture]
        if self.c_dim > 0:
            self.c_mapper = FullyConnectedLayer(self.c_dim, self.c_map_dim, bias_act_impl=self.bias_act_impl)
            self.c_map_gain = 1 / (np.sqrt(self.c_map_dim))
            self.cls = self.make_conv2d_layer(main_channels, self.c_map_dim, kernel_size=cls_kernel_size,
                is_trainable=True, force_conv_type='spectral')
        else:
            self.c_mapper = None
            self.c_map_gain = 1
            self.cls = self.make_conv2d_layer(main_channels, 1, kernel_size=cls_kernel_size,
                is_trainable=True, force_conv_type='spectral')

    def make_down_block(self, in_channels, main_channels, conv_clamp):
        shared_conv_kwargs = dict(is_trainable=True, activation=self.activation, conv_clamp=conv_clamp)
        use_reduction = True
        channel_reduction = 4
        if self.use_wavelet:
            # After fusing wavelet coeffs shape update: [N, C, H, W] => [N, 2 * C, H // 2, W // 2]
            wavelet_layer = self.make_wavelet_layer(self.dwt_params, extract_and_fuse_coeffs=True, always_forward=True)
            if use_reduction:
                channels = in_channels // channel_reduction
                squeeze_conv_layer = self.make_conv2d_layer(in_channels * 2, channels, kernel_size=1,
                    **shared_conv_kwargs)
                conv_layer = self.make_conv2d_layer(channels, channels, kernel_size=3, **shared_conv_kwargs)
                excitation_conv_layer = self.make_conv2d_layer(channels, main_channels, kernel_size=1,
                    **shared_conv_kwargs)
                layers = [wavelet_layer, squeeze_conv_layer, conv_layer, excitation_conv_layer]
                names = ['wavelet', 'squeeze_conv', 'conv', 'excitation_conv']
            else:
                conv_layer = self.make_conv2d_layer(in_channels * 2, main_channels, kernel_size=3, **shared_conv_kwargs)
                layers = [wavelet_layer, conv_layer]
                names = ['wavelet', 'conv']
        else:
            if use_reduction:
                channels = in_channels // channel_reduction
                squeeze_conv_layer = self.make_conv2d_layer(in_channels, channels, kernel_size=1,
                    **shared_conv_kwargs)
                conv_layer = self.make_conv2d_layer(channels, channels, kernel_size=3, **shared_conv_kwargs)
                excitation_conv_layer = self.make_conv2d_layer(channels, main_channels, kernel_size=1,
                    **shared_conv_kwargs)
                layers = [squeeze_conv_layer, conv_layer, excitation_conv_layer]
                names = ['squeeze_conv', 'conv', 'excitation_conv']
            else:
                conv_layer = self.make_conv2d_layer(in_channels, kernel_size=3, down=2, **shared_conv_kwargs)
                layers = [conv_layer]
                names = ['conv']
        return torch.nn.Sequential(OrderedDict(zip(names, layers)))

    def make_main_block(self, in_channels, main_channels, conv_clamp, input_res):
        shared_conv_kwargs = dict(is_trainable=True, activation=self.activation, conv_clamp=conv_clamp)
        if self.architecture == 'sg2':
            # Similar to StyleGAN2
            layers = []
            names = []
            if self.mbstd_num_channels > 0:
                mbstd_layer = MinibatchStdLayer(group_size=self.mbstd_group_size, num_channels=self.mbstd_num_channels)
                layers.append(mbstd_layer)
                names.append('mbstd')
            conv_layer = self.make_conv2d_layer(in_channels + self.mbstd_num_channels, main_channels,
                kernel_size=3, **shared_conv_kwargs)
            flatten_layer = FlattenLayer()
            fc_layer = FullyConnectedLayer(main_channels * (input_res ** 2), main_channels,
                activation=self.activation, bias_act_impl=self.bias_act_impl)
            layers += [conv_layer, flatten_layer, fc_layer]
            names += ['conv', 'flatten', 'fc']
        elif self.architecture == 'sg-t':
            # Similar to StyleGAN-T. Different kernel sizes but no MBSTD layer and residual connection is used
            conv0_layer = self.make_conv2d_layer(in_channels, main_channels, kernel_size=3,
                **shared_conv_kwargs)
            conv1_layer = self.make_conv2d_layer(main_channels, main_channels, kernel_size=3,
                **shared_conv_kwargs)
            layers = [conv0_layer, ResidualBlock(conv1_layer)]
            names = ['conv0', 'conv1_residual']
        elif self.architecture == 'swg':
            # Similar to StyleGAN-T but with MBSTD layer
            assert self.mbstd_num_channels > 0, 'Head architecture SWG must only be used with MBSTD layer'
            mbstd_layer = MinibatchStdLayer(group_size=self.mbstd_group_size, num_channels=self.mbstd_num_channels)
            conv0_layer = self.make_conv2d_layer(in_channels + self.mbstd_num_channels, main_channels,
                kernel_size=3, **shared_conv_kwargs)
            conv1_layer = self.make_conv2d_layer(main_channels, main_channels,
                kernel_size=3, **shared_conv_kwargs)
            layers = [mbstd_layer, conv0_layer, ResidualBlock(conv1_layer)]
            names = ['mbstd', 'conv0', 'conv1_residual']
        elif self.architecture == 'swg-v2':
            # Similar to StyleGAN2 but with SpectralFullyConnected layer
            layers = []
            names = []
            if self.mbstd_num_channels > 0:
                mbstd_layer = MinibatchStdLayer(group_size=self.mbstd_group_size, num_channels=self.mbstd_num_channels)
                layers.append(mbstd_layer)
                names.append('mbstd')
            conv_layer = self.make_conv2d_layer(in_channels + self.mbstd_num_channels, main_channels,
                kernel_size=3, **shared_conv_kwargs)
            flatten_layer = FlattenLayer()
            fc_layer = SpectralFullyConnectedLayer(main_channels * (input_res ** 2), main_channels,
                activation=self.activation, bias_act_impl=self.bias_act_impl)
            layers += [conv_layer, flatten_layer, fc_layer]
            names += ['conv', 'flatten', 'fc']
        else:
            assert False, f'{self.__class__.__name__}: main block architecture={self.architecture} is not supported'
        return torch.nn.Sequential(OrderedDict(zip(names, layers)))

    def forward(self, x, img, c, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if DEBUG_MODE:
            print(f'Head res={self.resolution}: x_input.shape={x.shape}')

        # Input.
        # Note: for src blocks res is referred to input res, output res is reduced by 2
        misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
        x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        x = self.main(x)
        if self.architecture in ['sg2', 'swg-v2']:
            # FC layer is the last layer in main block for architectures above
            assert x.ndim == 2
            x = x.unsqueeze(2).unsqueeze(3)
        if DEBUG_MODE:
            print(f'Head res={self.resolution}: x_main.shape={x.shape}')

        # Final layers.
        dtype = torch.float32
        memory_format = torch.contiguous_format
        x = x.to(dtype=dtype, memory_format=memory_format)
        x = self.cls(x)
        if DEBUG_MODE:
            print(f'Head res={self.resolution}: x_cls.shape={x.shape}')
        if self.c_dim > 0:
            c_mapped = self.c_mapper(c).unsquueze(-1)
            x = (x * c_mapped).sum(1, keepdims=True) * self.c_map_gain

        assert x.dtype == dtype
        return x

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'
