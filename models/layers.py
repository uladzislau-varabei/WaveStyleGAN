import numpy as np
import torch
from torch.nn.utils.spectral_norm import SpectralNorm
from torch.nn.utils.parametrizations import _SpectralNorm

from models.misc import get_num_params_and_buffers_message
from models.ops import bias_act, conv2d_gradfix, conv2d_resample, fma, upfirdn2d
from models.utils import get_activation, get_normalization
from models import misc, utils
from shared_utils import check_equal_lists, is_auto_option, NCHW_DATA_FORMAT, USE_NEW_BIAS_ACT, USE_LEGACY_BEHAVIOUR
from wavelets import WAVELETS_DICT, FW_KEY, BW_KEY, KERNEL_KEY
from wavelets.utils import DEFAULT_SCALE_1D_COEFFS, DEFAULT_SCALE_2D_COEFFS, COEFFS_SCALES_V, NCHW_FORMAT, \
    get_default_coeffs_scales_2d, extract_coeffs_from_channels, LAYER_COEFFS_SCALES


# ----- Modulated convolution -----

class ModulatedConv2d(torch.nn.Module):
    # Thanks to https://github.com/TencentARC/GFPGAN/issues/184 by Zwei-Rakete
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_style_feat,
                 demodulate=True,
                 sample_mode=None,
                 eps=1e-8):
        super(ModulatedConv2d, self).__init__()
        import math
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.sample_mode = sample_mode
        self.eps = eps

        # modulation inside each modulated conv
        self.modulation = torch.nn.Linear(num_style_feat, in_channels, bias=True)
        # initialization
        # default_init_weights(self.modulation, scale=1, bias_fill=1, a=0, mode='fan_in', nonlinearity='linear')

        self.weight = torch.nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size) /
            math.sqrt(in_channels * kernel_size**2))
        self.padding = kernel_size // 2
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=self.padding, bias=False)
        self.conv2d.weight.data = self.weight.view(1 * self.out_channels, self.in_channels,self.kernel_size,self.kernel_size)

    def forward(self, x, style):
        b, c, h, w = x.shape  # c = c_in
        # weight modulation
        style = self.modulation(style).view(b, c, 1, 1)
        x = x * style
        if self.demodulate:
            if self.sample_mode == 'upsample':
                x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            elif self.sample_mode == 'downsample':
                x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
            x = self.conv2d(x)
            weight = self.weight * style
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            out = x * demod.view(b, self.out_channels, 1, 1)
        else:
            if self.sample_mode == 'upsample':
                x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            elif self.sample_mode == 'downsample':
                x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
            x = x.view(1, b * c, h, w)
            out = self.conv2d(x)

        out = out.view(b, self.out_channels, *out.shape[2:4])

        return out


@misc.profiled_function
def modulated_conv2d_v1(
    x,                              # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                         # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                         # Modulation coefficients of shape [batch_size, in_channels].
    noise                = None,    # Optional noise tensor to add to the output activations.
    up                   = 1,       # Integer upsampling factor.
    down                 = 1,       # Integer downsampling factor.
    padding              = 0,       # Padding with respect to the upsampled image.
    input_gain           = None,    # Optional scale factors for the input channels: [], [in_channels], or [batch_size, in_channels]
    resample_filter      = None,    # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate           = True,    # Apply weight demodulation?
    flip_weight          = True,    # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv        = True,    # Perform modulation, convolution, and demodulation as a single fused operation?
    use_custom_conv2d_op = True,    # Enable conv2d_gradfix (default: True)?
    upfirdn2d_impl       = 'cuda'   # Implementation of upfirdn2d. One of ['cuda', 'ref', 'custom_grad'].
):
    # Used in StyleGAN2: conv + optional upsampling/downsampling
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # Apply input scaling.
    if input_gain is not None:
        input_gain = input_gain.expand(batch_size, in_channels) # [NI]
        w = w * input_gain.unsqueeze(1).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down,
            padding=padding, flip_weight=flip_weight, use_custom_conv2d_op=use_custom_conv2d_op,
            upfirdn2d_impl=upfirdn2d_impl)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x + noise.to(x.dtype) # fix backward error for custom grad
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding,
        groups=batch_size, flip_weight=flip_weight, use_custom_conv2d_op=use_custom_conv2d_op,
        upfirdn2d_impl=upfirdn2d_impl)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        # x = x.add_(noise)
        x = x + noise.to(dtype=x.dtype) # fix backward error for custom grad
    return x


@misc.profiled_function
def modulated_conv2d_v2(
    x,                           # Input tensor: [batch_size, in_channels, in_height, in_width]
    w,                           # Weight tensor: [out_channels, in_channels, kernel_height, kernel_width]
    s,                           # Style tensor: [batch_size, in_channels]
    demodulate           = True, # Apply weight demodulation?
    padding              = 0,    # Padding: int or [padH, padW]
    input_gain           = None, # Optional scale factors for the input channels: [], [in_channels], or [batch_size, in_channels]
    use_custom_conv2d_op = True, # Enable conv2d_gradfix (default: True)?
):
    # Used in StyleGAN3: only conv, no upsampling/downsampling
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(x.shape[0])
    out_channels, in_channels, kh, kw = w.shape
    misc.assert_shape(w, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(s, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs.
    if demodulate:
        w = w * w.square().mean([1, 2, 3], keepdim=True).rsqrt()
        s = s * s.square().mean().rsqrt()

    # Modulate weights.
    w = w.unsqueeze(0) # [NOIkk]
    w = w * s.unsqueeze(1).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Demodulate weights.
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt() # [NO]
        w = w * dcoefs.unsqueeze(2).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Apply input scaling.
    if input_gain is not None:
        input_gain = input_gain.expand(batch_size, in_channels) # [NI]
        w = w * input_gain.unsqueeze(1).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Execute as one fused op using grouped convolution.
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_gradfix.conv2d(input=x, weight=w.to(x.dtype), padding=padding, groups=batch_size,
        use_custom_op=use_custom_conv2d_op)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x


# ----- General layers -----

class BiasActivationLayer(torch.nn.Module):
    def __init__(self, activation_name, dim=1, gain=None, clamp=None, gain_mult=None):
        super().__init__()
        self.activation_name = activation_name
        activation_data = utils.get_activation(activation_name)
        self.activation = activation_data[0]
        self.def_gain = activation_data[1]
        self.dim = dim
        gain = float(gain) if gain is not None else self.def_gain
        clamp = float(clamp) if clamp is not None else -1
        if gain_mult is not None:
            # For ResNet connections additional gain is provided
            assert gain_mult > 0, f'gain_mult must be > 0, received {gain_mult}'
            gain = gain_mult * gain
            clamp = gain_mult * clamp if clamp is not None else clamp
        self.gain = gain
        self.clamp = clamp
        self.gain_mult = gain_mult

    def forward(self, x, b):
        # 1. Add bias
        if b is not None:
            assert isinstance(b, torch.Tensor) and b.ndim == 1
            assert 0 <= self.dim < x.ndim
            assert b.shape[0] == x.shape[self.dim]
            x = x + b.reshape([-1 if i == self.dim else 1 for i in range(x.ndim)])
        # 2. Evaluate activation function
        x = self.activation(x)
        # 3. Scale by gain
        if self.gain != 1:
            x = x * self.gain
        # 4. Clamp
        if self.clamp >= 0:
            x = x.clamp(-self.clamp, +self.clamp)
        return x


class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        bias            = True,     # Apply additive bias before the activation function?
        lr_multiplier   = 1,        # Learning rate multiplier.
        weight_init     = 1,        # Initial standard deviation of the weight tensor.
        bias_init       = 0,        # Initial value of the additive bias.
        bias_act_impl   = 'cuda',   # Bias with activation implementation to use: 'ref', 'cuda'.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) * (weight_init / lr_multiplier))
        bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [out_features])
        self.bias = torch.nn.Parameter(torch.from_numpy(bias_init / lr_multiplier)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier
        if USE_NEW_BIAS_ACT:
            self.bias_act_impl = bias_act_impl
        else:
            self.bias_act = BiasActivationLayer(self.activation)

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain
        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            if USE_NEW_BIAS_ACT:
                x = bias_act.bias_act(x, b, act=self.activation, impl=self.bias_act_impl)
            else:
                # Note: replaced original custom bias_act op
                x = self.bias_act(x, b)
        return x

    def extra_repr(self):
        return ' '.join([
            f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s},',
            f'{get_num_params_and_buffers_message(self)}'])


class SpectralFullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,              # Number of input features.
        out_features,             # Number of output features.
        activation    = 'linear', # Activation function: 'relu', 'lrelu', etc.
        bias          = True,     # Apply additive bias before the activation function?
        lr_multiplier = 1,        # Learning rate multiplier.
        weight_init   = 1,        # Initial standard deviation of the weight tensor.
        bias_init     = 0,        # Initial value of the additive bias.
        bias_act_impl = 'cuda',   # Bias with activation implementation to use: 'ref', 'cuda'.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        # Note: check if fan_in or fan_out should be used
        features_scale = 1 / np.sqrt(in_features)  # LeCun normalization
        weight = torch.randn([out_features, in_features], dtype=torch.float32) * weight_init * features_scale
        self.weight = torch.nn.Parameter(weight)
        bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [out_features])
        self.bias = torch.nn.Parameter(torch.from_numpy(bias_init / lr_multiplier)) if bias else None
        self.weight_gain = lr_multiplier
        self.bias_gain = lr_multiplier
        if USE_NEW_BIAS_ACT:
            self.bias_act_impl = bias_act_impl
        else:
            self.bias_act = BiasActivationLayer(self.activation)
        self.spectral_norm = _SpectralNorm(self.weight, n_power_iterations=1, dim=0, eps=1e-12)

    def forward(self, x):
        # Note: perform normalization in fp32 and after that cast to x dtype
        w = self.spectral_norm(self.weight).to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain
        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            if USE_NEW_BIAS_ACT:
                x = bias_act.bias_act(x, b, act=self.activation, impl=self.bias_act_impl)
            else:
                # Note: replaced original custom bias_act op
                x = self.bias_act(x, b)
        return x

    def extra_repr(self):
        return ' '.join([
            f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s},',
            f'{get_num_params_and_buffers_message(self)}'])


class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.flatten(1)


class ResidualBlock(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return (self.fn(x) + x) / np.sqrt(2)


class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                            # Number of input channels.
        out_channels,                           # Number of output channels.
        kernel_size,                            # Width and height of the convolution kernel.
        bias                 = True,            # Apply additive bias before the activation function?
        activation           = 'linear',        # Activation function: 'relu', 'lrelu', etc.
        up                   = 1,               # Integer upsampling factor.
        down                 = 1,               # Integer downsampling factor.
        resample_filter      = [1, 3, 3, 1],    # Low-pass filter to apply when resampling activations.
        conv_clamp           = None,            # Clamp the output to +-X, None = disable clamping.
        gain_mult            = 1,               # Additional gain factor to multiply by existing gain and clamp.
        channels_last        = False,           # Expect the input to have memory_format=channels_last?
        use_custom_conv2d_op = True,            # Enable conv2d_gradfix (default: True)?
        upfirdn2d_impl       = 'cuda',          # Implementation of upfirdn2d. One of ['cuda', 'ref', 'custom_grad'].
        bias_act_impl        = 'cuda',          # Implementation of bias with activation. One of ['cuda', 'ref'].
        trainable            = True,            # Update the weights of this layer during training?
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.upfirdn2d_impl = upfirdn2d_impl
        self.use_custom_conv2d_op = use_custom_conv2d_op
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.activation = activation
        self.use_custom_conv2d_op = use_custom_conv2d_op
        self.upfirdn2d_impl = upfirdn2d_impl
        if USE_NEW_BIAS_ACT:
            self.act_gain = bias_act.activation_funcs[activation]['def_gain']
            self.bias_act_impl = bias_act_impl
        else:
            self.gain_mult = gain_mult
            self.bias_act = BiasActivationLayer(self.activation, gain_mult=self.gain_mult)

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1) # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down,
            padding=self.padding, flip_weight=flip_weight, use_custom_conv2d_op=self.use_custom_conv2d_op,
            upfirdn2d_impl=self.upfirdn2d_impl)
        if USE_NEW_BIAS_ACT:
            act_gain = self.act_gain * gain
            act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
            x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp, impl=self.bias_act_impl)
        else:
            # Note: gain and clamp are adjusted in __init__
            x = self.bias_act(x, b)
        return x

    def extra_repr(self):
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, kernel_size={self.kernel_size},',
            f'activation={self.activation:s}, up={self.up}, down={self.down},',
            f'{get_num_params_and_buffers_message(self)}'])


class SMConv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                            # Number of input channels.
        out_channels,                           # Number of output channels.
        kernel_size,                            # Width and height of the convolution kernel.
        bias                 = True,            # Apply additive bias before the activation function?
        activation           = 'linear',        # Activation function: 'relu', 'lrelu', etc.
        up                   = 1,               # Integer upsampling factor.
        down                 = 1,               # Integer downsampling factor.
        resample_filter      = [1, 3, 3, 1],    # Low-pass filter to apply when resampling activations.
        conv_clamp           = None,            # Clamp the output to +-X, None = disable clamping.
        gain_mult            = 1,               # Additional gain factor to multiply by existing gain and clamp.
        channels_last        = False,           # Expect the input to have memory_format=channels_last?
        use_custom_conv2d_op = True,            # Enable conv2d_gradfix (default: True)?
        upfirdn2d_impl       = 'cuda',          # Implementation of upfirdn2d. One of ['cuda', 'ref', 'custom_grad'].
        bias_act_impl        = 'cuda',          # Implementation of bias with activation. One of ['cuda', 'ref'].
        trainable            = True,            # Update the weights of this layer during training?
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.activation = activation
        self.use_custom_conv2d_op = use_custom_conv2d_op
        self.upfirdn2d_impl = upfirdn2d_impl
        if USE_NEW_BIAS_ACT:
            self.act_gain = bias_act.activation_funcs[activation]['def_gain']
            self.bias_act_impl = bias_act_impl
        else:
            self.gain_mult = gain_mult
            self.bias_act = BiasActivationLayer(self.activation, gain_mult=self.gain_mult)

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

        self.gain = torch.nn.Parameter(torch.ones(1))
        self.scales = torch.nn.Parameter(torch.ones(in_channels))

        self.use_v2_modconv_impl = False
        # Note: different forward() implementations must be called from forward() itself.
        # No forward modification in __init__ as it breaks nn.DataParallel()

    def upsample2d(self, x):
        return upfirdn2d.upsample2d(x, self.resample_filter, up=self.up, impl=self.upfirdn2d_impl,
            use_custom_conv2d_op=self.use_custom_conv2d_op)

    def downsample2d(self, x):
        return upfirdn2d.downsample2d(x, self.resample_filter, down=self.down, impl=self.upfirdn2d_impl,
            use_custom_conv2d_op=self.use_custom_conv2d_op)

    def forward_modconv_v2(self, x, gain=1):
        scales = self.scales.expand(x.shape[0], -1).to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        # conv2d_resample:
        # 1) upsample2d => conv2d => downsample2d (reference),
        # 2) conv_transpose_2d => upfirdn2d (upsampling)
        # 3) downsample2d => conv2d (downsampling)
        # Note: base Discriminator only uses downsampling if needed
        # Note: projected Discriminator uses upsampling (CSM) and downsampling (DownBlock)
        if self.down > 1:
            x = self.downsample2d(x)
        # Note: with enabled demodulation w is scaled by L2 norm with dims=[2, 3, 4] before conv op,
        # s is scaled by weight_gain for toRGB layers but then demodulation is disabled
        x = modulated_conv2d_v2(x, w=self.weight, s=scales, demodulate=True, padding=self.padding,
            input_gain=self.gain, use_custom_conv2d_op=self.use_custom_conv2d_op)
        if self.up > 1:
            x = self.upsample2d(x)
        if USE_NEW_BIAS_ACT:
            act_gain = self.act_gain * gain
            act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
            x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp, impl=self.bias_act_impl)
        else:
            # Note: gain and clamp are adjusted in __init__
            x = self.bias_act(x, b)
        return x

    def forward_modconv_v1(self, x, gain=1):
        # Note: the implementation uses modulated_conv2d_v1, so up and down factors can be passed to conv op
        scales = self.scales.expand(x.shape[0], -1).to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1)  # slightly faster
        # Note: scales are scaled by weights_gain only for toRGB layer,
        # fp16 + demodulate => w is scaled by weight_gain before other ops
        x = modulated_conv2d_v1(x, weight=self.weight, styles=scales, up=self.up, down=self.down, padding=self.padding,
            input_gain=self.gain, resample_filter=self.resample_filter, demodulate=True, flip_weight=flip_weight,
            fused_modconv=True, use_custom_conv2d_op=self.use_custom_conv2d_op, upfirdn2d_impl=self.upfirdn2d_impl)
        if USE_NEW_BIAS_ACT:
            act_gain = self.act_gain * gain
            act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
            x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp, impl=self.bias_act_impl)
        else:
            # Note: gain and clamp are adjusted in __init__
            x = self.bias_act(x, b)
        return x

    def forward(self, x, gain=1):
        # Note: only here different forward versions can be used.
        # No modification in __init__ as it breaks nn.DataParallel
        return self.forward_modconv_v2(x, gain) if self.use_v2_modconv_impl else self.forward_modconv_v1(x, gain)

    def extra_repr(self):
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, kernel_size={self.kernel_size},',
            f'activation={self.activation:s}, up={self.up}, down={self.down},',
            f'{get_num_params_and_buffers_message(self)}'])


class SpectralConv2d(torch.nn.Conv2d):
    # TODO: update implementation to be consistent with other conv layers
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Note: dim refers to dimension with output channels
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)

    def extra_repr(self):
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, kernel_size={self.kernel_size},',
            f'{get_num_params_and_buffers_message(self)}'])


class WaveletLayer(torch.nn.Module):
    def __init__(self,
        wavelet,
        use_affine               = True,
        init_affine_scales       = 'AUTO',
        train_affine             = True,
        affine_lr_multiplier     = 1,
        train_kernel             = False,
        scale_1d_coeffs          = 'AUTO',
        scale_2d_coeffs          = 'AUTO',
        coeffs_scales_2d_version = 'AUTO',
        extract_and_fuse_coeffs  = False, # Extract wavelet decomposition coeffs and merge high frequency data?
        always_forward           = False, # Always apply forward transform?
        always_inverse           = False, # Always apply inverse transform?
    ):
        super().__init__()
        self.init_affine_transform(use_affine, init_affine_scales, train_affine, affine_lr_multiplier)
        self.wavelet = wavelet
        self.train_kernel = train_kernel
        if is_auto_option(scale_1d_coeffs):
            scale_1d_coeffs = DEFAULT_SCALE_1D_COEFFS
        self.scale_1d_coeffs = scale_1d_coeffs
        self.init_scales_2d(scale_2d_coeffs, coeffs_scales_2d_version)
        wavelet_data = WAVELETS_DICT[self.wavelet.lower()]
        self.wavelet_fw = wavelet_data[FW_KEY]
        self.wavelet_bw = wavelet_data[BW_KEY]
        kernel = torch.tensor(wavelet_data[KERNEL_KEY], dtype=torch.float32)
        # Note: maybe can just use Parameter wit requires_grad setting
        # but previously during distributed training gradient was computed for some kernels
        if self.train_kernel:
            self.kernel = torch.nn.Parameter(kernel)
            assert False, 'Training of wavelet kernel is disabled for now'
        else:
            self.register_buffer('kernel', kernel)
        self.extract_and_fuse_coeffs = extract_and_fuse_coeffs
        assert not (always_forward and always_inverse)
        self.always_forward = always_forward
        self.always_inverse = always_inverse

    def init_affine_transform(self, use_affine, init_affine_scales, train_affine, affine_lr_multiplier):
        self.use_affine = use_affine
        self.train_affine = train_affine
        if self.train_affine:
            assert affine_lr_multiplier > 0
        else:
            affine_lr_multiplier = 1
        self.affine_lr_multiplier = affine_lr_multiplier
        affine_scales = LAYER_COEFFS_SCALES if is_auto_option(init_affine_scales) else init_affine_scales
        if self.use_affine:
            assert len(affine_scales) == 4
            # Divide by these value when initialized and then multiply by it at every step
            affine_scales = torch.tensor(affine_scales, dtype=torch.float32) / self.affine_lr_multiplier
        if self.train_affine:
            self.affine_scales = torch.nn.Parameter(affine_scales)
            if USE_LEGACY_BEHAVIOUR:
                self.affine_shifts = torch.nn.Parameter(torch.tensor([0, 0, 0, 0], dtype=torch.float32))
            else:
                self.affine_shifts = None
        else:
            if use_affine:
                self.register_buffer('affine_scales', affine_scales)
            else:
                self.affine_scales = None
            self.affine_shifts = None

    def init_scales_2d(self, scale_2d_coeffs, coeffs_scales_2d_version):
        if self.use_affine:
            self.scale_2d_coeffs = False
            self.coeffs_scales_2d_version = -1
            self.coeffs_scales_2d = None
        else:
            if is_auto_option(scale_2d_coeffs):
                scale_2d_coeffs = DEFAULT_SCALE_2D_COEFFS
            self.scale_2d_coeffs = scale_2d_coeffs
            if is_auto_option(coeffs_scales_2d_version):
                coeffs_scales_2d_version = COEFFS_SCALES_V
            self.coeffs_scales_2d_version = coeffs_scales_2d_version
            self.coeffs_scales_2d = get_default_coeffs_scales_2d(self.coeffs_scales_2d_version)

    def broadcast_params(self, x, C):
        # To be called with scales and shifts
        assert x.ndim == 1 and x.numel() == 4 and C % 4 == 0
        x = x[None, :].permute(1, 0).repeat(1, C // 4).reshape(1, C)
        return self.affine_lr_multiplier * x[:, :, None, None]

    def dwt(self, x):
        x = self.wavelet_fw(x, kernel=self.kernel.to(dtype=x.dtype), scale_1d_coeffs=self.scale_1d_coeffs,
            scale_2d_coeffs=self.scale_2d_coeffs, coeffs_scales_2d=self.coeffs_scales_2d,
            data_format=NCHW_DATA_FORMAT)
        if self.use_affine:
            # Without explicit channels extraction and merging
            C = x.size(1)
            scales = self.broadcast_params(self.affine_scales, C)
            # For DWT use regular values
            x = x * scales
            if self.train_affine and USE_LEGACY_BEHAVIOUR:
                shift = self.broadcast_params(self.affine_shifts, C)
                x = x + shift
        return x

    def idwt(self, x):
        if self.use_affine:
            # Without explicit channels extraction and merging
            C = x.size(1)
            scales = self.broadcast_params(self.affine_scales, C)
            # For IDWT use inverse values
            x = x / scales
            if self.train_affine and USE_LEGACY_BEHAVIOUR:
                shift = self.broadcast_params(self.affine_shifts, C)
                x = x + shift
        x = self.wavelet_bw(x, kernel=self.kernel.to(dtype=x.dtype), scale_1d_coeffs=self.scale_1d_coeffs,
            scale_2d_coeffs=self.scale_2d_coeffs, coeffs_scales_2d=self.coeffs_scales_2d,
            data_format=NCHW_DATA_FORMAT)
        return x

    def forward(self, x, inverse=False):
        if inverse or self.always_inverse:
            assert not self.always_forward, f"{self.__class__.__name__}: inverse=False, but always_forward=True"
            x = self.idwt(x)
        else:
            x = self.dwt(x)
        if self.extract_and_fuse_coeffs:
            x_LL, x_LH, x_HL, x_HH = extract_coeffs_from_channels(x, data_format=NCHW_DATA_FORMAT)
            x = torch.cat([x_LL, x_LH + x_HL + x_HH], dim=1)
        return x

    def extra_repr(self):
        return ' '.join([
            f'wavelet={self.wavelet:s}, use_affine={self.use_affine},'
            f'train_affine={self.train_affine}, train_kernel={self.train_kernel},',
            f'coeffs_scale_2d_version={self.coeffs_scales_2d_version},',
            f'extract_and_fuse_coeffs={self.extract_and_fuse_coeffs},',
            f'{get_num_params_and_buffers_message(self)}'])


class WaveletUpFIRDn2dLayer(torch.nn.Module):
    def __init__(self,
        wavelet,
        train_kernel,
        scale_1d_coeffs,
        scale_2d_coeffs,
        coeffs_scales_2d_version,
        up                   = False,           # Upsample input?
        down                 = False,           # Downsample input?
        resample_filter      = [1, 3, 3, 1],    # # Low-pass filter to apply when resampling activations.
        use_fp16             = False,           # Use FP16 for this layer?
        use_custom_conv2d_op = True,            # Enable conv2d_gradfix (default: True)?
        upfirdn2d_impl       = 'cuda',          # Implementation of upfirdn2d op. One of ['ref', 'cuda', 'custom_grad'] (default: 'ref').
    ):
        super().__init__()
        self.wavelet = wavelet
        self.train_kernel = train_kernel
        if is_auto_option(scale_1d_coeffs):
            scale_1d_coeffs = DEFAULT_SCALE_1D_COEFFS
        self.scale_1d_coeffs = scale_1d_coeffs
        if is_auto_option(scale_2d_coeffs):
            scale_2d_coeffs = DEFAULT_SCALE_2D_COEFFS
        self.scale_2d_coeffs = scale_2d_coeffs
        if is_auto_option(coeffs_scales_2d_version):
            coeffs_scales_2d_version = COEFFS_SCALES_V
        self.coeffs_scales_2d_version = coeffs_scales_2d_version
        self.coeffs_scales_2d = get_default_coeffs_scales_2d(self.coeffs_scales_2d_version)
        wavelet_data = WAVELETS_DICT[self.wavelet.lower()]
        self.wavelet_fw = wavelet_data[FW_KEY]
        self.wavelet_bw = wavelet_data[BW_KEY]
        # Avoid errors with distributed training: use fp32 for parameter
        kernel = torch.tensor(wavelet_data[KERNEL_KEY], dtype=torch.float32)
        # Note: maybe can just use Parameter wit requires_grad setting
        # but previously during distributed training gradient was computed for some kernels
        if self.train_kernel:
            self.kernel = torch.nn.Parameter(kernel)
        else:
            self.register_buffer('kernel', kernel)
        assert up or down and not (up and down), 'Only up or down must be set to True'
        self.up = up
        self.down = down
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.use_custom_conv2d_op = use_custom_conv2d_op
        self.upfirdn2d_impl = upfirdn2d_impl

    def dwt(self, x):
        return self.wavelet_fw(x, kernel=self.kernel.to(dtype=x.dtype), scale_1d_coeffs=self.scale_1d_coeffs,
            scale_2d_coeffs=self.scale_2d_coeffs, coeffs_scales_2d=self.coeffs_scales_2d,
            data_format=NCHW_DATA_FORMAT)

    def idwt(self, x):
        return self.wavelet_bw(x, kernel=self.kernel.to(dtype=x.dtype), scale_1d_coeffs=self.scale_1d_coeffs,
            scale_2d_coeffs=self.scale_2d_coeffs, coeffs_scales_2d=self.coeffs_scales_2d,
            data_format=NCHW_DATA_FORMAT)

    def forward(self, x):
        x = self.idwt(x)
        if self.up:
            x = upfirdn2d.upsample2d(x, self.resample_filter, up=2, impl=self.upfirdn2d_impl,
                    use_custom_conv2d_op=self.use_custom_conv2d_op)
        else:
            x = upfirdn2d.downsample2d(x, self.resample_filter, down=2, impl=self.upfirdn2d_impl,
                    use_custom_conv2d_op=self.use_custom_conv2d_op)
        x = self.dwt(x)
        return x

    def extra_repr(self):
        return ' '.join([
            f'wavelet={self.wavelet:s}, train_kernel={self.train_kernel},',
            f'coeffs_scale_2d_version={self.coeffs_scales_2d_version},',
            f'up={self.up:d}, down={self.down:d},',
            f'{get_num_params_and_buffers_message(self)}'])


# ----- Generator layers -----

class FourierSynthesisInput(torch.nn.Module):
    # Used in StyleGAN3 as input features
    def __init__(self,
        w_dim,                  # Intermediate latent (W) dimensionality.
        channels,               # Number of output channels.
        size,                   # Output spatial size: int or [width, height].
        sampling_rate,          # Output sampling rate.
        bandwidth,              # Output bandwidth.
        bias_act_impl = 'cuda', # Implementation of bias with activation. One of ['cuda', 'ref'].
    ):
        super().__init__()
        self.w_dim = w_dim
        self.channels = channels
        self.size = np.broadcast_to(np.asarray(size), [2])
        self.sampling_rate = sampling_rate
        self.bandwidth = bandwidth

        # Draw random frequencies from uniform 2D disc.
        freqs = torch.randn([self.channels, 2])
        radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
        freqs /= radii * radii.square().exp().pow(0.25)
        freqs *= bandwidth
        phases = torch.rand([self.channels]) - 0.5

        # Setup parameters and buffers.
        self.weight = torch.nn.Parameter(torch.randn([self.channels, self.channels]))
        self.affine = FullyConnectedLayer(w_dim, 4, weight_init=0, bias_init=[1, 0, 0, 0], bias_act_impl=bias_act_impl)
        self.register_buffer('transform', torch.eye(3, 3)) # User-specified inverse transform wrt. resulting image.
        self.register_buffer('freqs', freqs)
        self.register_buffer('phases', phases)

    def forward(self, w):
        # Introduce batch dimension.
        transforms = self.transform.unsqueeze(0) # [batch, row, col]
        freqs = self.freqs.unsqueeze(0) # [batch, channel, xy]
        phases = self.phases.unsqueeze(0) # [batch, channel]

        # Apply learned transformation.
        t = self.affine(w) # t = (r_c, r_s, t_x, t_y)
        t = t / t[:, :2].norm(dim=1, keepdim=True) # t' = (r'_c, r'_s, t'_x, t'_y)
        m_r = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse rotation wrt. resulting image.
        m_r[:, 0, 0] = t[:, 0]  # r'_c
        m_r[:, 0, 1] = -t[:, 1] # r'_s
        m_r[:, 1, 0] = t[:, 1]  # r'_s
        m_r[:, 1, 1] = t[:, 0]  # r'_c
        m_t = torch.eye(3, device=w.device).unsqueeze(0).repeat([w.shape[0], 1, 1]) # Inverse translation wrt. resulting image.
        m_t[:, 0, 2] = -t[:, 2] # t'_x
        m_t[:, 1, 2] = -t[:, 3] # t'_y
        transforms = m_r @ m_t @ transforms # First rotate resulting image, then translate, and finally apply user-specified transform.

        # Transform frequencies.
        phases = phases + (freqs @ transforms[:, :2, 2:]).squeeze(2)
        freqs = freqs @ transforms[:, :2, :2]

        # Dampen out-of-band frequencies that may occur due to the user-specified transform.
        # print(f'freqs={freqs.norm(dim=2)}, bandwidth={self.bandwidth}, sampling_rate={self.sampling_rate}')
        amplitudes = (1 - (freqs.norm(dim=2) - self.bandwidth) / (self.sampling_rate / 2 - self.bandwidth)).clamp(0, 1)
        # print(f'amplitudes={amplitudes}')

        # Construct sampling grid.
        theta = torch.eye(2, 3, device=w.device)
        theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate
        theta[1, 1] = 0.5 * self.size[1] / self.sampling_rate
        grids = torch.nn.functional.affine_grid(theta.unsqueeze(0), [1, 1, self.size[1], self.size[0]], align_corners=False)

        # Compute Fourier features.
        x = (grids.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)).squeeze(3) # [batch, height, width, channel]
        x = x + phases.unsqueeze(1).unsqueeze(2)
        x = torch.sin(x * (np.pi * 2))
        x = x * amplitudes.unsqueeze(1).unsqueeze(2)

        # Apply trainable mapping.
        weight = self.weight / np.sqrt(self.channels)
        x = x @ weight.t()

        # Ensure correct shape.
        x = x.permute(0, 3, 1, 2) # [batch, channel, height, width]
        misc.assert_shape(x, [w.shape[0], self.channels, int(self.size[1]), int(self.size[0])])
        return x

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, channels={self.channels:d}, size={list(self.size)},',
            f'sampling_rate={self.sampling_rate:g}, bandwidth={self.bandwidth:g},',
            f'{get_num_params_and_buffers_message(self)}'])


class ToRGBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None,
        channels_last=False, use_custom_conv2d_op=True, upfirdn2d_impl='cuda', bias_act_impl='cuda'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size])
        self.weight = torch.nn.Parameter(weight).to(memory_format=memory_format)
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.activation = 'linear'
        self.upfirdn2d_impl = upfirdn2d_impl
        self.use_custom_conv2d_op = use_custom_conv2d_op
        if USE_NEW_BIAS_ACT:
            self.bias_act_impl = bias_act_impl
        else:
            self.bias_act = BiasActivationLayer(self.activation, clamp=self.conv_clamp)

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d_v1(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv,
            use_custom_conv2d_op=self.use_custom_conv2d_op, upfirdn2d_impl=self.upfirdn2d_impl)
        if USE_NEW_BIAS_ACT:
            x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp, impl=self.bias_act_impl)
        else:
            x = self.bias_act(x, self.bias.to(x.dtype))
        return x

    def extra_repr(self):
        return f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d}'


# ----- Discriminator blocks -----

class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2, 3, 4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x

    def extra_repr(self):
        return f'group_size={self.group_size}, num_channels={self.num_channels:d}'


# ----- Fast Fourier convolution -----

def make_fft_conv_block(in_channels, out_channels, kernel_size, down=1, resample_filter=[1, 3, 3, 1],
    conv_type='base', groups=1, conv_clamp=None, conv_norm=None, activation='lrelu',
    use_custom_conv2d_op=True, upfirdn2d_impl='cuda', bias_act_impl='cuda', channels_last=False):
    assert conv_type.lower() in ['base', 'selfmod', 'spectral']
    # 1. Determine some params
    # When normalization is used, it must be applied outside of conv layer, so activation must be a separate layer as well
    if conv_norm is None:
        # In this case conv -> norm -> act, so conv activation is linear
        bias = True
        conv_activation = activation
    else:
        bias = False
        conv_activation = 'linear'
    # 2. Set conv layer
    conv_type = conv_type.lower()
    conv_params = dict(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
        bias=bias, activation=conv_activation, down=down, resample_filter=resample_filter, conv_clamp=conv_clamp,
        use_custom_conv2d_op=use_custom_conv2d_op, upfirdn2d_impl=upfirdn2d_impl, bias_act_impl=bias_act_impl,
        channels_last=channels_last)
    if conv_type == 'selfmod':
        # Note: normalization can be applied even here
        conv_layer = SMConv2dLayer(**conv_params)
    elif conv_type == 'spectral':
        assert False, 'Spectral conv is not implemented for FFC conv block'
    else:
        conv_layer = Conv2dLayer(**conv_params)
    # 3. Set other layers
    act_layer, _ = get_activation(activation, inplace=True)
    if conv_norm is None:
        conv_layers = [conv_layer]
    else:
        norm_layer = get_normalization(conv_norm, out_channels, groups=None, virtual_bs=None)
        conv_layers = [conv_layer, norm_layer, act_layer]
    # 4. Make final block
    conv_block = torch.nn.Sequential(*conv_layers)
    return conv_block


class FourierUnit(torch.nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        fft_norm             = 'full',        # Normalization for FFT2d. One of ['forward', 'backward', 'ortho'].
        conv_type            = 'selfmod',     # Convolution layer type: ['base', 'selfmod', 'spectral'].
        groups               = 1,             # Number of groups for conv op.
        conv_clamp           = None,          # Clamp the output to +-X, None = disable clamping.
        conv_norm            = None,          # Normalization for conv outputs. One of ['batch_norm', 'layer_norm', 'goup_norm'].
        activation           = 'lrelu',       # Activation function: 'relu', 'lrelu', etc.
        use_custom_conv2d_op = True,          # Enable conv2d_gradfix (default: True)?
        bias_act_impl        = 'cuda',        # Implementation of bias with activation. One of ['ref', 'cuda'].
        channels_last        = False,         # Expect the input to have memory_format=channels_last?
    ):
        # Note: for now conv layers only use groups=1
        assert groups == 1
        super(FourierUnit, self).__init__()
        assert fft_norm in ['forward', 'backward', 'ortho', 'full']
        if fft_norm == 'full':
            self.fft_norm = 'forward'
            self.ifft_norm = 'backward'
        else:
            self.fft_norm = fft_norm
            self.ifft_norm = fft_norm
        self.conv_type = conv_type
        self.conv_norm = conv_norm
        self.activation = activation
        self.conv_block = make_fft_conv_block(in_channels=in_channels * 2, out_channels=out_channels * 2,
            kernel_size=1, conv_type=conv_type, groups=groups, conv_clamp=conv_clamp, conv_norm=conv_norm,
            activation=activation, use_custom_conv2d_op=use_custom_conv2d_op, bias_act_impl=bias_act_impl,
            channels_last=channels_last
        )

    def forward(self, x):
        b, c, h, w = x.size()
        orig_dtype = x.dtype
        x = x.to(dtype=torch.float32)

        ffted = torch.fft.rfft2(x, dim=(-2, -1), norm=self.fft_norm) # [b, c, h, w/2+1], dtype=complex64
        ffted_re, ffted_im = ffted.real, ffted.imag
        ffted = torch.stack([ffted_re, ffted_im], dim=4)             # [b, c, h, w/2+1, 2], dtype=float32
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()            # [b, c, 2, h, w/2 + 1]
        ffted = ffted.view((b, -1,) + ffted.size()[3:])              # [b, c * 2, h, w/2 + 1]

        ffted = ffted.to(dtype=orig_dtype)
        ffted = self.conv_block(ffted)  # [b, c*2, h, w/2+1], from now c refers to out channels
        ffted = ffted.to(dtype=torch.float32)

        # [b, c*2, h, w/2+1] -> [b, -1, 2, h, w/2 + 1] -> [b, c, h, w/2 + 1, 2]
        ffted = ffted.view((b, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()                               # [b, c, h, w/2+1, 2]
        ffted = torch.complex(real=ffted[..., 0], imag=ffted[..., 1]) # [b, c, h, w/2 + 1]

        # Provide original signal size as suggested in the documentation
        output = torch.fft.irfft2(ffted, s=(h, w), dim=(-2, -1), norm=self.ifft_norm)
        output = output.to(dtype=orig_dtype)
        return output

    def extra_repr(self):
        return f'fft_norm={self.fft_norm}, ifft_norm={self.ifft_norm}, ' \
               f'conv_type={self.conv_type}, conv_norm={self.conv_norm}, activation={self.activation}'


class SpectralTransform(torch.nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        down                 = 1,                 # Integer downsampling factor.
        resample_filter      = [1, 3, 3, 1],      # Low-pass filter to apply when resampling activations.
        use_lfu              = True,              # Enable local Fourier unit? Default is True
        lfu_mode             = 'wavelet_full',    # Local Fourier unit feature aggregation mode.
                                                  # One of ['base', 'conv','wavelet_lite', 'wavelet_full'].
        fft_norm             = 'full',            # Normalization for FFT2d. One of ['forward', 'backward', 'ortho', 'full'].
        conv_type            = 'selfmod',         # Convolution layer type: ['base', 'selfmod', 'spectral'].
        groups               = 1,                 # Number of groups for conv op.
        conv_clamp           = None,              # Clamp the output to +-X, None = disable clamping.
        conv_norm            = None,              # Normalization for conv outputs. One of ['batch_norm', 'layer_norm', 'goup_norm'].
        activation           = 'lrelu',           # Activation function: 'relu', 'lrelu', etc.
        use_custom_conv2d_op = True,              # Enable conv2d_gradfix (default: True)?
        upfirdn2d_impl       = 'cuda',            # Implementation of upfirdn2d. One of ['cuda', 'ref', 'custom_grad'].
        bias_act_impl        = 'cuda',            # Implementation if bias with activation. One of ['ref', 'cuda'].
        dwt_params           = None,              # Wavelet params.
        channels_last        = False,             # Expect the input to have memory_format=channels_last?
    ):
        # Note: for now conv layers only use groups=1
        super(SpectralTransform, self).__init__()
        self.use_lfu = use_lfu
        assert lfu_mode in ['base', 'conv', 'wavelet_lite', 'wavelet_full'], \
            f'lfu_mode={self.lfu_mode} is not supported'
        self.lfu_mode = lfu_mode
        self.down = down
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.use_custom_conv2d_op = use_custom_conv2d_op
        self.upfirdn2d_impl = upfirdn2d_impl
        self.bias_act_impl = bias_act_impl

        shared_kwargs = {
            'conv_type': conv_type,
            'groups': groups,
            'conv_clamp': conv_clamp,
            'conv_norm': conv_norm,
            'activation': activation,
            'use_custom_conv2d_op': use_custom_conv2d_op,
            'bias_act_impl': bias_act_impl,
            'channels_last': channels_last
        }
        self.conv_block1 = make_fft_conv_block(in_channels=in_channels, out_channels=out_channels // 2,
            kernel_size=1, **shared_kwargs)
        self.fu = FourierUnit(in_channels=out_channels // 2, out_channels=out_channels // 2, fft_norm=fft_norm,
            **shared_kwargs)
        if self.use_lfu:
            n_lfu_blocks = self.init_lfu(out_channels, fft_norm, shared_kwargs, dwt_params)
        else:
            n_lfu_blocks = 0
        # Similar to ResNet blocks. Outputs: x, FU, LFU. Subtract 1 to avoid too aggressive scaling
        self.gain = 1 / np.sqrt(2 + n_lfu_blocks - 1)
        self.conv_block2 = make_fft_conv_block(in_channels=out_channels // 2, out_channels=out_channels,
            kernel_size=1, **shared_kwargs)

    def init_lfu(self, out_channels, fft_norm, shared_kwargs, dwt_params):
        lfu_in_channels = out_channels // 2
        lfu_out_channels = out_channels // 2
        if self.lfu_mode in ['base', 'conv', 'wavelet_lite']:
            if self.lfu_mode == 'conv':
                # Note: think about other changed params
                # Note: order of dicts is important
                lfu_conv_kwargs = {
                    **shared_kwargs,
                    # This dict overrides keys from the previous dict
                    **{
                        'activation': 'linear'
                    }
                }
                assert not check_equal_lists(lfu_conv_kwargs.values(), shared_kwargs.values()), \
                    'lfu_conv must have different kwargs'
                self.lfu_conv = make_fft_conv_block(in_channels=out_channels // 2,
                    out_channels=out_channels // (2 * 4), kernel_size=1, **lfu_conv_kwargs)
            self.lfu = FourierUnit(in_channels=lfu_in_channels, out_channels=lfu_out_channels,
                fft_norm=fft_norm, **shared_kwargs)
            n_lfu_blocks = 1
        elif self.lfu_mode in ['wavelet_full']:
            self.lfu1 = FourierUnit(in_channels=lfu_in_channels, out_channels=lfu_out_channels,
                fft_norm=fft_norm, **shared_kwargs)
            self.lfu2 = FourierUnit(in_channels=lfu_in_channels, out_channels=lfu_out_channels,
                fft_norm=fft_norm, **shared_kwargs)
            n_lfu_blocks = 2
        else:
            assert False, f'lfu_mode={self.lfu_mode} is not supported'
        if self.lfu_mode in ['wavelet_lite', 'wavelet_full']:
            self.wavelet_layer = WaveletLayer(dwt_params['wavelet'], use_affine=dwt_params['use_affine'],
                init_affine_scales=dwt_params['init_affine_scales'], train_affine=dwt_params['train_affine'],
                affine_lr_multiplier=dwt_params['affine_lr_multiplier'], train_kernel=False,
                scale_1d_coeffs=dwt_params['scale_1d_coeffs'], scale_2d_coeffs=dwt_params['scale_2d_coeffs'],
                coeffs_scales_2d_version=dwt_params['coeffs_scales_2d_version'], always_forward=True)
        return n_lfu_blocks

    def downsample2d(self, x):
        return upfirdn2d.downsample2d(x, self.resample_filter, down=self.down, impl=self.upfirdn2d_impl,
            use_custom_conv2d_op=self.use_custom_conv2d_op)

    def run_lfu(self, x):
        # Note: xs is scaled later with overall layer gain
        _, c, h, w = x.shape
        if self.lfu_mode in ['base', 'conv']:
            reduction = 2
            c_reduction = reduction * reduction
            split_s_h = h // reduction
            split_s_w = w // reduction
            if self.lfu_mode == 'base':
                xs = x[:, :c // c_reduction]
            else: # if self.lfu_mode == 'conv':
                xs = self.lfu_conv(x)
            xs = torch.cat(torch.split(xs, split_s_h, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, reduction, reduction).contiguous()
        elif self.lfu_mode in ['wavelet_lite', 'wavelet_full']:
            x = self.wavelet_layer(x, inverse=False)
            x_LL, x_LH, x_HL, x_HH = extract_coeffs_from_channels(x, data_format=NCHW_FORMAT)
            if self.lfu_mode == 'wavelet_lite':
                xs = self.lfu(x_LL)
            else: # if self.lfu_mode == 'wavelet_full':
                xs1 = self.lfu1(x_LL)
                xs2 = self.lfu2(x_LH + x_HL + x_HH)
                xs = xs1 + xs2
            # Spatial dimension is reduced by 2 after wavelets
            xs = xs.repeat(1, 1, 2, 2).contiguous()
        else:
            assert False, f'lfu_mode={self.lfu_mode} is not supported'
        return xs

    def forward(self, x):
        orig_dtype = x.dtype
        # TODO: check if convs should use downsampling inside
        # Note: convs use 1x1 kernels, so apply downsampling before
        if self.down > 1:
            x = self.downsample2d(x)
        x = self.conv_block1(x)
        output = self.fu(x)

        xs = self.run_lfu(x) if self.use_lfu else 0

        output = self.conv_block2(self.gain * (x + output + xs))
        output = output.to(dtype=orig_dtype)
        return output

    def extra_repr(self):
        return f'use_lfu={self.use_lfu}, lfu_mode={self.lfu_mode}, down={self.down}'


class FastFourierConv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        impl_idx,                                 # Integer. One of [1, 2].
        ratio_in_global,                          # Ratio of in channels for global branch.
        ratio_out_global,                         # Ratio of out channels for global branch.
        kernel_size,                              # Width and height of the convolution kernel.
        down                 = 1,                 # Integer downsampling factor.
        resample_filter      = [1, 3, 3, 1],      # Low-pass filter to apply when resampling activations.
        fft_norm             = 'full',            # Normalization for FFT2d. One of ['forward', 'backward', 'ortho', 'full'].
        use_lfu              = True,              # Enable local Fourier unit? Default is True
        lfu_mode             = 'wavelet_full',    # Local Fourier unit feature aggregation mode.
                                                  # One of ['base', 'conv','wavelet_lite', 'wavelet_full'].
        conv_type            = 'selfmod',         # Convolution layer type: ['base', 'selfmod', 'spectral'].
        groups               = 1,                 # Number of groups for conv op.
        conv_clamp           = None,              # Clamp the output to +-X, None = disable clamping.
        conv_norm            = None,              # Normalization for conv outputs. One of ['batch_norm', 'layer_norm', 'group_norm'].
        activation           = 'lrelu',           # Activation function: 'relu', 'lrelu', etc.
        use_custom_conv2d_op = True,              # Enable conv2d_gradfix (default: True)?
        upfirdn2d_impl       = 'cuda',            # Implementation of upfirdn2d. One of ['cuda', 'ref', 'custom_grad'].
        bias_act_impl        = 'cuda',            # Implementation of bias with activation. One of ['ref', 'cuda'].
        dwt_params           = None,              # Wavelet params.
        channels_last        = False,             # Expect the input to have memory_format=channels_last?
    ):
        # Note: the 1st layer must have ratio_in_global = 0 and the last layer ratio_out_global = 0
        super().__init__()
        assert impl_idx in [1, 2]
        assert down in [1, 2]
        assert 0 <= ratio_in_global <= 1
        assert 0 <= ratio_out_global <= 1
        self.ratio_in_global = ratio_in_global
        self.ratio_out_global = ratio_out_global
        in_channels_global = int(in_channels * self.ratio_in_global)
        in_channels_local = in_channels - in_channels_global
        out_channels_global = int(out_channels * self.ratio_out_global)
        out_channels_local = out_channels - out_channels_global

        if impl_idx == 1:
            shared_conv_activation = 'linear'
            shared_conv_norm = None
            spectral_conv_activation = activation
            spectral_conv_norm = conv_norm
        elif impl_idx == 2:
            shared_conv_activation = activation
            shared_conv_norm = conv_norm
            spectral_conv_activation = activation
            # Default normalization for SpectralTransformer
            spectral_conv_norm = 'batch_norm_local'
        else:
            assert False
        shared_conv_kwargs = {
            'down': down,
            'resample_filter': resample_filter,
            'conv_type': conv_type,
            'groups': groups,
            'conv_clamp': conv_clamp,
            'conv_norm': shared_conv_norm,
            'activation': shared_conv_activation,
            'use_custom_conv2d_op': use_custom_conv2d_op,
            'upfirdn2d_impl': upfirdn2d_impl,
            'bias_act_impl': bias_act_impl,
            'channels_last': channels_last
        }
        spectral_conv_kwargs = {
            **shared_conv_kwargs,
            # This dict overrides keys from the previous dict
            **{
                'conv_norm': spectral_conv_norm,
                'activation': spectral_conv_activation
            }
        }

        # 1: local -> local
        if in_channels_local == 0 or out_channels_local == 0:
            self.conv_l2l = torch.nn.Identity()
            self.empty_l2l = True
        else:
            self.conv_l2l = make_fft_conv_block(in_channels=in_channels_local, out_channels=out_channels_local,
                kernel_size=kernel_size, **shared_conv_kwargs)
            self.empty_l2l = False

        # 2: local -> global
        if in_channels_local == 0 or out_channels_global == 0:
            self.conv_l2g = torch.nn.Identity()
            self.empty_l2g = True
        else:
            self.conv_l2g = make_fft_conv_block(in_channels=in_channels_local, out_channels=out_channels_global,
                kernel_size=kernel_size, **shared_conv_kwargs)
            self.empty_l2g = False

        # 3: global -> local
        if in_channels_global == 0 or out_channels_local == 0:
            self.conv_g2l = torch.nn.Identity()
            self.empty_g2l = True
        else:
            self.conv_g2l = make_fft_conv_block(in_channels=in_channels_global, out_channels=out_channels_local,
                kernel_size=kernel_size, **shared_conv_kwargs)
            self.empty_g2l = False

        # 4: global -> global
        if in_channels_global == 0 or out_channels_global == 0:
            self.conv_g2g = torch.nn.Identity()
            self.empty_g2g = True
        else:
            self.conv_g2g = SpectralTransform(in_channels=in_channels_global, out_channels=out_channels_global,
                use_lfu=use_lfu, lfu_mode=lfu_mode, fft_norm=fft_norm, dwt_params=dwt_params, **spectral_conv_kwargs)
            self.empty_g2g = False

        # Set values for extra representation of layer
        self.in_channels_global = in_channels_global
        self.in_channels_local = in_channels_local
        self.out_channels_global = out_channels_global
        self.out_channels_local = out_channels_local

        self.use_output_block = impl_idx == 1
        if self.use_output_block:
            self.output_block_local = self.make_output_block(self.out_channels_local, conv_norm, activation)
            self.output_block_global = self.make_output_block(self.out_channels_global, conv_norm, activation)
            self.gain_local = None
            self.gain_global = None
        else:
            self.output_block_local = None
            self.output_block_global = None
            # Gain similar to ResNet blocks
            self.gain_local = 1 / np.sqrt(2) if (not self.empty_l2l) and (not self.empty_g2l) else 1
            self.gain_global = 1 / np.sqrt(2) if (not self.empty_l2g) and (not self.empty_g2g) else 1

    def make_output_block(self, in_channels, norm, act):
        return torch.nn.Sequential(
            get_normalization(norm, in_channels, groups=None, virtual_bs=None),
            get_activation(act, inplace=True)[0]
        )

    def forward(self, x):
        x_local, x_global = x if type(x) is tuple else (x, 0)
        out_x_local, out_x_global = 0, 0

        if self.ratio_out_global != 1:
            out_x_local = self.conv_l2l(x_local) + self.conv_g2l(x_global)
            if self.use_output_block:
                out_x_local = self.output_block_local(out_x_local)
            else:
                out_x_local = self.gain_local * out_x_local
        if self.ratio_out_global != 0:
            out_x_global = self.conv_l2g(x_local) + self.conv_g2g(x_global)
            if self.use_output_block:
                out_x_global = self.output_block_global(out_x_global)
            else:
                out_x_global = self.gain_global * out_x_global

        return  out_x_local, out_x_global

    def extra_repr(self):
        return ' '.join([
            f'in_channels_local={self.in_channels_local}, in_channels_global={self.in_channels_global}',
            f'out_channels_local={self.out_channels_local}, out_channels_global={self.out_channels_global}',
            f'{get_num_params_and_buffers_message(self)}'])


# ----- Tests -----

def test_other_layers(bs, device):
    ch_in = 128
    x_linear = torch.randn(bs, ch_in).to(device)
    fc_layer = FullyConnectedLayer(ch_in, ch_in * 2, 'lrelu').to(device)
    fc_layer(x_linear)
    print('Other: FullyConnected layer test passed')
    # 5. Test Fourier input features
    ch_in, ch_out = 32, 64
    x_input = torch.randn(bs, ch_in).to(device)
    input_layer = FourierSynthesisInput(ch_in, ch_out, (9, 8), 1, 1).to(device)
    input_layer(x_input)
    print("Other: FourierSynthesisInput layer test passed")


def test_conv_layers(bs, device):
    # 2. Test modulated conv layers
    ch_in, ch_out, ksize = 32, 64, 3
    x_conv = torch.randn(bs, ch_in, 128, 128).to(device)
    w_conv = torch.randn(ch_out, ch_in, ksize, ksize).to(device)
    s_conv = torch.randn(bs, ch_in).to(device)
    up_factor, down_factor = 2, 4
    padding = ksize // 2
    x_conv_out1 = modulated_conv2d_v1(x_conv, w_conv, s_conv,
                                      up=up_factor, down=down_factor, padding=padding)
    print('Conv: modulated_conv_v1 op test passed')
    x_conv_out2 = modulated_conv2d_v2(x_conv, w_conv, s_conv)
    print('Conv: modulated_conv_v2 op test passed')
    # 4. Test Conv2d layer
    conv_layer = Conv2dLayer(ch_in, ch_out, 3).to(device)
    conv_layer(x_conv)
    print('Conv: Conv2d layer test passed')


def test_ffc_fu(x_input, device, conv_type, conv_norm):
    ch_in = x_input.shape[1]
    layer = FourierUnit(ch_in, 2 * ch_in, conv_type=conv_type, conv_norm=conv_norm).to(device)
    layer(x_input)
    print('FFC: Fourier unit test passed')


def test_ffc_st(x_input, device, conv_type, conv_norm, lfu_mode, dwt_params):
    ch_in = x_input.shape[1]
    layer = SpectralTransform(ch_in, 2 * ch_in, down=2, use_lfu=True, lfu_mode=lfu_mode,
        conv_type=conv_type, conv_norm=conv_norm, dwt_params=dwt_params).to(device)
    layer(x_input)
    print('FFC: SpectralTransform test passed')


def test_ffc_main(x_input, device, use_global_input, down, conv_type, conv_norm, lfu_mode, dwt_params):
    ch_in = x_input.shape[1]
    if use_global_input:
        ratio_in_global = 0.75
        ratio_out_global = 0.75
        x_local_input = x_input[:, int(ratio_in_global * ch_in):]
        x_global_input = x_input[:, :int(ratio_in_global * ch_in)]
        x_input = (x_local_input, x_global_input)
        print(f'x_local: {tuple(x_local_input.shape)}, x_global: {tuple(x_global_input.shape)}')
    else:
        ratio_in_global = 0
        ratio_out_global = 0.75
    layer = FastFourierConv2dLayer(ch_in, 2 * ch_in, impl_idx=2,
        ratio_in_global=ratio_in_global, ratio_out_global=ratio_out_global,
        kernel_size=3, conv_type=conv_type, conv_norm=conv_norm,
        down=down, lfu_mode=lfu_mode, dwt_params=dwt_params).to(device)
    output = layer(x_input)
    print('FFC: main block forward test passed')
    loss = (output[0].mean() + output[1].mean())
    loss.backward()
    for idx, (name, param) in enumerate(layer.named_parameters()):
        grad = param.grad
        if grad is not None:
            value = grad.mean()
            shape = tuple(grad.shape)
        else:
            value = None
            shape = None
        print(f'{idx}) FFC main block grad: name={name}, shape={shape}, mean={value:.8f}')
    print('FFC: main block backward test passed')



def test_ffc_layers(bs, device):
    ch_in, y_size, x_size = 48, 128, 128
    x_input = torch.randn(bs, ch_in, y_size, x_size).to(device)
    conv_type = 'selfmod'
    # conv_norm = None if conv_type == 'selfmod' else 'batch_norm'
    conv_norm = 'batch_norm_local'
    test_ffc_fu(x_input, device, conv_type=conv_type, conv_norm=conv_norm)
    lfu_mode = ['base', 'conv', 'wavelet_lite', 'wavelet_full'][3]
    dwt_params = {
        'wavelet': 'cdf-9/7',
        'use_affine': False,
        'init_affine_scales': None,
        'train_affine': False,
        'affine_lr_multiplier': 1,
        'scale_1d_coeffs': True,
        'scale_2d_coeffs': False,
        'coeffs_scales_2d_version': 'AUTO'
    }
    test_ffc_st(x_input, device, conv_type=conv_type, conv_norm=conv_norm, lfu_mode=lfu_mode, dwt_params=dwt_params)
    use_global_input = True
    down = 2
    test_ffc_main(x_input, device, use_global_input=use_global_input, down=down, conv_type=conv_type,
        conv_norm=conv_norm, lfu_mode=lfu_mode, dwt_params=dwt_params)
    print('FFC layers tests passed')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Testing {__file__} with device={device} ...')
    bs = 16
    # 1. Other layers
    test_other_layers(bs, device)
    # 2. Convolution
    test_conv_layers(bs, device)
    # 3. Fast Fourier convolution
    test_ffc_layers(bs, device)

    print(f'\n--- All tests completed for {__file__} ---')
