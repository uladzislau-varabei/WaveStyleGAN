import numpy as np
import torch
import torch.nn as nn

from models import misc
from models.DiffAugment import DiffAug
from models.layers import FullyConnectedLayer, FourierSynthesisInput, WaveletLayer
from models.networks_blocks import DiscriminatorBlock, DiscriminatorEpilogue, DiscriminatorHead, \
    CCMBlock, CSMBlock, SynthesisBlock
from models.utils import normalize_2nd_moment
from logger import log_message
from shared_utils import NHWC_DATA_FORMAT, DEBUG_MODE, LOG_STRIDE_INFO, G_TANH_OUTPUT, USE_LEGACY_BEHAVIOUR, \
    check_input_imgs, log_model_res_info, num_channels


class MappingNetwork(nn.Module):
    def __init__(self, model_config, config):
        super().__init__()
        general_params = config['general_params']
        # Input latent (Z) dimensionality.
        self.z_dim = general_params['z_dim']
        # Conditioning label (C) dimensionality, 0 = no labels.
        num_classes = general_params['num_classes']
        c_dim = 0 if num_classes <= 1 else num_classes
        self.c_dim = c_dim
        # Intermediate latent (W) dimensionality.
        self.w_dim = general_params['w_dim']
        # Number of intermediate latents to output.
        self.num_ws = model_config['num_ws']
        # Number of mapping layers. Default is 8
        self.num_layers = model_config['num_layers']
        # Activation for the mapping layers. Default is lrelu
        self.activation = model_config['activation']
        # Learning rate multiplier for the mapping layers. Default is 0.01
        self.lr_multiplier = model_config['lr_multiplier']
        # Decay for tracking the moving average of W during training. Default is 0.998
        self.w_avg_beta = model_config['w_avg_beta']
        # Implementation of bias with activation. One of ['ref', 'cuda'].
        self.bias_act_impl = config['training_params']['bias_act_impl']
        # Build model.
        self.build_model()

    def build_model(self):
        # Construct layers.
        self.embed = FullyConnectedLayer(self.c_dim, self.w_dim, bias_act_impl=self.bias_act_impl) \
            if self.c_dim > 0 else None
        features = [self.z_dim + (self.w_dim if self.c_dim > 0 else 0)] + [self.w_dim] * self.num_layers
        for idx, in_features, out_features in zip(range(self.num_layers), features[:-1], features[1:]):
            layer = FullyConnectedLayer(in_features, out_features, activation=self.activation,
                lr_multiplier=self.lr_multiplier, bias_act_impl=self.bias_act_impl)
            setattr(self, f'fc{idx}', layer)
        self.register_buffer('w_avg', torch.zeros([self.w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        misc.assert_shape(z, [None, self.z_dim])
        if truncation_cutoff is None:
            truncation_cutoff = self.num_ws

        # Embed, normalize, and concatenate inputs.
        x = z.to(torch.float32)
        x = normalize_2nd_moment(x)
        if self.c_dim > 0:
            misc.assert_shape(c, [None, self.c_dim])
            y = self.embed(c.to(torch.float32))
            y = normalize_2nd_moment(y)
            x = torch.cat([x, y], dim=1) if x is not None else y

        # Execute layers.
        for idx in range(self.num_layers):
            x = getattr(self, f'fc{idx}')(x)

        # Update moving average of W.
        if update_emas:
            self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast and apply truncation.
        x = x.unsqueeze(1).repeat([1, self.num_ws, 1])
        if truncation_psi != 1:
            x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

    def extra_repr(self):
        return f'z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}'


class SynthesisNetwork(nn.Module):
    def __init__(self, model_config, config, rank=0, logger=None):
        super().__init__()
        # 1. --- General params ---
        general_params = config['general_params']
        training_params = config['training_params']
        self.dwt_params = config['dwt_params']
        self.dwt_params['init_torgb_wavelet_scales'] = model_config['init_torgb_wavelet_scales']
        self.use_wavelet = self.dwt_params['wavelet'] is not None
        # Intermediate latent (W) dimensionality.
        self.w_dim = general_params['w_dim']
        # Output image resolution and channels.
        img_shape_yxc = general_params['img_shape_yxc']
        self.img_channels = img_shape_yxc[2]
        # Note: for rectangular data number of predicted pixels is the same for H and W
        self.img_start_resolution = int(general_params['start_resolution'])
        # If wavelet is used then output image resolution must be 2x smaller
        self.img_target_resolution = int(general_params['target_resolution']) // (2 if self.use_wavelet else 1)
        self.img_start_resolution_log2 = int(np.log2(self.img_start_resolution))
        # If wavelet is used then output image resolution must be 2x smaller
        self.img_target_resolution_log2 = int(np.log2(self.img_target_resolution))
        # Note: start resolutions of models are StyleGAN2 - 2, StyleGAN3 - 4, StyleGAN-T - 3
        self.block_resolutions = [2 ** i for i in
            range(self.img_start_resolution_log2, self.img_target_resolution_log2 + 1)]
        # Use FP16 for the N highest resolutions. Default is 4.
        self.num_fp16_res = general_params['num_fp16_res']
        self.fp16_resolution = max(2 ** (self.img_target_resolution_log2 + 1 - self.num_fp16_res), 8)
        self.channels_last = training_params['data_format'] == NHWC_DATA_FORMAT
        self.use_custom_conv2d_op = training_params['use_custom_conv2d_op']
        self.upfirdn2d_impl = training_params['upfirdn2d_impl']
        self.bias_act_impl = training_params['bias_act_impl']
        # 2. --- Model params ---
        # Input implementation: 1 - Fourier input separately, 2 - Fourier input inside model block,
        # 3 - Fourier input inside model block and the 1st model block has the same resolution without upsampling
        self.input_impl_idx = model_config['input_impl_idx']
        # Input implementation: Fourier - Fourier features, Const - trainable const input.
        self.input_type = model_config['input_type']
        # Architecture: 'orig', 'skip', 'resnet'. Default is 'slip'.
        self.architecture = model_config['architecture']
        # Overall multiplier for the number of channels. Default is 32768 for SG3-T.
        self.channel_base = model_config['channel_base']
        # Maximum number of channels in any layer. Default is 512.
        self.channel_max = model_config['channel_max']
        # Number of output channels for each resolution.
        self.channels_dict = {
            res: num_channels(res, self.channel_base, self.channel_max) for res in self.block_resolutions
        }
        log_message(f'Generator channels dict: {self.channels_dict}', rank, logger)
        # Activation for conv layers.
        self.activation = model_config['activation']
        # Clamp the output of convolution layers to +-X, None = disable clamping.
        self.conv_clamp = model_config['conv_clamp']
        # Low-pass filter to apply when resampling activations.
        self.resample_filter = model_config['resample_filter']
        # Default value of fused_modconv. 'inference_only' = True for inference, False for training.
        self.fused_modconv_default = model_config['fused_modconv_default']
        # Build model.
        self.build_model()

    def build_input_layer(self, common_kwargs):
        self.input_block = None
        if self.input_impl_idx == 1:
            if self.input_type.lower() == 'Fourier'.lower():
                # The same params as for StyleGAN-T
                input_res = self.block_resolutions[0]
                # TODO: fix sampling_rate. Division by sampling_rate / 2 - bandwidth
                if USE_LEGACY_BEHAVIOUR:
                    sampling_rate_add = 0
                    bandwidth = 2
                else:
                    # sampling_rate_add = 4 if resolution == 4 else 0
                    sampling_rate_add = 0
                    bandwidth = 1 if input_res == 4 else 2
                self.input_block = FourierSynthesisInput(w_dim=self.w_dim,
                                                         channels=self.channels_dict[input_res],
                                                         size=input_res,
                                                         sampling_rate=input_res + sampling_rate_add,
                                                         bandwidth=bandwidth,
                                                         bias_act_impl=self.bias_act_impl)
                self.num_ws = 1
            elif self.input_type.lower() == 'const':
                assert False, 'input_type=const must be used with input_impl_idx in [2, 3], not 1'
            else:
                assert False, f'input_type={self.input_type} is not supported'
        elif self.input_impl_idx == 2:
            # Input features are inside the 1st block
            self.num_ws = 0
        elif self.input_impl_idx == 3:
            input_res = self.block_resolutions[0]
            self.input_block = SynthesisBlock(0, self.channels_dict[input_res],
                resolution=input_res, is_first=True, is_last=False, use_fp16=False, **common_kwargs)
            self.num_ws = self.input_block.num_conv + self.input_block.num_torgb
        else:
            assert False, f'input_impl_idx={self.input_impl_idx} is not supported for SynthesisNetwork'
        self.input_num_ws = self.num_ws

    def build_model(self):
        if self.use_wavelet:
            self.wavelet_layer = WaveletLayer(self.dwt_params['wavelet'],
                                              use_affine=self.dwt_params['use_affine'],
                                              init_affine_scales=self.dwt_params['init_affine_scales'],
                                              train_affine=self.dwt_params['train_affine'],
                                              affine_lr_multiplier=self.dwt_params['affine_lr_multiplier'],
                                              train_kernel=self.dwt_params['train_kernel'],
                                              scale_1d_coeffs=self.dwt_params['scale_1d_coeffs'],
                                              scale_2d_coeffs=self.dwt_params['scale_2d_coeffs'],
                                              coeffs_scales_2d_version=self.dwt_params['coeffs_scales_2d_version'])
        else:
            self.wavelet_layer = None
        common_kwargs = dict(w_dim=self.w_dim, img_channels=self.img_channels, architecture=self.architecture,
            activation=self.activation, resample_filter=self.resample_filter, conv_clamp=self.conv_clamp,
            fp16_channels_last=self.channels_last, channels_last=self.channels_last,
            use_custom_conv2d_op=self.use_custom_conv2d_op, upfirdn2d_impl=self.upfirdn2d_impl,
            bias_act_impl=self.bias_act_impl, fused_modconv_default=self.fused_modconv_default,
            dwt_params=self.dwt_params, input_impl_idx=self.input_impl_idx, input_type=self.input_type)
        # print(f"Synthesis common kwargs: {common_kwargs}")
        self.build_input_layer(common_kwargs)
        self.blocks = nn.ModuleDict()
        start_res = self.img_start_resolution
        for res in self.block_resolutions:
            if res > start_res:
                in_channels = self.channels_dict[res // 2]
            else:
                if self.input_impl_idx == 1:
                    in_channels = self.channels_dict[start_res]
                elif self.input_impl_idx == 2:
                    in_channels = 0
                elif self.input_impl_idx == 3:
                    in_channels = self.channels_dict[start_res]
                else:
                    assert False, f'input_impl_idx={self.input_impl_idx} is not supported for SynthesisNetwork'
            out_channels = self.channels_dict[res]
            if DEBUG_MODE:
                print(f'Synthesis: res={res}, in_ch={in_channels}, out_ch={out_channels}')
            use_fp16 = (res >= self.fp16_resolution)
            is_first = (res == self.block_resolutions[0])
            is_last = (res == self.img_target_resolution)
            block = SynthesisBlock(in_channels, out_channels,
                resolution=res, is_first=is_first, is_last=is_last, use_fp16=use_fp16, **common_kwargs)
            self.num_ws += block.num_conv
            # if is_last:
            self.num_ws += block.num_torgb  # originally only for the last block
            self.blocks[str(res)] = block

    def output_idwt(self, x):
        assert self.wavelet_layer is not None, 'Synthesis: output_idwt() can only be used with wavelets'
        x = self.wavelet_layer(x, inverse=True)
        if G_TANH_OUTPUT:
            x = nn.functional.tanh(x)
        return x

    def output_dwt(self, x):
        assert self.wavelet_layer is not None, 'Synthesis: output_dwt() can only be used with wavelets'
        return self.wavelet_layer(x, inverse=False)

    def forward(self, ws, apply_idwt, **layer_kwargs):
        # 1. Prepare mapping network outputs
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            if self.input_num_ws > 0:
                block_ws.append(ws.narrow(1, 0, self.input_num_ws))
            w_idx = self.input_num_ws
            for res in self.block_resolutions:
                block = self.blocks[str(res)]
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv + block.num_torgb

        # 2. Run layers
        x = None
        img = None
        if self.input_impl_idx == 1:
            x = self.input_block(block_ws[0].squeeze(1))
            ws_start_idx = 1
        elif self.input_impl_idx == 2:
            # Input is inside the 1st resolution block
            ws_start_idx = 0
        elif self.input_impl_idx == 3:
            # Note: img must still be None
            x, _ = self.input_block(x, img, block_ws[0], **layer_kwargs)
            ws_start_idx = 1
        else:
            assert False, f'input_impl_idx={self.input_impl_idx} is not supported for SynthesisNetwork'
        for res, cur_ws in zip(self.block_resolutions, block_ws[ws_start_idx:]):
            if DEBUG_MODE:
               log_model_res_info(x, img, res, 'G start')
            block = self.blocks[str(res)]
            # print(f'res={res}, cur_ws.shape={cur_ws.shape}, ws.shape={ws.shape}')
            x, img = block(x, img, cur_ws, **layer_kwargs)
            if DEBUG_MODE:
                log_model_res_info(x, img, res, 'G end')

        # 3. Check shape and dtype
        misc.assert_shape(img, [None, self.img_channels * (4 if self.use_wavelet else 1),
            self.img_target_resolution, self.img_target_resolution])
        img = img.to(torch.float32)

        # 4. Convert wavelet coeffs to image
        if apply_idwt and self.use_wavelet:
            # Tanh is applied inside output_idwt
            img = self.output_idwt(img)

        if G_TANH_OUTPUT and not self.use_wavelet:
            img = nn.functional.tanh(img)
        return img

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_start_resolution={self.img_start_resolution:d},'
            f'img_target_resolution={self.img_target_resolution:d},'
            f'img_channels={self.img_channels:d}, num_fp16_res={self.num_fp16_res:d}'])


class Generator(nn.Module):
    def __init__(self, model_config, config, rank=0, logger=None):
        super().__init__()
        # TODO: maybe find a better way for determination of num_ws parameters
        self.synthesis = SynthesisNetwork(model_config['Synthesis'], config, rank, logger)
        mapping_config = model_config['Mapping']
        mapping_config['num_ws'] = self.synthesis.num_ws
        self.mapping = MappingNetwork(mapping_config, config)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, apply_idwt=True, **synthesis_kwargs):
        if DEBUG_MODE:
            print(f'Synthesis kwargs: {synthesis_kwargs}')
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, update_emas=update_emas, apply_idwt=apply_idwt, **synthesis_kwargs)
        return img


class Discriminator(nn.Module):
    def __init__(self, model_config, config, rank=0, logger=None):
        super().__init__()
        # 1. ----- General params -----
        general_params = config['general_params']
        training_params = config['training_params']
        self.dwt_params = config['dwt_params']
        if USE_LEGACY_BEHAVIOUR:
            self.dwt_params['init_fromrgb_wavelet_scales'] = None
        else:
            self.dwt_params['init_fromrgb_wavelet_scales'] = model_config['init_fromrgb_wavelet_scales']
        self.use_wavelet = self.dwt_params['wavelet'] is not None
        # Output image resolution and channels.
        img_shape_yxc = general_params['img_shape_yxc']
        self.img_channels = img_shape_yxc[2] * (4 if self.use_wavelet else 1)
        # Note: for rectangular data number of predicted pixels is the same for H and W
        self.img_start_resolution = 4 # for discriminator use all the layers, general_params['start_resolution']
        # If wavelet is used then input is DWT transform of image and its resolution is 2x lower
        self.img_target_resolution = int(general_params['target_resolution']) // (2 if self.use_wavelet else 1)
        self.img_start_resolution_log2 = int(np.log2(self.img_start_resolution))
        self.img_target_resolution_log2 = int(np.log2(self.img_target_resolution))
        self.block_resolutions = [2 ** i for i in
            range(self.img_target_resolution_log2, self.img_start_resolution_log2 - 1, -1)]
        # Conditioning label (C) dimensionality, 0 = no labels.
        num_classes = general_params['num_classes']
        c_dim = 0 if num_classes <= 1 else num_classes
        self.c_dim = c_dim
        # Use FP16 for the N highest resolutions. Default is 4.
        self.num_fp16_res = general_params['num_fp16_res']
        self.fp16_resolution = max(2 ** (self.img_target_resolution_log2 + 1 - self.num_fp16_res), 8)
        self.channels_last = training_params['data_format'] == NHWC_DATA_FORMAT
        self.use_custom_conv2d_op = training_params['use_custom_conv2d_op']
        self.upfirdn2d_impl = training_params['upfirdn2d_impl']
        self.bias_act_impl = training_params['bias_act_impl']
        # 2. --- Model params ---
        # Convolution layer type: ['base', 'selfmod', 'spectral']
        self.conv_type = model_config['conv_type']
        # Fast Fourier convolution params
        self.ffc_params = model_config['ffc_params']
        # Projected model params
        self.projection_params = model_config['projection_params']
        # Architecture: 'orig', 'skip', 'resnet'. Default is 'resnet'.
        self.architecture = model_config['architecture']
        # Overall multiplier for the number of channels. Default is 16384.
        self.channel_base = model_config['channel_base']
        # Maximum number of channels in any layer. Default is 512.
        self.channel_max = model_config['channel_max']
        # Number of output channels for each resolution.
        self.channels_dict = {
            res: num_channels(res, self.channel_base, self.channel_max) for res in self.block_resolutions
        }
        log_message(f'Discriminator channels dict: {self.channels_dict}', rank, logger)
        # Activation for conv layers.
        self.activation = model_config['activation']
        # Clamp the output of convolution layers to +-X, None = disable clamping.
        self.conv_clamp = model_config['conv_clamp']
        # Low-pass filter to apply when resampling activations.
        self.resample_filter = model_config['resample_filter']
        # Update cls mapping info.
        self.c_map_dim = self.channels_dict[self.img_start_resolution]
        if self.c_dim == 0:
            self.c_map_dim = 0
        self.use_diff_aug = general_params['use_diff_aug']
        # Other params (FFC + projection)
        self.use_ffc = self.ffc_params['use_ffc']
        self.ffc_num_skipped_res = 0
        if self.use_ffc:
            self.ffc_num_skipped_res = self.ffc_params['num_skipped_res']
        self.ffc_start_res = self.block_resolutions[self.ffc_num_skipped_res]
        self.use_projection = self.projection_params['use_projection']
        self.num_projection_heads = self.projection_params['num_heads']
        self.projection_min_res = 8 # resolution refers to block input size, output size is reduced by 2
        self.projection_max_res = self.projection_min_res * (2 ** (self.num_projection_heads - 1))
        self.projection_resolutions = self.block_resolutions[::-1][1: self.num_projection_heads + 1]
        self.projection_mixing_out_max_channels = self.projection_params['mixing_out_max_channels']
        # Update output from the lowest resolution as well?
        self.projection_update_all_outputs = self.projection_mixing_out_max_channels is not None
        # Shared params for blocks.
        self.shared_general_params = dict(conv_type=self.conv_type, conv_clamp=self.conv_clamp,
            fp16_channels_last=self.channels_last, channels_last=self.channels_last,
            use_custom_conv2d_op = self.use_custom_conv2d_op, bias_act_impl=self.bias_act_impl)
        self.shared_upfirdn_params = dict(resample_filter=self.resample_filter, upfirdn2d_impl=self.upfirdn2d_impl,
            dwt_params=self.dwt_params)
        # Build model.
        self.build_model()

    def build_projection_blocks(self):
        # Make CCM (cross channel mixing) blocks
        self.ccm_blocks = nn.ModuleDict()
        projected_channels = {}
        ccm_channels = {}
        for res in self.projection_resolutions:
            res_key = str(res)
            in_channels = self.channels_dict[res // 2]
            # in_channels = out_channels from this block
            if self.projection_update_all_outputs:
                out_channels = min(in_channels, self.projection_mixing_out_max_channels)
            else:
                out_channels = in_channels
            if res > self.projection_min_res or self.projection_update_all_outputs:
                # Note: src res is referred to input res, output res is reduced by 2
                use_fp16 = (res >= self.fp16_resolution)
                ccm_block = CCMBlock(in_channels, out_channels, res, architecture='base',
                    activation='linear', use_fp16=use_fp16, **self.shared_general_params)
            else:
                # If for the last layer number of channels is the same then don't use extra layer
                ccm_block = nn.Identity()
            self.ccm_blocks[res_key] = ccm_block
            projected_channels[res_key] = out_channels
            ccm_channels[res] = out_channels
        # Make CSM (cross stage mixing) blocks
        self.csm_blocks = nn.ModuleDict()
        csm_channels = {}
        for res in self.projection_resolutions:
            use_fp16 = (res >= self.fp16_resolution)
            res_ccm_channels = ccm_channels[res]
            if res > self.projection_min_res:
                res_csm_channels = csm_channels[res // 2]
                # TODO: add cond to project to another number of filters
                out_channels = res_ccm_channels
                csm_block = CSMBlock(csm_in_channels=res_csm_channels, ccm_in_channels=res_ccm_channels,
                    out_channels=out_channels, resolution=res, architecture='base', activation=self.activation,
                    use_fp16=use_fp16, **self.shared_general_params, **self.shared_upfirdn_params)
            else:
                out_channels = res_ccm_channels
                csm_block = nn.Identity()
            self.csm_blocks[str(res)] = csm_block
            csm_channels[res] = out_channels
        self.heads_channels = csm_channels
        print(f'ccm_blocks: {self.ccm_blocks}, csm_blocks: {self.csm_blocks}')

    def build_heads(self):
        self.heads = nn.ModuleDict()
        for res in self.projection_resolutions:
            in_channels = self.heads_channels[res]
            use_fp16 = (res >= self.fp16_resolution)
            if DEBUG_MODE:
                print(f'Head for res={res}: in_channels={in_channels}')
            # Note: resolution refers to block input size, output size is reduced by 2
            self.heads[str(res)] = DiscriminatorHead(in_channels,
                main_channels=self.projection_params['head_main_channels'], c_dim=self.c_dim, c_map_dim=self.c_map_dim,
                resolution=res, target_resolution=self.projection_min_res // 2, activation=self.activation,
                use_fp16=use_fp16, architecture=self.projection_params['head_architecture'],
                **self.shared_general_params, **self.shared_upfirdn_params)
        print(f'Discriminator heads: {self.heads}')

    def build_resolutions_blocks(self, common_kwargs):
        cur_layer_idx = 0
        self.blocks = nn.ModuleDict()
        # The last resolution is for epilogue
        for res in self.block_resolutions[:-1]:
            in_channels = self.channels_dict[res] if res < self.img_target_resolution else 0
            tmp_channels = self.channels_dict[res]
            out_channels = self.channels_dict[res // 2]
            use_fp16 = (res >= self.fp16_resolution)
            is_first_ffc_block = res == self.ffc_start_res
            is_last_ffc_block = res == self.block_resolutions[-2]
            ffc_params = self.ffc_params if res <= self.ffc_start_res else None # resolution decreases
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels,
                resolution=res, first_layer_idx=cur_layer_idx, use_fp16=use_fp16,
                is_first_ffc_block=is_first_ffc_block, is_last_ffc_block=is_last_ffc_block,
                ffc_params=ffc_params, **common_kwargs)
            self.blocks[str(res)] = block
            cur_layer_idx += block.num_layers

    def build_conditioning_network(self):
        if self.c_dim > 0 and not self.use_projection:
            # TODO: check later on a real config
            G_mapping_config = {}
            D_mapping_config = dict(z_dim=0, c_dim=self.c_dim, w_dim=self.c_map_dim, num_ws=None, w_avg_beta=None)
            D_mapping_general_config = {}
            self.mapping = MappingNetwork(D_mapping_config, D_mapping_general_config)
        else:
            self.mapping = None

    def build_output_blocks(self, common_kwargs):
        if self.use_projection:
            self.build_heads()
            self.epilogue = None
        else:
            self.heads = None
            unused_epilogue_keys = ['resample_filter', 'fp16_channels_last', 'channels_last']
            for key in unused_epilogue_keys:
                common_kwargs.pop(key, None)
            self.epilogue = DiscriminatorEpilogue(self.channels_dict[self.img_start_resolution],
                c_map_dim=self.c_map_dim, resolution=self.img_start_resolution, ffc_params=self.ffc_params,
                **common_kwargs)

    def build_model(self):
        # 1. Build DiffAug module
        self.diff_aug = DiffAug(policy='color,translation,cutout') if self.use_diff_aug else None
        # 2. Build input wavelet layer
        if self.use_wavelet:
            self.wavelet_layer = WaveletLayer(self.dwt_params['wavelet'],
                                              use_affine=self.dwt_params['use_affine'],
                                              init_affine_scales=self.dwt_params['init_affine_scales'],
                                              train_affine=self.dwt_params['train_affine'],
                                              affine_lr_multiplier=self.dwt_params['affine_lr_multiplier'],
                                              train_kernel=self.dwt_params['train_kernel'],
                                              scale_1d_coeffs=self.dwt_params['scale_1d_coeffs'],
                                              scale_2d_coeffs=self.dwt_params['scale_2d_coeffs'],
                                              coeffs_scales_2d_version=self.dwt_params['coeffs_scales_2d_version'],
                                              always_forward=True)
        else:
            self.wavelet_layer = None
        # 3. Build projection layers
        if self.use_projection:
            self.build_projection_blocks()
        # 4. Build resolution layers
        common_kwargs = dict(img_channels=self.img_channels, architecture=self.architecture,
                activation=self.activation)
        common_kwargs = {**common_kwargs, **self.shared_general_params, **self.shared_upfirdn_params}
        # print(f"Discriminator common kwargs: {common_kwargs}")
        self.build_resolutions_blocks(common_kwargs)
        # 5. Build conditioning labels mapping network (only with disabled projection)
        self.build_conditioning_network()
        # 6. Build output layers: projection heads or epilogue
        self.build_output_blocks(common_kwargs)

    def input_dwt(self, x):
        assert self.wavelet_layer is not None, 'Discriminator: input_dwt() can only be used with wavelets'
        return self.wavelet_layer(x, inverse=False)

    def forward_ccm(self, outputs):
        ccm_outputs = {}
        for res in self.projection_resolutions:
            x = outputs[res]
            block = self.ccm_blocks[str(res)]
            force_fp32 = False
            if DEBUG_MODE:
                print(f'Forward CCM for res={res}')
            if res == self.projection_min_res and (not self.projection_update_all_outputs):
                ccm_outputs[res] = x # identity
            else:
                ccm_outputs[res] = block(x, None, force_fp32)  # img is None, force_fp32=False
        if DEBUG_MODE:
            for k, v in ccm_outputs.items():
                print(f'ccm output: res={k}, shape={v.shape}')
        return ccm_outputs

    def forward_csm(self, outputs):
        csm_outputs = {}
        for res in self.projection_resolutions:
            x = outputs[res]
            block = self.csm_blocks[str(res)]
            force_fp32 = False
            if DEBUG_MODE:
                print(f'Forward CSM for res={res}')
            if res == self.projection_min_res:
                csm_outputs[res] = x # identity
            else:
                prev_csm_output = csm_outputs[res // 2]
                # Note: x = (x_csm, x_ccm)
                csm_outputs[res] = block((prev_csm_output, x), None, force_fp32)  # img is None, force_fp32=False
        if DEBUG_MODE:
            for k, v in csm_outputs.items():
                print(f'csm output: res={k}, shape={v.shape}')
        return csm_outputs

    def forward_heads(self, outputs, c):
        logits = []
        for res in self.projection_resolutions:
            x = outputs[res]
            block = self.heads[str(res)]
            force_fp32 = False
            if DEBUG_MODE:
                print(f'Forward projected head for res={res}')
            logits.append(
                block(x, c, None, force_fp32).view(x.size(0), -1)
            )
            if DEBUG_MODE:
                print(f'Logits shape for res={res}: {logits[-1].shape}')
        logits = torch.cat(logits, dim=1)
        if DEBUG_MODE:
            print(f'Projected logits prepared: {logits.shape}')
        return logits

    def forward(self, img, c, update_emas=False, **block_kwargs):
        if DEBUG_MODE:
            img_stride = img.stride() if LOG_STRIDE_INFO else -1
            print(f'D input img: shape={img.shape}, stride={img_stride}, dtype={img.dtype}')
        _ = update_emas # unused
        x = None
        # Real img is in range [-1, 1]. Gen img is unbounded unless tanh is applied
        if G_TANH_OUTPUT:
            check_input_imgs(img)
        if self.use_diff_aug:
            img = self.diff_aug(img)
        if self.use_wavelet:
            img = self.input_dwt(img)
        # The last resolution is for epilogue
        projection_outputs = {}
        for res in self.block_resolutions[:-1]:
            block = self.blocks[str(res)]
            if DEBUG_MODE:
                log_model_res_info(x, img, res, 'D start')
            x, img = block(x, img, **block_kwargs)
            if self.use_projection and res <= self.projection_max_res:
                projection_output = x[0] if self.use_ffc and res == self.projection_min_res else x
                projection_outputs[res] = projection_output
            if DEBUG_MODE:
                log_model_res_info(x, img, res, 'D end')

        c_map = None
        if self.c_dim > 0 and not self.use_projection:
            # Note: for projection conditioning labels are processed in heads
            c_map = self.mapping(None, c)

        if self.use_projection:
            ccm_outputs = self.forward_ccm(projection_outputs)
            csm_outputs = self.forward_csm(ccm_outputs)
            x = self.forward_heads(csm_outputs, c)
        else:
            if DEBUG_MODE:
                log_model_res_info(x, img, 0, 'D before epilogue')
            x = self.epilogue(x, img, c_map)
        return x

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_start_resolution={self.img_start_resolution:d}, ' \
               f'img_target_resolution={self.img_target_resolution:d}, img_channels={self.img_channels:d}'
