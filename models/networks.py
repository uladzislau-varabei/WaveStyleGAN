from collections import OrderedDict

import torch

from logger import log_message
from models.config import AVAILABLE_MODELS
from models.networks_stylegan2 import SG2Generator, SG2Discriminator
from models.networks_stylegan3 import SG3Generator
from models.networks_stylewavegan import Generator, Discriminator


# ----- Generator -----

def build_SG2_Generator(model_config, config):
    general_params = config['general_params']
    training_params = config['training_params']
    mapping_params = model_config['Mapping']
    synthesis_params = model_config['Synthesis']
    mapping_kwargs = dict(
        num_layers=mapping_params['num_layers'],
        # embed_features=None, # keep None for auto setting
        # layer_features=None,
        activation=mapping_params['activation'],
        lr_multiplier=mapping_params['lr_multiplier'],
        w_avg_beta=mapping_params['w_avg_beta'],
        bias_act_impl=training_params['bias_act_impl']
    )
    synthesis_layer_kwargs = dict(
        kernel_size=synthesis_params['kernel_size'],
        # Note: different modes for noise provided in forward:  ['random', 'const', 'none']
        use_noise=synthesis_params['use_noise'],
        activation=synthesis_params['activation']
    )
    synthesis_block_kwargs = dict(
        architecture=synthesis_params['architecture'],
        resample_filter=synthesis_params['resample_filter'],
        conv_clamp=synthesis_params['conv_clamp'],
        fp16_channels_last=general_params['channels_last'],
        fused_modconv_default=synthesis_params['fused_modconv_default'],
        use_custom_conv2d_op=training_params['use_custom_conv2d_op'],
        upfirdn2d_impl=training_params['upfirdn2d_impl'],
        bias_act_impl=training_params['bias_act_impl'],
        **synthesis_layer_kwargs
    )
    synthesis_kwargs = dict(
        channel_base=synthesis_params['channel_base'],
        channel_max=synthesis_params['channel_max'],
        num_fp16_res=general_params['num_fp16_res'],
        **synthesis_block_kwargs
    )
    model = SG2Generator(z_dim=general_params['z_dim'],
                         c_dim=general_params['c_dim'],
                         w_dim=general_params['w_dim'],
                         img_resolution=general_params['img_resolution'],
                         img_channels=general_params['img_channels'],
                         mapping_kwargs=mapping_kwargs,
                         **synthesis_kwargs)
    return model


def build_SG3_Generator(model_config, config):
    general_params = config['general_params']
    training_params = config['training_params']
    mapping_params = model_config['Mapping']
    synthesis_params = model_config['Synthesis']
    mapping_kwargs = dict(
        num_layers=mapping_params['num_layers'],
        lr_multiplier=mapping_params['lr_multiplier'],
        w_avg_beta=mapping_params['w_avg_beta'],
        # Not used
        # activation=mapping_params['activation'],
        # bias_act_impl=training_params['bias_act_impl']
    )
    synthesis_layer_kwargs = dict(
        conv_kernel=synthesis_params['conv_kernel'],
        filter_size=synthesis_params['filter_size'],
        lrelu_upsampling=synthesis_params['lrelu_upsampling'],
        use_radial_filters=synthesis_params['use_radial_filters'],
        conv_clamp=synthesis_params['conv_clamp'],
        magnitude_ema_beta=synthesis_params['magnitude_ema_beta']
    )
    synthesis_kwargs = dict(
        channel_base=synthesis_params['channel_base'],
        channel_max=synthesis_params['channel_max'],
        num_layers=synthesis_params['num_layers'],
        num_critical=synthesis_params['num_critical'],
        first_cutoff=synthesis_params['first_cutoff'],
        first_stopband=synthesis_params['first_stopband'],
        last_stopband_rel=synthesis_params['last_stopband_rel'],
        margin_size=synthesis_params['margin_size'],
        output_scale=synthesis_params['output_scale'],
        num_fp16_res=general_params['num_fp16_res'],
        **synthesis_layer_kwargs
    )
    model = SG3Generator(z_dim=general_params['z_dim'],
                         c_dim=general_params['c_dim'],
                         w_dim=general_params['w_dim'],
                         img_resolution=general_params['img_resolution'],
                         img_channels=general_params['img_channels'],
                         mapping_kwargs=mapping_kwargs,
                         **synthesis_kwargs)
    return model


def build_G_model(model_config, config, rank=1, logger=None):
    architecture = config['general_params']['architecture']
    assert architecture.lower() in AVAILABLE_MODELS
    log_message(f'Building Generator model with architecture={architecture}...', rank, logger)
    architecture = architecture.lower()
    if architecture == 'stylewavegan':
        G_model = Generator(model_config, config, rank=rank, logger=logger)
    elif architecture == 'stylegan2':
        G_model = build_SG2_Generator(model_config, config)
    elif architecture in ['stylegan3-t', 'stylegan3-r']:
        G_model = build_SG3_Generator(model_config, config)
    else:
        assert False, f'Model architecture={architecture} is not supported'
    return G_model


# ----- Discriminator -----

def build_SG2_Discriminator(config):
    general_params = config['general_params']
    model_params = config['models_params']['Discriminator']
    training_params =config['training_params']
    block_kwargs = dict(
        activation=model_params['activation'],
        resample_filter=model_params['resample_filter'],
        fp16_channels_last=general_params['channels_last'],
        freeze_layers=0, # different value only for fine-tuning
    )
    mapping_kwargs = {} # only set for dataset with multiple classes, maybe keep empty for auto setting
    epilogue_kwargs = dict(
        # mbstd_group_size=4, # set to default value
        # mbstd_num_channels=1, # set to default value
        activation=model_params['activation'],
    )
    model = SG2Discriminator(c_dim=general_params['c_dim'],
                             img_resolution=general_params['img_resolution'],
                             img_channels=general_params['img_channels'],
                             architecture=model_params['architecture'],
                             channel_base=model_params['channel_base'],
                             channel_max=model_params['channel_max'],
                             num_fp16_res=general_params['num_fp16_res'],
                             conv_clamp=model_params['conv_clamp'],
                             cmap_dim=None, # keep None for auto setting
                             use_custom_conv2d_op=training_params['use_custom_conv2d_op'],
                             upfirdn2d_impl=training_params['upfirdn2d_impl'],
                             bias_act_impl=training_params['bias_act_impl'],
                             block_kwargs=block_kwargs,
                             mapping_kwargs=mapping_kwargs,
                             epilogue_kwargs=epilogue_kwargs)
    return model


def build_D_model(model_config, config, rank=0, logger=None):
    architecture = config['general_params']['architecture']
    assert architecture.lower() in AVAILABLE_MODELS
    log_message(f'Building Discriminator model with architecture={architecture}...', rank, logger)
    architecture = architecture.lower()
    if architecture == 'stylewavegan':
        D_model = Discriminator(model_config, config, rank=rank, logger=logger)
    elif architecture in ['stylegan2', 'stylegan3-t', 'stylegan3-r']:
        # StyleGAN2 and StyleGAN3 use the same Discriminator
        D_model = build_SG2_Discriminator(config)
    else:
        assert False, f'Model architecture={architecture} is not supported'
    return D_model


# ----- Inference models -----

def prepare_G_model(weights_path, config, device):
    G_model = build_G_model(config['models_params']['Generator'], config).to(device)
    weights = torch.load(weights_path, weights_only=True, map_location=device)
    weights = OrderedDict([k.replace('module.', ''), v] for k, v in weights.items())
    G_model.load_state_dict(weights)
    G_model.eval().requires_grad_(False)
    print(f'Weight loaded from {weights_path}')
    return G_model
