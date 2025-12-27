import copy

from shared_utils import get_total_batch_size
from logger import log_message

AVAILABLE_MODELS = ['stylewavegan', 'stylegan2', 'stylegan3-r', 'stylegan3-t']


def update_config_shared(config):
    # The same update for all StyleGAN2 and StyleGAN3 architectures (which are supported here)
    # Note: models_params for Generator and loss_params must be processed separately for each model
    config = copy.deepcopy(config)
    # 1. Update general params
    general_params = config['general_params']
    general_params['img_resolution'] = general_params['target_resolution']
    general_params['img_channels'] = general_params['img_shape_yxc'][2]
    num_classes = general_params['num_classes']
    general_params['c_dim'] = 0 if num_classes <= 1 else num_classes
    general_params['z_dim'] = 512
    general_params['w_dim'] = 512
    general_params['num_fp16_res'] = 4
    general_params['z_distribution'] = 'normal'
    general_params['use_compilation'] = False
    general_params['use_diff_aug'] = False
    data_format = config['training_params']['data_format']
    general_params['channels_last'] = (data_format.lower() == 'NHWC'.lower())
    config['general_params'] = general_params
    # 2. Additional params
    config['dwt_params'] = {'wavelet': None}
    config['ema_params'] = {
        'mode': 'base',
        'sema_kimg': None,
        'kimg': 10,
        'rampup': 0.05
    }
    # 3. Optimizers
    config['optimizers_params'] = {
        'Generator': {
            'type': 'adam',
            'lr': 0.002,
            'eps': 1e-8,
            'betas': [0.0, 0.99]  # arg for Adam optimizer
        },
        'Discriminator': {
            'type': 'adam',
            'lr': 0.002,
            'eps': 1e-8,
            'betas': [0.0, 0.99] # arg for Adam optimizer
        }
    }
    # 4. Training params
    # Pass: keep values from config
    return config


def update_config_for_SG2(architecture, config, rank=0, logger=None):
    assert architecture.lower() == 'stylegan2'
    config = update_config_shared(config)
    # 1. Update models architecture
    models_params = config['models_params']
    # 1.1 Shared params
    activation = 'lrelu'
    channel_max = 512
    channel_base = 32768
    conv_clamp = 256
    resample_filter = [1, 3, 3, 1]
    # 1.2 Mapping network
    G_mapping_params = models_params['Generator']['Mapping']
    G_mapping_params['num_layers'] = 8
    G_mapping_params['activation'] = activation
    G_mapping_params['lr_multiplier'] = 0.01
    # 1.3 Synthesis network
    G_synthesis_params = models_params['Generator']['Synthesis']
    G_synthesis_params['kernel_size'] = 3
    G_synthesis_params['architecture'] = 'skip'
    G_synthesis_params['channel_base'] = channel_base
    G_synthesis_params['channel_max'] = channel_max
    G_synthesis_params['activation'] = activation
    G_synthesis_params['conv_clamp'] = conv_clamp
    G_synthesis_params['resample_filter'] = resample_filter
    # Speed up training by using regular convolutions instead of grouped convolutions
    G_synthesis_params['fused_modconv_default'] = 'inference_only'
    G_synthesis_params['use_noise'] = True
    # 1.4 Discriminator network
    D_params = models_params['Discriminator']
    D_params['activation'] = activation
    D_params['resample_filter'] = resample_filter
    D_params['architecture'] = 'resnet'
    D_params['channel_base'] = channel_base
    D_params['channel_max'] = channel_max
    D_params['conv_clamp'] = conv_clamp
    # 1.5 Update config
    models_params['Generator']['Mapping'] = G_mapping_params
    models_params['Generator']['Synthesis'] = G_synthesis_params
    models_params['Discriminator'] = D_params
    # 2. Loss params (taken from StyleGAN3/StyleGAN2-ADA repo)
    loss_params = config['loss_params']
    loss_params['G_reg_interval'] = 4
    loss_params['D_reg_interval'] = 16
    loss_params['r1_gamma'] = 10
    loss_params['style_mixing_prob'] = 0.9
    loss_params['pl_weight'] = 2
    loss_params['pl_batch_shrink'] = 2
    loss_params['pl_decay'] = 0.01
    # Speed up training by using regular convolutions instead of grouped convolutions
    loss_params['pl_no_weight_grad'] = True
    loss_params['blur_init_sigma'] = 0
    loss_params['blur_fade_kimg'] = 0
    loss_params['G_reg_fade_kimg'] = 0
    loss_params['use_projection'] = False
    config['loss_params'] = loss_params
    log_message(f'Updated config for {architecture} architecture', rank, logger)
    return config


def update_config_for_SG3(architecture, config, rank=0, logger=None):
    # TODO: finish update of this config
    # https://github.com/NVlabs/stylegan3/blob/c233a919a6faee6e36a316ddd4eddababad1adf9/train.py#L234
    assert architecture.lower() in ['stylegan3-t', 'stylegan3-r']
    config = update_config_shared(config)
    # 1. Update models architecture
    models_params = config['models_params']
    # Note: maybe use total batch_size from the original implementation. Default is 32
    batch_size = get_total_batch_size(config=config)
    # 1.1 Shared params
    activation = 'lrelu'
    channel_max = 512
    channel_base = 32768
    conv_clamp = 256
    resample_filter = [1, 3, 3, 1]
    is_rotation_config = architecture.lower() in ['stylegan3-r']
    # 1.2 Mapping network
    G_mapping_params = models_params['Generator']['Mapping']
    G_mapping_params['num_layers'] = 2
    G_mapping_params['activation'] = activation
    G_mapping_params['lr_multiplier'] = 0.01
    G_mapping_params['w_avg_beta'] = 0.998
    # 1.3 Synthesis network
    G_synthesis_params = models_params['Generator']['Synthesis']
    G_synthesis_params['conv_kernel'] = 1 if is_rotation_config else 3
    G_synthesis_params['channel_base'] = int(channel_base * (2 if is_rotation_config else 1))
    G_synthesis_params['channel_max'] = int(channel_max * (2 if is_rotation_config else 1))
    G_synthesis_params['activation'] = activation
    G_synthesis_params['conv_clamp'] = conv_clamp
    G_synthesis_params['num_layers'] = 14
    G_synthesis_params['num_critical'] = 2
    G_synthesis_params['first_cutoff'] = 2
    G_synthesis_params['first_stopband'] = 2 ** 2.1
    G_synthesis_params['last_stopband_rel'] = 2 ** 0.3
    G_synthesis_params['margin_size'] = 10
    G_synthesis_params['output_scale'] = 0.25
    # Hyperparameters.
    G_synthesis_params['filter_size'] = 6
    G_synthesis_params['lrelu_upsampling'] = 2
    G_synthesis_params['use_radial_filters'] = is_rotation_config
    G_synthesis_params['magnitude_ema_beta'] = 0.5 ** (batch_size / (20 * 1e3))
    # 1.4 Discriminator network (the same as for SG2)
    D_params = models_params['Discriminator']
    D_params['activation'] = activation
    D_params['resample_filter'] = resample_filter
    D_params['architecture'] = 'resnet'
    D_params['channel_base'] = channel_base
    D_params['channel_max'] = channel_max
    D_params['conv_clamp'] = conv_clamp
    # 1.5 Update config
    models_params['Generator']['Mapping'] = G_mapping_params
    models_params['Generator']['Synthesis'] = G_synthesis_params
    models_params['Discriminator'] = D_params
    # 2. Loss params (taken from StyleGAN3 repo)
    loss_params = config['loss_params']
    loss_params['G_reg_interval'] = None
    loss_params['D_reg_interval'] = 16
    loss_params['r1_gamma'] = 10
    loss_params['style_mixing_prob'] = 0.0
    loss_params['pl_weight'] = 0
    loss_params['pl_batch_shrink'] = 2
    loss_params['pl_decay'] = 0.01
    # Speed up training by using regular convolutions instead of grouped convolutions
    loss_params['pl_no_weight_grad'] = True
    loss_params['blur_init_sigma'] = 10 if is_rotation_config else 0
    loss_params['blur_fade_kimg'] = batch_size * 200 / 32 if is_rotation_config else 0
    loss_params['G_reg_fade_kimg'] = 0
    loss_params['use_projection'] = False
    config['loss_params'] = loss_params
    # 3. Optimizer lr
    config['optimizers_params']['Generator']['lr'] = 0.0025
    log_message(f'Updated config for {architecture} architecture', rank, logger)
    return config


def update_config(config, rank=0, logger=None):
    config.pop('define') # remove key from params declaration
    architecture = config['general_params']['architecture']
    assert architecture.lower() in AVAILABLE_MODELS
    architecture = architecture.lower()
    if architecture == 'stylewavegan':
        log_message('No need to update config for StyleWaveGAN architecture', rank, logger)
    elif architecture == 'stylegan2':
        config = update_config_for_SG2(architecture, config, rank, logger)
    elif architecture in ['stylegan3-t', 'stylegan3-r']:
        config = update_config_for_SG3(architecture, config, rank, logger)
    else:
        assert False
    return config
