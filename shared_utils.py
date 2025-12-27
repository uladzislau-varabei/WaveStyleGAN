import os
import platform
import time

import cv2
import yaml
from glob import glob
from packaging.version import Version, parse

import numpy as np
import torch

from logger import log_message


USE_LEGACY_BEHAVIOUR = False # FFHQ_v1 and Landscapes configs (disable for projection)
DEBUG_MODE = False # True
LOG_STRIDE_INFO = False
LOG_CONV_RESAMPLE_IMPL = False # True
ALLOW_CUDA_KERNEL_USAGE = True
USE_GRAD_SCALER_FOR_FP16 = False

# Looks like training is faster with NCHW on RTX 3090
NCHW_DATA_FORMAT = 'NCHW'
NHWC_DATA_FORMAT = 'NHWC'

DEFAULT_DATA_FORMAT = NCHW_DATA_FORMAT

USE_DEFAULT_DATA_FORMAT_BEHAVIOUR = True

NUM_SKIPPED_BENCHMARK_STEPS = 3 # 3 can be a good choice evan if model compilation is used

USE_NEW_BIAS_ACT = True

G_TANH_OUTPUT = False # only enable for debugging


# ----- General -----

def setup_module(random_seed, num_gpus, rank, enable_benchmark=True):
    # Note: enabling benchmarking can make the first steps extremely slow.
    # Also, clean max memory usage after some fist models steps
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = enable_benchmark   # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    # torch.backends.cudnn.deterministic = True           # Sometimes helps to disable the error
    # cv2.setNumThreads(0)

def read_yaml(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def save_yaml(data, path):
    with open(path, 'w') as fp:
        yaml.dump(data, fp, default_flow_style=False)


def read_txt(path):
    with open(path, "r") as fp:
        lines = fp.readlines()
    lines = [line.strip() for line in lines]
    return lines

def create_dir(dir, rank, logger=None):
    log_message(f'Creating dir {dir}...', rank, logger)
    os.makedirs(dir, exist_ok=True)


def is_running_on_linux():
    return platform.system() == 'Linux'


def is_auto_option(x):
    state = False
    if isinstance(x, str):
        if x.lower() == 'auto':
            state = True
    return state


def check_equal_lists(l1, l2):
    l1 = sorted([str(x) for x in l1])
    l2 = sorted([str(x) for x in l2] )
    assert len(l1) == len(l2), f'List1: {l1}, list2: {l2}'
    return all(v1 == v2 for v1, v2 in zip(l1, l2))


def check_compilation_state(use_compilation, rank=0, logger=None):
    # Double backward is not supported in PyTorch 2.5, so disable for models used in loss.
    # Also, some issues in forward, so just skip for GAN models
    torch_v = torch.__version__
    min_compilation_torch_v = "2.6"
    torch_ver_cond = parse(torch_v) >= Version(min_compilation_torch_v)
    os_cond = is_running_on_linux()
    if use_compilation and torch_ver_cond and os_cond:
        message = 'Enabling torch compilation'
        use_compilation = True
    else:
        running_os = platform.system()
        message = f'Disabling torch compilation: os={running_os}, torch_v={torch_v}, ' \
            f'min_compilation_torch_v={min_compilation_torch_v}'
        use_compilation = False
    log_message(message, rank, logger)
    return use_compilation


def cur_nimg_to_fname(cur_nimg):
    digits_in_number = 8  # Total number of training images is 25000k for resolution 1024
    fname = ('%0' + str(digits_in_number) + 'd') % (cur_nimg // 1000 * 1000)
    return fname


def get_distributed_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def get_total_batch_size(batch_size_per_gpu=None, gpus=None, config=None):
    if batch_size_per_gpu is not None:
        # Params are already provided
        pass
    elif config is not None:
        # Read data from config
        batch_size_per_gpu = config['training_params']['batch_size_per_gpu']
        gpus = config['training_params']['gpus']
    else:
        assert False, 'No data is provided'
    world_size = get_distributed_world_size()
    process_num_gpus = len(gpus) if (gpus is not None) and (world_size == 1) else 1
    return batch_size_per_gpu * process_num_gpus


def get_metric_batch_size(mult, batch_size_per_gpu=None, gpus=None, config=None):
    # Distributed => use batch_size_per_gpu, DataParallel or default => batch_size
    world_size = get_distributed_world_size()
    if world_size == 1:
        if batch_size_per_gpu is not None:
            batch_size = batch_size_per_gpu
        else:
            batch_size = config['training_params']['batch_size_per_gpu']
    else:
        batch_size = get_total_batch_size(batch_size_per_gpu, gpus, config)
    return int(mult * batch_size)


def get_total_steps(total_kimg, batch_size):
    # Note: number of GPUs is included in total batch size
    world_size = get_distributed_world_size()
    return int(np.ceil(total_kimg * 1000 / batch_size / world_size).astype(np.int32))


def check_freq_cond(cur_nimg, batch_idx, freq_kimg, total_steps, batch_size, world_size, benchmark_mode=False):
    prev_value = int((cur_nimg - world_size * batch_size) / (freq_kimg * 1000))
    cur_value = int(cur_nimg / (freq_kimg * 1000))
    is_multiple = cur_nimg % (freq_kimg * 1000) == 0
    is_last_batch = batch_idx == (total_steps - 1)
    # Extra checkpoint will be removed later and other tasks are good to run
    is_first_batch = batch_idx == 1
    cond = (is_first_batch or is_last_batch or (cur_value > prev_value) or is_multiple) and not benchmark_mode
    return cond


def get_ckpt_oaths(ckpt_dir, use_grad_scalers):
    def get_single_ckpt(pattern):
        paths = glob(os.path.join(ckpt_dir, pattern))
        assert len(paths) == 1, \
            f'Error with checkpoint pattern={pattern}. Found {len(paths)} files'
        return paths[0]

    ckpt_dict = {
        'G_model'       : get_single_ckpt('G_model*.pt'),
        'G_ema_model'   : get_single_ckpt('G_ema_model*.pt'),
        'D_model'       : get_single_ckpt('D_model*.pt'),
        'G_optimizer'   : get_single_ckpt('G_optimizer*.pt'),
        'D_optimizer'   : get_single_ckpt('D_optimizer*.pt'),
        'G_scaler'      : get_single_ckpt('G_scaler*.pt') if use_grad_scalers else None,
        'D_scaler'      : get_single_ckpt('D_scaler*pt') if use_grad_scalers else None,
        'train_data'    : get_single_ckpt('train_data.pt'),
    }
    return ckpt_dict


def get_config_performance_params_message(self):
    general_model_params = f'start_res={self.start_resolution}, target_res={self.target_resolution}, ' \
                           f'architecture={self.architecture}, wavelet={self.wavelet}'
    G_params = f'act={self.G_activation}, architecture={self.G_architecture}, ' \
               f'mapping_num_layers={self.mapping_num_layers}'
    D_params = f'act={self.D_activation}, architecture={self.D_architecture}, conv_type={self.D_conv_type}, ' \
               f'use_ffc={self.use_ffc}'
    projection_params = f'use_projection={self.use_projection}, num_heads={self.projection_heads}, ' \
                        f'architecture={self.projection_head_architecture}'
    loss_reg_params = f'G_reg_interval={self.G_reg_interval}, D_reg_interval={self.D_reg_interval}, ' \
                      f'G_pl_no_weight_grad={self.G_pl_no_weight_grad}'
    gpus_params = f'num_gpus={self.num_gpus}, DDP={self.is_distributed}, DP={self.is_data_parallel}'
    train_params = f'data_format={self.data_format}, num_fp16_res={self.num_fp16_res}, ' \
                   f'batch_size={self.batch_size}, use_compilation={self.use_compilation}'
    custom_ops_params = f'use_custom_conv2d_op={self.use_custom_conv2d_op}, ' \
                        f'upfirdn2n_impl={self.upfirdn2d_impl}, bias_act_impl={self.bias_act_impl}'
    titles = ['general_model_params', 'G_params', 'D_params', 'projection_params', 'loss_reg_params',
              'gpus_params', 'train_params', 'custom_ops_params']
    messages = [general_model_params, G_params, D_params, projection_params, loss_reg_params,
                gpus_params, train_params, custom_ops_params]
    performance_params_message = '\n'.join([f'{title}:\n{message}' for title, message in zip(titles, messages)])
    return performance_params_message


def to_device(x, device):
    return x.to(device=device, non_blocking=True)


def to_data_parallel(model, config):
    gpus = config['training_params']['gpus']
    num_gpus = len(gpus) if gpus is not None else 1
    if num_gpus > 1:
        # GPU1 is better
        device_ids = sorted(gpus)[::-1]
        print(f'Using data parallel with device_ids={device_ids}')
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        print(f'Data parallel is enabled but no multiple gpus in config')
    return model, num_gpus


def fetch_real_data(data_loader_iterator, device):
    real_imgs, real_c = next(data_loader_iterator)
    real_imgs = gpu_preprocess_real_imgs(to_device(real_imgs, device))
    real_c = to_device(real_c, device)
    return real_imgs, real_c


def load_ckpt(module, ckpt_paths_dict, name):
    module.load_state_dict(torch.load(ckpt_paths_dict[name], weights_only=True))
    return module


# ----- Time -----

def format_time(seconds):
    """Convert the seconds to human readable string with days, hours, minutes and seconds."""
    s = int(np.rint(seconds))
    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m {2:02}s".format(s // (60 * 60), (s // 60) % 60, s % 60)
    else:
        return "{0}d {1:02}h {2:02}m".format(s // (24 * 60 * 60), (s // (60 * 60)) % 24, (s // 60) % 60)


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        total_time = time.time() - start_time
        message = f'<{func.__name__}> completed in {format_time(total_time)}'
        print(message)
        return result
    return wrapper


def log_model_res_info(x, img, res, prefix):
    if LOG_STRIDE_INFO:
        x_stride = x.stride if x is not None else -1
        img_stride = img.stride if img is not None else -1
    else:
        x_stride = -1
        img_stride = -1
    if type(x) is tuple:
        x1, x2 = x
        x_shape = f'x1: {tuple(x1.shape)}, ' \
                  f'x2: {tuple(x2.shape) if isinstance(x2, torch.Tensor) else -1}'
    else:
        x_shape = tuple(x.shape) if x is not None else None
    img_shape = tuple(img.shape) if img is not None else None

    print(f'{prefix} res={res}: x.shape={x_shape}, img.shape={img_shape}, '
          f'x.stride={x_stride}, img.stride={img_stride}')


def check_input_imgs(imgs):
    eps = 1e-5
    imgs_min, imgs_max = imgs.min(), imgs.max()
    assert -1 - eps <= imgs_min and imgs_max <= 1 + eps, f'img_min={imgs_min}, img_max={imgs_max}'


def log_ffc_output_info(x, name):
    x_local, x_global = x[0], x[1]
    x_global_shape = tuple(x_global.shape) if isinstance(x_global, torch.Tensor) else -1
    x_global_dtype = x_global.dtype if isinstance(x_global, torch.Tensor) else -1
    print(f'{name}_local: shape={tuple(x_local.shape)} dtype={x_local.dtype}, '
          f'{name}_global: shape={x_global_shape}, dtype={x_global_dtype}')


def num_channels(res, channel_base, channel_max):
    return min(channel_base // res, channel_max)


def report_stats(name, value):
    pass


# ----- Images -----

def gpu_preprocess_real_imgs(imgs):
    # Reduce data transfer to GPU
    # Cast fo FP32 and scale to range [-1, 1]
    # Note: data format is updated inside model layers
    imgs = imgs.to(dtype=torch.float32) / 127.5 - 1.0
    return imgs


def adjust_dynamic_range(x, src_range, target_range):
    # Shift to the same zero point and then scale
    src_min, src_max = src_range
    target_min, target_max = target_range
    x = (x.to(dtype=torch.float32) - src_min) * ((target_max - target_min) / (src_max - src_min))
    x = torch.round(x).clamp(target_min, target_max)
    return x


def convert_NCHW_to_NHWC(imgs):
    return torch.permute(imgs, (0, 2, 3, 1))


def postprocess_imgs(imgs, src_range=None, should_adjust_dynamic_range=True, should_convert_BGR_to_RGB=False,
    to_numpy=True):
    assert isinstance(imgs, torch.Tensor)
    imgs = imgs.detach()
    # Note: wavelets are processed inside models. No additional postprocessing is required
    if should_adjust_dynamic_range:
        # Target dynamic range is always [0, 255] for images
        assert src_range is not None
        imgs = adjust_dynamic_range(imgs, src_range, (0, 255))
    # UINT8, NCHW -> NHWC
    imgs = convert_NCHW_to_NHWC(imgs.to(dtype=torch.uint8))
    assert not should_convert_BGR_to_RGB # old, was used with BGR training
    if should_convert_BGR_to_RGB:
        # BGR -> RGB
        imgs = imgs[..., ::-1].contiguous()
    # Final convert
    if to_numpy:
        imgs = imgs.cpu().numpy()
    return imgs


# ----- GPU stats -----

def get_gpu_stats_for_device(device):
    device_index = torch.cuda._get_device_index(device)
    name = f"GPU{device_index}"
    peak_ram = torch.cuda.max_memory_allocated(device) / 2 ** 30
    reserved_ram = torch.cuda.max_memory_reserved(device) / 2 ** 30
    core_usage = torch.cuda.utilization(device)
    memory_usage = torch.cuda.memory_usage(device)
    clock_rate = torch.cuda.clock_rate(device)
    temp = torch.cuda.temperature(device)
    power = torch.cuda.power_draw(device) / 1000
    stats = f"{name}: peak {peak_ram:>5.2f} GB, reserved {reserved_ram:>5.2f} GB, " \
            f"CHIP usage {core_usage:>3}%, MEM usage {memory_usage:>3}%, " \
            f"{clock_rate:>4} MHz, {temp} C, {power:>3.0f} W"
    torch.cuda.reset_peak_memory_stats(device)
    return stats


def get_gpu_stats(device=None):
    if device is None:
        all_devices = torch.cuda.device_count()
        stats = []
        for device_idx in range(all_devices):
            device = torch.cuda.device(f'cuda:{device_idx}')
            stats.append(get_gpu_stats_for_device(device))
    else:
        stats = [get_gpu_stats_for_device(device)]
    return stats
