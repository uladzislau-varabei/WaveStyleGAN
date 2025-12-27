import os
import psutil
import shutil
import time
from glob import glob

import torch
from tqdm import tqdm

from dataset import prepare_dataset, prepare_dataloader
from metrics.utils import prepare_metric
from models.config import AVAILABLE_MODELS, update_config
from models.ema import init_ema_model
from models.misc import copy_params_and_buffers, named_params_and_buffers,\
    get_num_params_and_buffers_message, print_module_summary_v2
from models.networks import build_G_model, build_D_model
from models.utils import prepare_optimizer, GenInputsSampler
from export import export_to_onnx
from logger import init_logger
from loss import StyleWaveGANLoss
from shared_utils import read_yaml, save_yaml, setup_module, format_time, create_dir, postprocess_imgs, \
    get_gpu_stats, cur_nimg_to_fname, get_total_batch_size, get_total_steps, get_ckpt_oaths, load_ckpt, \
    get_config_performance_params_message, check_compilation_state, check_freq_cond, fetch_real_data, \
    NUM_SKIPPED_BENCHMARK_STEPS, USE_GRAD_SCALER_FOR_FP16
from vis_utils import create_images_grid, add_title_to_image, save_img


class ModelTrainer:
    def __init__(self, config, random_seed=0, rank=0, benchmark_mode=False):
        # torch.autograd.set_detect_anomaly(True)
        self.init_logging(config)
        config = update_config(config, rank, self.logger)
        self.init_devices(config, rank)
        self.init_params(config)
        setup_module(random_seed + self.ckpt_nimg, num_gpus=self.num_gpus, rank=rank)
        self.init_data_loader(config, full_init=True)
        self.init_models(config)
        self.init_metrics(config)
        self.init_train_phases(config)
        if not benchmark_mode:
            self.load_modules_from_ckpt()
        if self.rank == 0:
            self.logger.info('Trainer prepared!')

    def init_logging(self, config):
        img_shape_yx = config['general_params']['img_shape_yxc'][:2]
        dataset_name = config['dataset_params']['csv_path'].split('/')[-1].split('.')[0]
        config_name = config.get('config_name', dataset_name)
        logs_params = config['logs_params']
        logs_dir_postfix = f'{config_name}_{img_shape_yx[0]}x{img_shape_yx[1]}'
        self.logs_dir = os.path.join(logs_params['dir'], logs_dir_postfix)
        os.makedirs(self.logs_dir, exist_ok=True)
        self.valid_fixed_imgs_dir = os.path.join(self.logs_dir, 'imgs', 'fixed')
        self.valid_random_imgs_dir = os.path.join(self.logs_dir, 'imgs', 'random')
        self.ckpts_dir = os.path.join(self.logs_dir, 'ckpts')
        # TODO: update, so that file_append_mode is set based on loaded checkpoint info
        self.logger = init_logger(self.logs_dir, file_append_mode=True)
        create_dir(self.logs_dir, self.logger)
        create_dir(self.valid_fixed_imgs_dir, self.logger)
        create_dir(self.valid_random_imgs_dir, self.logger)
        create_dir(self.ckpts_dir, self.logger)
        save_yaml(config, os.path.join(self.logs_dir, 'config.yaml'))
        # Set other params
        self.logs_freq_kimg = logs_params['logs_freq_kimg']
        self.imgs_freq_kimg = logs_params['imgs_freq_kimg']
        self.ckpts_freq_kimg = logs_params['ckpts_freq_kimg']
        self.metrics_freq_kimg = logs_params['metrics_freq_kimg']
        self.imgs_grid_ncols = logs_params['imgs_grid_ncols']
        self.imgs_grid_nrows = logs_params['imgs_grid_nrows']
        self.max_ckpts = logs_params['max_ckpts']

    def init_devices(self, config, rank):
        self.rank = rank
        self.is_distributed = torch.distributed.is_initialized()
        self.is_data_parallel = False # adjust later based on config
        gpus = config['training_params']['gpus']
        self.num_gpus = len(gpus) if gpus is not None else 1
        self.device_ids = None  # only set for data parallel
        self.world_size = 1  # adjust later based on config
        use_default_gpu = False
        if use_default_gpu:
            main_device = 'cuda'
        else:
            # Note: GPU1 works better, so make it the main GPU
            # Note: if only GPU is available choose specified slot
            # main_device = 'cuda:1'
            main_device = 'cuda:0'
        if self.is_distributed:
            if self.num_gpus > 1:
                self.device = torch.device('cuda', self.rank)
                self.world_size = self.num_gpus
                self.logger.info(f'Distributed training is enabled. '
                    f'Set GPU with rank={rank}, world_size={self.world_size}')
            else:
                self.logger.info(f'Distributed training enabled, but only 1 GPU is available')
                self.device = torch.device(main_device)
        else:
            if self.num_gpus > 1:
                self.device_ids = gpus if use_default_gpu else sorted(gpus)[::-1]
                self.is_data_parallel = True
                self.logger.info(f'Init data parallel mode, num_gpus={self.num_gpus}')
            self.device = torch.device(main_device)
            # self.ema_device = torch.device(main_device) # maybe later keep EMA model simple, no parallelization
        if not self.is_data_parallel:
            self.logger.info(f'Set CUDA to use GPU={0}')
            torch.cuda.set_device(0)

    def init_params(self, config):
        general_params = config['general_params']
        training_params = config['training_params']
        self.config = config
        self.models_params = config['models_params']
        self.img_shape_yxc = general_params['img_shape_yxc']
        self.data_format = training_params['data_format']
        self.start_resolution = general_params['start_resolution']
        self.target_resolution = general_params['target_resolution']
        self.num_fp16_res = general_params['num_fp16_res']
        self.use_grad_scalers = self.num_fp16_res > 0 and USE_GRAD_SCALER_FOR_FP16
        self.use_custom_conv2d_op = training_params['use_custom_conv2d_op']
        self.upfirdn2d_impl = training_params['upfirdn2d_impl']
        self.bias_act_impl = training_params['bias_act_impl']
        # Compilation can also be used for metrics with models
        self.use_compilation = check_compilation_state(general_params['use_compilation'], self.rank, self.logger)
        # Other params
        self.total_kimg = training_params['total_kimg']
        self.total_nimg = self.total_kimg * 1000
        self.batch_size_per_gpu = training_params['batch_size_per_gpu']
        self.batch_size = get_total_batch_size(batch_size_per_gpu=self.batch_size_per_gpu,
            gpus=self.device_ids)
        self.total_batch_size = self.world_size * self.batch_size
        self.total_steps = get_total_steps(self.total_kimg, self.batch_size)
        self.sema_kimg = config['ema_params']['sema_kimg']
        self.fixed_gen_z, self.fixed_gen_c = None, None # set values during random samples generation
        self.dwt_params = config['dwt_params']
        self.wavelet = self.dwt_params['wavelet'] if self.dwt_params is not None else None
        self.load_from_checkpoint = general_params['load_from_checkpoint']
        self.ckpt_nimg = 0
        self.ckpt_paths_dict = None
        if self.load_from_checkpoint:
            available_ckpts = sorted([p for p in glob(os.path.join(self.ckpts_dir, '*')) if os.path.isdir(p)])
            if len(available_ckpts) > 0:
                last_ckpt = available_ckpts[-1]
                self.ckpt_nimg = int(os.path.split(last_ckpt)[-1])
                if self.rank == 0:
                    self.logger.info(f'Loading checkpoint: {last_ckpt}. Number of processed images: {self.ckpt_nimg}')
                self.ckpt_paths_dict = get_ckpt_oaths(last_ckpt, self.use_grad_scalers)
            else:
                if self.rank == 0:
                    self.logger.info('No checkpoint to load. Training from scratch')

    def init_data_loader(self, config, full_init=True):
        self.ds = prepare_dataset(config)
        # Needed for metrics
        self.ds_size = len(self.ds.df)
        if full_init:
            # For faster debugging disable this init
            self.data_loader = prepare_dataloader(self.ds, config, rank=self.rank, logger=self.logger)
            self.data_loader_iterator = iter(self.data_loader)
        self.G_inputs_sampler = GenInputsSampler(self.device, config)

    def init_models(self, config):
        gen_z, gen_c = self.G_inputs_sampler(self.batch_size)
        models_params = config['models_params']
        architecture = config['general_params']['architecture']
        assert architecture.lower() in AVAILABLE_MODELS
        G_model = build_G_model(models_params['Generator'], config, rank=self.rank, logger=self.logger)
        G_ema_model = init_ema_model(G_model, config, rank=self.rank, logger=self.logger)
        D_model = build_D_model(models_params['Discriminator'], config, rank=self.rank, logger=self.logger)
        augment_pipe = None  # disable for now
        if self.use_compilation:
            G_model = torch.compile(G_model)
            G_ema_model = torch.compile(G_ema_model)
            D_model = torch.compile(D_model)
        if self.is_data_parallel:
            # Note: maybe later keep EMA model simple, no parallelization
            G_model = torch.nn.DataParallel(G_model, device_ids=self.device_ids)
            G_ema_model = torch.nn.DataParallel(G_ema_model, device_ids=self.device_ids)
            D_model = torch.nn.DataParallel(D_model, device_ids=self.device_ids)
            augment_pipe = None
        # Note: grads are enabled and disabled for different models based on training phase
        self.G_model = G_model.to(self.device).requires_grad_(False)
        self.G_ema_model = G_ema_model.to(self.device).requires_grad_(False)
        self.D_model = D_model.to(self.device).requires_grad_(False)
        self.augment_pipe = augment_pipe
        # Log outputs
        if self.rank == 0:
            # Just for the case. Grads should be disabled previously
            with torch.no_grad():
                G_outputs = self.G_model(gen_z, gen_c)
                self.logger.info(f'G_outputs shape: {G_outputs.shape}, device: {G_outputs.device}')
                D_outputs = self.D_model(G_outputs, None)
                self.logger.info(f'D_outputs shape: {D_outputs.shape}, device: {D_outputs.device}')
                # Updated summary
                G_outputs, G_summary, _, _ = print_module_summary_v2(self.G_model, [gen_z, gen_c], num_top=15)
                D_outputs, D_summary, _, _ = print_module_summary_v2(self.D_model, [G_outputs, gen_c], num_top=15)
                self.logger.info(f'Generator complexity: {get_num_params_and_buffers_message(self.G_model)}')
                self.logger.info(f'Discriminator complexity: {get_num_params_and_buffers_message(self.D_model)}')
        self.distribute_modules()
        # For benchmarks logging
        self.architecture = architecture
        G_params = models_params['Generator']
        self.G_architecture = G_params['Synthesis']['architecture']
        self.G_activation = G_params['Synthesis']['activation']
        self.mapping_num_layers = G_params['Mapping']['num_layers']
        D_params = models_params['Discriminator']
        D_projection_params = D_params['projection_params']
        self.D_architecture = D_params['architecture']
        self.D_activation = D_params['activation']
        self.D_conv_type = D_params['conv_type']
        self.use_ffc = D_params['ffc_params']['use_ffc']
        self.use_projection = D_projection_params['use_projection']
        self.projection_heads = D_projection_params.get('num_heads', None)
        self.projection_head_architecture = D_projection_params.get('head_architecture', None)
        self.projection_mixing_out_max_channels = D_projection_params.get('mixing_out_max_channels', None)

    def distribute_modules(self):
        # Distribute params between models
        if self.is_distributed and self.num_gpus > 1:
            for module in [self.G_model, self.G_ema_model, self.D_model, self.augment_pipe]:
                if module is not None:
                    for name, param in named_params_and_buffers(module):
                        try:
                            torch.distributed.broadcast(param, src=0)
                        except Exception as e:
                            self.logger.error(f'Distributing error: {e}.\nParam={name}, shape={param.shape}')
                            assert False
            if self.rank == 0:
                self.logger.info(f'Distributed params abd buffers')

    def init_train_phases(self, config):
        # 1. Init loss
        def parse_reg_interval(x):
            if isinstance(x, int):
                if x < 0:
                    x = None
            assert x is None or (isinstance(x, int) and x > 0)
            return x

        loss_config = config['loss_params']
        self.G_pl_no_weight_grad = loss_config['pl_no_weight_grad']
        self.G_reg_interval = parse_reg_interval(loss_config['G_reg_interval'])
        self.D_reg_interval = parse_reg_interval(loss_config['D_reg_interval'])
        if self.rank == 0:
            self.logger.info(f"G_reg_interval: {self.G_reg_interval}, D_reg_interval: {self.D_reg_interval}")

        self.loss_fn = StyleWaveGANLoss(device=self.device, is_data_parallel=self.is_data_parallel,
            loss_config=loss_config, config=config)

        # 2. Init optimizers and scalers
        # Optimizers
        self.G_optimizer = prepare_optimizer(self.G_model, config['optimizers_params']['Generator'],
            reg_interval=self.G_reg_interval, rank=self.rank, logger=self.logger)
        self.D_optimizer = prepare_optimizer(self.D_model, config['optimizers_params']['Discriminator'],
            reg_interval=self.D_reg_interval, rank=self.rank, logger=self.logger)
        # Scalers
        self.G_scaler = torch.amp.GradScaler('cuda') if self.use_grad_scalers else None
        self.D_scaler = torch.amp.GradScaler('cuda') if self.use_grad_scalers else None
        # Ckpts dirs
        self.ckpts_dirs = []

        # 3. Build train phases
        def build_phase_dict(name, interval):
            return {
                'name': name,
                'interval': interval,
                # 'start_event': torch.cuda.Event(enable_timing=True) if self.rank == 0 else None,
                # 'end_event': torch.cuda.Event(enable_timing=True) if self.rank == 0 else None
                # Use separate event for each call
                'use_cuda_events': self.rank == 0,
                'start_events': [] if self.rank == 0 else None,
                'end_events': [] if self.rank == 0 else None
            }

        # Phases: G ([G_both] or [G_main, G_reg]), D ([D_both] or [D_main, D_reg])
        phases = []
        # G phases
        if self.G_reg_interval is None:
            phases.append(build_phase_dict('G_both', 1))
        else:
            phases.append(build_phase_dict('G_main', 1))
            phases.append(build_phase_dict('G_reg', self.G_reg_interval))
        # D phases
        if self.D_reg_interval is None:
            phases.append(build_phase_dict('D_both', 1))
        else:
            phases.append(build_phase_dict('D_main', 1))
            phases.append(build_phase_dict('D_reg', self.D_reg_interval))
        self.train_phases = phases

    def init_metrics(self, config):
        self.metrics = {}
        metrics_params = config['metrics_params']
        for v in metrics_params.values():
            metric_config = v
            # For automatic setting of number of images when datasets are small
            metric_config['ds_size'] = self.ds_size
            metric = prepare_metric(metric_config, config, device=self.device, ckpt_dir=self.ckpts_dir,
                                    rank=self.rank, logger=self.logger)
            if metric is not None:
                self.metrics[metric_config['type']] = metric

    def load_modules_from_ckpt(self):
        if self.load_from_checkpoint and self.ckpt_paths_dict is not None:
            self.G_model = load_ckpt(self.G_model, self.ckpt_paths_dict, 'G_model')
            self.G_ema_model = load_ckpt(self.G_ema_model, self.ckpt_paths_dict, 'G_ema_model')
            self.D_model = load_ckpt(self.D_model, self.ckpt_paths_dict, 'D_model')
            self.G_optimizer = load_ckpt(self.G_optimizer, self.ckpt_paths_dict, 'G_optimizer')
            self.D_optimizer = load_ckpt(self.D_optimizer, self.ckpt_paths_dict, 'D_optimizer')
            if self.use_grad_scalers:
                self.G_scaler = load_ckpt(self.G_scaler, self.ckpt_paths_dict, 'G_scaler')
                self.D_scaler = load_ckpt(self.D_scaler, self.ckpt_paths_dict, 'D_scaler')
            ckpt_train_data = torch.load(self.ckpt_paths_dict['train_data'], weights_only=True)
            self.fixed_gen_z = ckpt_train_data['fixed_gen_z']
            self.fixed_gen_c = ckpt_train_data['fixed_gen_c']
            loss_pl_mean = ckpt_train_data['pl_mean']
            self.loss_fn.pl_mean = loss_pl_mean.to(self.loss_fn.device)
            self.logger.info(f'[rank={self.rank}] Successfully loaded checkpoint (nimgs={self.ckpt_nimg}) '
                             f'for models, optimizers, scalers (if AMP is used) and fixed gen inputs')
            self.distribute_modules()

    def process_training_phase(self, phase):
        pass

    def get_module(self, phase_name, mode):
        phase_mode = phase_name[0]
        assert phase_mode in ['G', 'D'], f'phase_mode={phase_mode} is not supported, phase_name={phase_name}'
        assert mode in ['model', 'optimizer', 'scaler'], f'mode={mode} is not supported'
        if mode == 'model':
            module = self.G_model if phase_mode == 'G' else self.D_model
        elif mode == 'optimizer':
            module = self.G_optimizer if phase_mode == 'G' else self.D_optimizer
        elif mode == 'scaler':
            module = self.G_scaler if phase_mode == 'G' else self.D_scaler
        else:
            assert False
        return module

    def update_models_grads_mode(self, phase_name, mode):
        # Similar to SG2 and SG3 training script
        # Explanation: https://discuss.pytorch.org/t/what-happend-if-i-dont-set-certain-parameters-with-requires-grad-false-but-exclude-them-in-optimizer-params/150374
        phase_mode = phase_name[0]
        assert phase_mode in ['G', 'D'], f'phase_mode={phase_mode} is not supported, phase_name={phase_name}'
        assert mode in [True, False]
        if phase_mode == 'G':
            self.G_model.requires_grad_(mode)
            # D grads are disabled for False mode at phase start, so only process True
            if mode is True:
                self.D_model.requires_grad_(False)
        else:
            # G grads are disabled for False mode at phase start, so only process True
            if mode is True:
                self.G_model.requires_grad_(False)
            self.D_model.requires_grad_(mode)
        if self.augment_pipe is not None:
            self.augment_pipe.requires_grad_(mode)

    def process_grads(self, module):
        params = [param for param in module.parameters() if param.grad is not None]
        if len(params) > 0:
            flat = torch.cat([param.grad.flatten() for param in params])
            if self.num_gpus > 1 and self.is_distributed:
                torch.distributed.all_reduce(flat)
                flat /= self.num_gpus
            torch.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
            grads = flat.split([param.numel() for param in params])
            for param, grad in zip(params, grads):
                param.grad = grad.reshape(param.shape)

    def run_optimizer(self, phase_name, optimizer, scaler):
        with torch.autograd.profiler.record_function(f'{phase_name}_opt'):
            model = self.get_module(phase_name, 'model')
            self.process_grads(model)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

    def run_metrics(self, G_model, cur_nimg, force_fp32=False):
        kimg = round(cur_nimg / 1000)
        nimg_info = f'({kimg}k images)'
        metrics_values = {}
        for metric_name, metric_fn in self.metrics.items():
            metric_value = metric_fn.run(G_model, self.G_inputs_sampler, self.ds, force_fp32=force_fp32)
            self.logger.info(f'{metric_name} {nimg_info}: {metric_value:.3f}')
            metrics_values[metric_name] = metric_value
        self.logger.info(f'Full metrics {nimg_info}: {metrics_values}')
        torch.cuda.empty_cache() # release GPU cache as metrics evaluation consumes too many resources

    def log_training_stats(self, start_time, cur_nimg, cur_tick, tick_start_time, tick_end_time, tick_start_nimg,
        maintenance_time):
        total_time = format_time(time.time() - start_time)
        tick_time = tick_end_time - tick_start_time
        kimg_time = tick_time / (cur_nimg - tick_start_nimg) * 1e3
        train_speed = (cur_nimg - tick_start_nimg) / tick_time
        maintenance_time = format_time(maintenance_time)
        kimg = cur_nimg / 1e3
        augment_p = float(self.augment_pipe.p.cpu()) if self.augment_pipe is not None else 0
        cpu_mem = psutil.Process(os.getpid()).memory_info().rss / 2 ** 30
        fields = []
        fields += [f"tick {cur_tick:d}"]
        fields += [f"kimg {kimg:.1f}"]
        fields += [f"time {total_time:s}"]
        fields += [f"sec/tick {tick_time:.1f}"]
        fields += [f"sec/kimg {kimg_time:.2f}"]
        fields += [f"img/sec {train_speed:.2f}"]
        fields += [f"maintenance {maintenance_time:s}"]
        fields += [f"augment {augment_p:.3f}"]
        fields += [f"RAM {cpu_mem:.2f} GB"]
        training_stats = ', '.join(fields)
        phase_time_dict = {}
        # Values are updated, so use a simple loop
        for phase_idx in range(len(self.train_phases)):
            phase = self.train_phases[phase_idx]
            name = phase['name']
            start_events = phase['start_events']
            end_events = phase['end_events']
            if (start_events is not None) and (end_events is not None):
                phase_time = 0
                for start_event, end_event in zip(start_events, end_events):
                    end_event.synchronize()
                    # Time is in milliseconds
                    phase_time += start_event.elapsed_time(end_event) / 1000
                phase_time_dict[name] = phase_time
                self.train_phases[phase_idx]['start_events'] = []
                self.train_phases[phase_idx]['end_events'] = []
        phase_time_stats = []
        if len(phase_time_dict) > 0:
            total_phases_time = sum(phase_time_dict.values())
            for phase, phase_time in phase_time_dict.items():
                phase_time_format = format_time(phase_time)
                phase_time_ratio = round(100 * phase_time / total_phases_time, 2)
                phase_time_stats += [f'{phase}: {phase_time_format} ({phase_time_ratio} %)']
            phase_time_stats = ', '.join(phase_time_stats)
        if self.rank == 0:
            self.logger.info(training_stats)
            if len(phase_time_stats) > 0:
                self.logger.info(phase_time_stats)
        gpu_stats = get_gpu_stats()
        if self.is_distributed:
            single_gpu_stats = gpu_stats[self.rank]
            self.logger.info(f'[rank={self.rank}] {single_gpu_stats}')
        else:
            for single_gpu_stats in gpu_stats:
                self.logger.info(single_gpu_stats)

    def save_ckpt(self, cur_nimg):
        def add_ckpt_dir(fname):
            return os.path.join(cur_ckpt_dir, fname)

        def get_state_dict(module):
            return module.module.state_dict() if self.is_data_parallel else module.state_dict()

        nimg_postfix = cur_nimg_to_fname(cur_nimg)
        cur_ckpt_dir = os.path.join(self.ckpts_dir, nimg_postfix)
        create_dir(cur_ckpt_dir, self.logger)
        # Models
        torch.save(get_state_dict(self.G_ema_model), add_ckpt_dir(f'G_ema_model_{nimg_postfix}.pt'))
        torch.save(get_state_dict(self.G_model), add_ckpt_dir(f'G_model_{nimg_postfix}.pt'))
        torch.save(get_state_dict(self.D_model), add_ckpt_dir(f'D_model_{nimg_postfix}.pt'))
        # Optimizers
        torch.save(self.G_optimizer.state_dict(), add_ckpt_dir(f'G_optimizer_{nimg_postfix}.pt'))
        torch.save(self.D_optimizer.state_dict(), add_ckpt_dir(f'D_optimizer_{nimg_postfix}.pt'))
        # Scalers
        if self.G_scaler is not None:
            torch.save(self.G_scaler.state_dict(), add_ckpt_dir(f'G_scaler_{nimg_postfix}.pt'))
        if self.D_scaler is not None:
            torch.save(self.D_scaler.state_dict(), add_ckpt_dir(f'D_scaler_{nimg_postfix}.pt'))
        # Fixed grid
        train_data_state_dict = {
            'fixed_gen_z': self.fixed_gen_z, 'fixed_gen_c': self.fixed_gen_c, 'pl_mean': self.loss_fn.pl_mean
        }
        torch.save(train_data_state_dict, add_ckpt_dir(f'train_data.pt'))
        # Add new ckpts dir and remove previous
        self.ckpts_dirs.append(cur_ckpt_dir)
        if len(self.ckpts_dirs) > self.max_ckpts:
            remove_ckpts_dirs = self.ckpts_dirs[:-self.max_ckpts]
            keep_ckpts_dirs = self.ckpts_dirs[-self.max_ckpts:]
            self.ckpts_dirs = keep_ckpts_dirs
            for remove_ckpt_dir in remove_ckpts_dirs:
                try:
                    shutil.rmtree(remove_ckpt_dir)
                except Exception as e:
                    self.logger.warning(f'Error e={e} with removing ckpt dir={remove_ckpt_dir}')

    def save_valid_images(self, cur_nimg):
        # 1. Set params
        use_pil_for_saving = True
        nimg_postfix = cur_nimg_to_fname(cur_nimg)
        nimgs = self.imgs_grid_nrows * self.imgs_grid_ncols
        fixed_grid_imgs_fname = os.path.join(self.valid_fixed_imgs_dir, f'fixed_imgs_grid_{nimg_postfix}.png')
        random_grid_imgs_fname = os.path.join(self.valid_random_imgs_dir, f'random_imgs_grid_{nimg_postfix}.png')
        # 2. Generate images
        random_gen_z, random_gen_c = self.G_inputs_sampler(nimgs)
        if self.fixed_gen_z is None and self.fixed_gen_c is None:
            self.fixed_gen_z, self.fixed_gen_c = self.G_inputs_sampler(nimgs)
        gen_z = torch.cat([self.fixed_gen_z, random_gen_z], dim=0)
        gen_c = torch.cat([self.fixed_gen_c, random_gen_c], dim=0)
        with torch.no_grad():
            valid_imgs = self.G_ema_model(gen_z, gen_c, noise_mode='const')
            # All images are in RGB now. Postprocess on GPU to reduce data transfer to CPU
            valid_imgs = postprocess_imgs(valid_imgs, src_range=(-1, 1), should_adjust_dynamic_range=True)
        fixed_valid_imgs = valid_imgs[:nimgs]
        random_valid_imgs = valid_imgs[nimgs:]
        # 3. Save fixed images
        fixed_grid = create_images_grid(fixed_valid_imgs, n_cols=self.imgs_grid_ncols, n_rows=self.imgs_grid_nrows)
        fixed_grid = add_title_to_image(fixed_grid, nimg_postfix)
        save_img(fixed_grid, fixed_grid_imgs_fname, use_pil=use_pil_for_saving)
        # 4. Save random images
        random_grid = create_images_grid(random_valid_imgs, n_cols=self.imgs_grid_ncols, n_rows=self.imgs_grid_nrows)
        random_grid = add_title_to_image(random_grid, nimg_postfix)
        save_img(random_grid, random_grid_imgs_fname, use_pil=use_pil_for_saving)

    def process_ema_model(self, cur_nimg, batch_idx):
        # 1. Update EMA params
        if self.is_data_parallel:
            self.G_ema_model.module.update(self.G_model.module, batch_idx)
        else:
            self.G_ema_model.update(self.G_model, batch_idx)
        # 2. Update Smooth EMA params
        if self.sema_kimg is not None:
            should_update_sema = check_freq_cond(cur_nimg, batch_idx, freq_kimg=self.sema_kimg,
                total_steps=self.total_steps, batch_size=self.batch_size, world_size=self.world_size)
            if should_update_sema and batch_idx > 1:
                if self.is_data_parallel:
                    copy_params_and_buffers(src_module=self.G_ema_model.module, dst_module=self.G_model.module)
                else:
                    copy_params_and_buffers(src_module=self.G_ema_model, dst_module=self.G_model)

    def get_train_init_values(self, benchmark_mode=False):
        if (self.load_from_checkpoint) and (self.ckpt_paths_dict is not None) and (not benchmark_mode):
            # Add more than one batch to avoid conditions for metrics, ckpts, etc.
            ckpt_nimg = self.ckpt_nimg + int(1.5 * self.total_batch_size)
            batch_idx = ckpt_nimg // self.total_batch_size
            cur_nimg = ckpt_nimg
            cur_tick = ckpt_nimg // (self.logs_freq_kimg * 1000)
            if self.rank == 0:
                self.logger.info(f'Restored training from checkpoint: '
                                 f'batch_idx={batch_idx}, cur_nimg={cur_nimg}, cur_tick={cur_tick}')
        else:
            batch_idx = 0
            cur_nimg = 0
            cur_tick = 0
        return batch_idx, cur_nimg, cur_tick

    def benchmark(self, kimg):
        if kimg is None:
            kimg = 3
        self.loss_fn.G_reg_fade_kimg = 0  # disable fading of G regularization for benchmark
        self.logger = init_logger(self.logs_dir, add_file_handler=False) # only log info to console
        self.logs_freq_kimg = 1 # force logging every 1k imgs
        skipped_imgs = NUM_SKIPPED_BENCHMARK_STEPS * self.batch_size
        self.total_kimg = kimg
        self.total_nimg = int(kimg * 1000 / self.batch_size + 1) * self.batch_size + skipped_imgs
        self.total_steps = get_total_steps(kimg, self.batch_size) + NUM_SKIPPED_BENCHMARK_STEPS
        total_time = self.train(benchmark_mode=True)
        if self.rank == 0:
            speed = (self.total_nimg - skipped_imgs) / total_time
            kimg_time = 1000 / speed
            performance_params_message = get_config_performance_params_message(self)
            self.logger.info(f"Benchmarks performance params: \n{performance_params_message}\n")
            self.logger.info(f"Speed: {speed:.1f} imgs/sec. Time for 1k imgs: {kimg_time:.1f} sec.")
            gpu_stats = get_gpu_stats()
            if self.is_distributed:
                single_gpu_stats = gpu_stats[self.rank]
                self.logger.info(f'[rank={self.rank}] {single_gpu_stats}')
            for single_gpu_stats in gpu_stats:
                self.logger.info(single_gpu_stats)

    def train(self, benchmark_mode=False):
        # Note: generated data is different for each phase and real data is shared across all phases
        start_time = time.time()
        batch_idx, cur_nimg, cur_tick = self.get_train_init_values(benchmark_mode)
        # Reduce training duration if checkpoint is restored
        total_steps = self.total_steps - batch_idx
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        tick_end_time = tick_start_time
        maintenance_time = tick_start_time - start_time
        # self.run_metrics(self.G_ema_model, 0, force_fp32=False)
        # assert False, 'Manual stop'
        desc = 'Benchmark: batch_size={0}, total_steps={1}, total_kimg={2}, world_size={3}'.format(
            self.batch_size, self.total_steps, self.total_kimg, self.world_size)
        pbar = tqdm(desc=desc, total=self.total_steps) if benchmark_mode and self.rank == 0 else None
        if self.ckpt_nimg > 0 and self.rank == 0:
            self.logger.info('Training from checkpoint started')
        # Init here, as for benchmarks values are different
        freq_cond_kwargs = dict(total_steps=self.total_steps, batch_size=self.batch_size, world_size=self.world_size)
        while batch_idx < total_steps:
            if benchmark_mode:
                if batch_idx == NUM_SKIPPED_BENCHMARK_STEPS:
                    start_time = time.time()
                    tick_start_nimg = cur_nimg
                    tick_start_time = time.time()
            # 1. Get real data (shared across all phases within one step)
            real_imgs, real_c = fetch_real_data(self.data_loader_iterator, self.device)
            # 2. Process phase: loss and optimization
            for phase in self.train_phases:
                use_cuda_events = phase['use_cuda_events']
                if use_cuda_events:
                    start_event = torch.cuda.Event(enable_timing=True)
                    start_event.record(torch.cuda.current_stream(self.device))
                    end_event = torch.cuda.Event(enable_timing=True)
                else:
                    start_event = None
                    end_event = None
                phase_interval = phase['interval']
                phase_name = phase['name']
                if batch_idx % phase_interval != 0:
                    # Optionally skip phase
                    continue
                gen_z, gen_c = self.G_inputs_sampler(self.batch_size)
                # Prepare optimizer
                scaler = self.get_module(phase_name, 'scaler')
                optimizer = self.get_module(phase_name, 'optimizer')
                optimizer.zero_grad(set_to_none=True)
                # Process loss
                self.update_models_grads_mode(phase_name, True)
                should_run_optimizer = self.loss_fn(self.G_model, self.D_model, self.augment_pipe, scaler,
                    real_img=real_imgs, real_c=real_c, gen_z=gen_z, gen_c=gen_c,
                    phase=phase_name, gain=phase_interval, cur_nimg=cur_nimg)
                self.update_models_grads_mode(phase_name, False)
                # Process grads and update params
                if should_run_optimizer:
                    self.run_optimizer(phase_name, optimizer, scaler)
                # Finish phase
                if use_cuda_events:
                    end_event.record(torch.cuda.current_stream(self.device))
                    phase['start_events'].append(start_event)
                    phase['end_events'].append(end_event)
            # 3. Process EMA model
            self.process_ema_model(cur_nimg, batch_idx)
            # 4. Update states
            cur_nimg += self.total_batch_size
            batch_idx += 1
            if pbar is not None:
                pbar.update(1)
            # 5. Check tick and update summaries
            # Note: logs are fine during benchmark
            is_new_tick_started = check_freq_cond(cur_nimg, batch_idx, freq_kimg=self.logs_freq_kimg,
                **freq_cond_kwargs)
            if is_new_tick_started:
                tick_end_time = time.time()
                self.log_training_stats(start_time=start_time, cur_nimg=cur_nimg, cur_tick=cur_tick,
                    tick_start_time=tick_start_time, tick_end_time=tick_end_time,
                    tick_start_nimg=tick_start_nimg, maintenance_time=maintenance_time)
                # Update state.
                cur_tick += 1
                tick_start_nimg = cur_nimg
            # 6. Run metrics
            should_run_metrics = check_freq_cond(cur_nimg, batch_idx, freq_kimg=self.metrics_freq_kimg,
                benchmark_mode=benchmark_mode, **freq_cond_kwargs)
            if should_run_metrics and batch_idx > 1 and self.rank == 0:
                self.run_metrics(self.G_ema_model, cur_nimg)
            # 7. Save checkpoints
            should_save_ckpt = check_freq_cond(cur_nimg, batch_idx, freq_kimg=self.ckpts_freq_kimg,
                benchmark_mode=benchmark_mode, **freq_cond_kwargs)
            if should_save_ckpt and self.rank == 0:
                self.save_ckpt(cur_nimg)
            # 8. Save valid images
            should_save_gen_imgs = check_freq_cond(cur_nimg, batch_idx, freq_kimg=self.imgs_freq_kimg,
                benchmark_mode=benchmark_mode, **freq_cond_kwargs)
            if should_save_gen_imgs and self.rank == 0:
                self.save_valid_images(cur_nimg)
            # 9. Update summaries again after maintenance tasks
            if is_new_tick_started:
                tick_start_time = time.time()
                maintenance_time = tick_start_time - tick_end_time
        if pbar is not None:
            pbar.close()
        total_time = time.time() - start_time
        if self.rank == 0:
            self.logger.info(f"\n\n--- Training finished in {format_time(total_time)} ---")
        return total_time

    def convert_to_onnx_model(self, model_name=None):
        if model_name is None:
            resolution_postfix = f'{self.img_shape_yxc[0]}x{self.img_shape_yxc[1]}'
            model_name = f'{self.architecture}_{resolution_postfix}.onnx'
        onnx_model_path = os.path.join(self.logs_dir, model_name)
        gen_z, gen_c = self.G_inputs_sampler(1)
        export_to_onnx(self.G_model, gen_z=gen_z, gen_c=gen_c, config=self.config, onnx_model_path=onnx_model_path)


if __name__ == '__main__':
    config_name = 'FFHQ_v3.yaml'
    # config_name = 'Landscape.yaml'
    # config_name = 'test_ffhq.yaml'
    # config_name = 'SG2_FFHQ.yaml'
    config_path = os.path.join('configs', config_name)
    config = read_yaml(config_path)
    trainer = ModelTrainer(config)
    # trainer.convert_to_onnx_model()
    # assert False, 'Manual stop'
    run_training = False
    if run_training:
        trainer.train()
    else:
        benchmark_kimg = 10
        trainer.benchmark(benchmark_kimg)
