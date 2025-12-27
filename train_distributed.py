import datetime
import os
import tempfile

import torch

from trainer import ModelTrainer
from shared_utils import read_yaml


def subprocess_fn(rank, config, benchmark_mode, benchmark_kimg, num_gpus, temp_dir):
    # 1. Init torch.distributed
    if num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        # Default timeout is 10 minutes for NCCL and 30 minutes for other backend
        timeout = datetime.timedelta(seconds=7200)
        # timeout = None
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank,
                world_size=num_gpus, timeout=timeout)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank,
                world_size=num_gpus, timeout=timeout)

    # 2. Run trainer
    model_trainer = ModelTrainer(config, rank=rank)
    if benchmark_mode:
        model_trainer.benchmark(benchmark_kimg)
    else:
        model_trainer.train()


def launch_distributed_training(config, benchmark_mode, benchmark_kimg):
    gpus = config['training_params']['gpus']
    num_gpus = len(gpus) if gpus is not None else 1
    print(f'DistributedDataParallel: launching {num_gpus} processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if num_gpus == 1:
            subprocess_fn(rank=0, config=config, benchmark_mode=benchmark_mode, benchmark_kimg=benchmark_kimg,
                num_gpus=num_gpus, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn,
                args=(config, benchmark_mode, benchmark_kimg, num_gpus, temp_dir), nprocs=num_gpus)


if __name__  == '__main__':
    config_name = 'FFHQ_v3.yaml'
    # config_name = 'Landscape.yaml'
    config_name = 'SG2_FFHQ.yaml'
    config_path = os.path.join('configs', config_name)
    config = read_yaml(config_path)
    benchmark_mode = False
    benchmark_kimg = 3

    launch_distributed_training(config, benchmark_mode=benchmark_mode, benchmark_kimg=benchmark_kimg)
