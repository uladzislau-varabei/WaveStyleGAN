import copy
import os
import time

import pandas as pd
import torch
from tqdm import tqdm

from models.config import update_config
from models.misc import print_module_summary_v2
from models.networks import build_G_model
from models.utils import GenInputsSampler
from export import export_to_onnx
from shared_utils import read_yaml, setup_module


def benchmarks_model(config, num_gpu_iters, num_cpu_iters, num_warmup_iters, batch_size, process_onnx=False):
    torch.cuda.empty_cache()
    config = update_config(config, rank=0)
    architecture = config['general_params']['architecture']
    results = {'architecture': architecture}
    if 'stylegan3' in architecture.lower():
        num_cpu_warmup_iters = 2
        num_cpu_iters = 2
    else:
        num_cpu_warmup_iters = num_warmup_iters
    device_gpu = 'cuda'
    device_cpu = 'cpu'

    G_inputs_sampler_gpu = GenInputsSampler(device_gpu, config)
    G_inputs_sampler_cpu = GenInputsSampler(device_cpu, config)

    G_model = build_G_model(config['models_params']['Generator'], config, rank=0).requires_grad_(False)
    G_model_gpu = copy.deepcopy(G_model.eval()).to(device_gpu)
    G_model_cpu = copy.deepcopy(G_model.eval()).to(device_cpu)

    with torch.no_grad():
        gen_z, gen_c = G_inputs_sampler_gpu(batch_size)
        G_outputs = G_model_gpu(gen_z, gen_c, force_fp32=True)
        G_outputs, G_summary, G_params, G_buffers = print_module_summary_v2(G_model_gpu, [gen_z, gen_c], num_top=15)
        results['params'] = G_params
        results['buffers'] = G_buffers

    # GPU benchmarks
    with torch.no_grad():
        # 1. Warmup
        for _ in tqdm(range(2 * num_warmup_iters), f'GPU warmup (arch={architecture})'):
            gen_z, gen_c = G_inputs_sampler_gpu(batch_size)
            _ = G_model_gpu(gen_z, gen_c, force_fp32=True)

        # 2. Benchmarks
        start_time = time.time()
        for _ in tqdm(range(num_gpu_iters), f'GPU inference (arch={architecture})'):
            gen_z, gen_c = G_inputs_sampler_gpu(batch_size)
            _ = G_model_gpu(gen_z, gen_c, force_fp32=True)
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        results['torch_gpu_avg (ms)'] = round(1000 * total_time / num_gpu_iters, 1)

    sleep_time = 3
    with tqdm(total=sleep_time, desc='Sleeping after GPU benchmarks', leave=False) as pbar:
        for _ in range(sleep_time):
            time.sleep(1)
            pbar.update(1)

    # CPU benchmarks
    with torch.no_grad():
        # 1. Warmup
        for _ in tqdm(range(num_cpu_warmup_iters), f'CPU warmup (arch={architecture})'):
            gen_z, gen_c = G_inputs_sampler_cpu(batch_size)
            _ = G_model_cpu(gen_z, gen_c, force_fp32=True)

        # 2. Benchmarks
        start_time = time.time()
        for _ in tqdm(range(num_cpu_iters), f'CPU inference (arch={architecture})'):
            gen_z, gen_c = G_inputs_sampler_cpu(batch_size)
            _ = G_model_cpu(gen_z, gen_c, force_fp32=True)
        total_time = time.time() - start_time
        results['torch_cpu_avg (ms)'] = round(1000 * total_time / num_cpu_iters, 1)

    # ONNX benchmarks
    if process_onnx:
        temp_onnx_dir = os.path.join('.', 'temp_onnx_dir')
        os.makedirs(temp_onnx_dir)
        onnx_model_path = os.path.join(temp_onnx_dir, f'G_model_{architecture}.onnx')
        status = export_to_onnx(G_model_gpu, gen_z=gen_z, gen_c=gen_c, config=config, onnx_model_path=onnx_model_path)
        # results['onnx_cpu_avg (ms)'] = -1
    else:
        # results['onnx_cpu_avg (ms)'] = -1
        pass

    return results


if __name__ == '__main__':
    print('Benchmarking architectures...')
    num_gpu_iters = 500
    num_cpu_iters = 100
    num_warmup_iters = 5
    batch_size = 1
    process_onnx = False  # export doesn't work for now
    shared_params = dict(
        num_gpu_iters=num_gpu_iters,
        num_cpu_iters=num_cpu_iters,
        num_warmup_iters=num_warmup_iters,
        batch_size=batch_size,
        process_onnx=process_onnx
    )
    setup_module(0, 0, 0, enable_benchmark=True)

    use_swave = True
    if use_swave:
        config_path_swave = os.path.join('configs', 'FFHQ_v1_best.yaml')
        config_swave = read_yaml(config_path_swave)
        results_swave = benchmarks_model(config_swave, **shared_params)
        print(results_swave)
    else:
        results_swave = None

    use_sg2 = True
    if use_sg2:
        config_path_sg2 = os.path.join('configs', 'SG2_FFHQ_paper.yaml')
        config_sg2 = read_yaml(config_path_sg2)
        results_sg2 = benchmarks_model(config_sg2, **shared_params)
        print(results_sg2)
    else:
        results_sg2 = None

    use_sg3t = True
    if use_sg3t:
        config_path_sg3t = os.path.join('configs', 'SG3T_FFHQ_paper.yaml')
        config_sg3t = read_yaml(config_path_sg3t)
        results_sg3t = benchmarks_model(config_sg3t, **shared_params)
        print(results_sg3t)
    else:
        results_sg3t = None

    use_sg3r = True
    if use_sg3r:
        config_path_sg3r = os.path.join('configs', 'SG3R_FFHQ_paper.yaml')
        config_sg3r = read_yaml(config_path_sg3r)
        results_sg3r = benchmarks_model(config_sg3r, **shared_params)
        print(results_sg3r)
    else:
        results_sg3r = None

    # Merge results
    all_results = [x for x in [results_swave, results_sg2, results_sg3t, results_sg3r] if x is not None]
    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)
        # See all the columns
        pd.options.display.max_columns = None
        pd.options.display.max_rows = None
        print(f'Benchmarks results:\n{results_df}')
    else:
        print('All configs are disabled')
    print('\n--- Finished architecture benchmarking ---')

    """
    resolution=1024, gpu=500, cpu=100
        architecture |    params |   buffers | torch_gpu_avg (ms) | torch_cpu_avg (ms)
    0   StyleWaveGAN |  23662647 |    701658 |               27.5 |              232.8
    1      StyleGAN2 |  30370060 |   2797104 |               18.5 |              702.5
    2    StyleGAN3-T |  22313167 |      2480 |               40.3 |            16278.5
    3    StyleGAN3-R |  15093151 |      5600 |               44.8 |            31312.5
    """
