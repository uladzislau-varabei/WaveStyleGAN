import os
import json

import numpy as np
import torch

from models.networks import prepare_G_model
from models.utils import GenInputsSampler
from metrics.utils import prepare_metric
from dataset import prepare_dataset
from shared_utils import read_yaml, to_data_parallel


def calculate_metric(metric_config, config_path, weights_path, force_fp32, num_iters):
    config = read_yaml(config_path)
    ds = prepare_dataset(config)
    metric_config['ds_size'] = len(ds)
    G_model = prepare_G_model(weights_path, config, device)
    use_data_parallel = metric_config['use_data_parallel']
    if use_data_parallel:
        # Note: batch size is adjusted inside metric
        G_model, _ = to_data_parallel(G_model, config)
    gen_inputs_sampler = GenInputsSampler(device, config)
    metric = prepare_metric(metric_config, config, device)
    values = []
    metric_type = metric_config['type']
    kimg = metric_config['kimg']
    resize_method = metric_config.get('resize_method', None)
    metric_info = f'force_fp32={force_fp32}, num_iters={num_iters}, kimg={kimg}, resize_method={resize_method}'
    for idx in range(num_iters):
        metric_value = metric.run(G_model=G_model, G_inputs_sampler=gen_inputs_sampler, ds=ds, force_fp32=force_fp32)
        print(f'{idx + 1}/{num_iters}) {metric_type} metric ({metric_info}): {metric_value:.3f}')
        values.append(metric_value)
    print(f'{metric_type} metric ({metric_info}): {values}')
    return metric_info, values


def get_values_stats(values):
    values = np.array(values, dtype=np.float32)
    min_value = values.min()
    max_value = values.max()
    mean_value = values.mean()
    std_value = values.std()
    stats_info = f'num={len(values)}, mean={mean_value:.3f}, std={std_value:.3f}, ' \
                 f'min_value={min_value:.3f}, max_value={max_value:.3f}'
    return stats_info


def get_metric_stats(result_dict):
    print('Metric stats:')
    for desc, values in result_dict.items():
        stats = f'{desc}:\n{get_values_stats(values)}\n-----'
        print(stats)



if __name__ == '__main__':
    device = torch.device('cuda:1')
    use_data_parallel = True

    config_name = 'FFHQ_v1.yaml'
    config_path = os.path.join('configs', config_name)
    weights_path = './G_ema_model.pt'

    run_single_metric = False
    if run_single_metric:
        force_fp32 = True
        num_iters = 2
        fid_resize_method = ['linear', 'area'][0]
        fid_metric_config = {
            'type': 'FID',
            'kimg': 3, # 70k images for FFHQ
            'data_format': 'NHWC',
            # No padding is needed for FFHQ, so just use resize instead of M transform
            'resize_in_model': True,
            'resize_method': fid_resize_method,
            'batch_size': 32,
            'use_data_parallel': use_data_parallel
        }
        calculate_metric(fid_metric_config, config_path=config_path, weights_path=weights_path,
                         force_fp32=force_fp32, num_iters=num_iters)

    run_multiple_iters = False
    if run_multiple_iters:
        metric_type = 'FID'
        num_iters = 5
        kimg = 70
        res_dict = {}
        idx = 0
        for force_fp32_mode in [True, False]:
            for resize_method in ['linear', 'area']:
                fid_metric_config = {
                    'type': metric_type,
                    'kimg': kimg,
                    'data_format': 'NHWC',
                    # No padding is needed for FFHQ, so just use resize instead of M transform
                    'resize_in_model': True,
                    'resize_method': resize_method,
                    'batch_size': 32,
                    'use_data_parallel': use_data_parallel
                }
                iter_res = calculate_metric(fid_metric_config, config_path=config_path, weights_path=weights_path,
                                            force_fp32=force_fp32_mode, num_iters=num_iters)
                metric_info, values = iter_res
                res_dict[metric_info] = values
                # Only to deal with unstable system
                idx += 1
                json_fname = config_name.rsplit('.', 1)[0] + f'_{metric_type}{kimg}k_idx{idx}.json'
                json_path = os.path.join('results', 'metrics', json_fname)
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                with open(json_path, 'w') as fp:
                    json.dump(res_dict, fp, indent=4)
        print(f'Final metrics dict:\n{res_dict}')
        json_fname = config_name.rsplit('.', 1)[0] + f'_{metric_type}{kimg}k.json'
        json_path = os.path.join('results', 'metrics', json_fname)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as fp:
            json.dump(res_dict, fp, indent=4)
        print(f'Saved metrics dict to {json_path}')

    metric_type = 'FID'
    kimg = 70
    json_fname = config_name.rsplit('.', 1)[0] + f'_{metric_type}{kimg}k.json'
    json_path = os.path.join('results', 'metrics', json_fname)
    with open(json_path, 'r') as fp:
        res_dict = json.load(fp)
    get_metric_stats(res_dict)

    print('\n\n--- Finished metrics calculation script ---')

"""
Runtime (FID, 70k, 1024x1024, bs=32, DP=True):
1. Linear + force_fp32=True
Real: 8m 13s
Gen:  10m 09s, 10m 51s, 10m 40s, 11m 22s, 10m 46s
---
2. Area + force_fp32=True
Real: 8m 27s
Gen: 10m 13s, 10m 26s, 10m 27s, 10m 32s, 10m 38s
---
3. Linear + force_fp32=False
Real: 8m 39s
Gen: 9m 39s, 9m 39s, 9m 46s, 9m 49s, 9m 47s
---
4. Area + force_fp32=False
Real: 8m 18s
Gen: 9m 26s, 9m 33s, 9m 36s, 10m 55s, 10m 11s
"""
