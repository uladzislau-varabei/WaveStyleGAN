import torch

from metrics.frechet_inception_distance import FIDMetric
from logger import log_message


def prepare_metric(metric_config, config, device, ckpt_dir, rank=0, logger=None):
    metric_type = metric_config['type'].lower()
    skip = metric_config.get('skip', False)
    # TODO: implement evaluation of metrics for DDP
    if skip:
        log_message(f'Skipped metric={metric_type}', rank, logger)
        return None
    if metric_type =='FID'.lower():
        metric = FIDMetric(device, metric_config, config, ckpt_dir=ckpt_dir, rank=rank, logger=logger)
    else:
        assert False, f'Metric={metric_type} is not supported'
    return metric
