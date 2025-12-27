import os

from shared_utils import read_yaml
from trainer import ModelTrainer


if __name__  == '__main__':
    # config_name = 'FFHQ_v3.yaml'
    # config_name = 'Landscape.yaml'
    config_name = 'SG2_FFHQ.yaml'
    config_path = os.path.join('configs', config_name)
    config = read_yaml(config_path)
    run_training = True

    trainer = ModelTrainer(config, rank=0)
    if run_training:
        trainer.train()
    else:
        benchmark_kimg = 3
        trainer.benchmark(benchmark_kimg)