import os
from copy import deepcopy

import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm

from metrics.inception import InceptionV3, RGB_255
from dataset import resize_img, prepare_valid_dataloader
from logger import log_message
from shared_utils import get_metric_batch_size, measure_time, postprocess_imgs, check_compilation_state, \
    to_data_parallel


# ----- Utils -----

def get_nimg(kimg, ds_size):
    if isinstance(kimg, str):
        if kimg == 'full':
            nimg = ds_size
        else:
            assert False, f'kimg={kimg} is not supported'
    else:
        nimg = kimg * 1000
    return nimg

def process_activations(preds):
    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if preds.shape[2] != 1 or preds.shape[3] != 1:
        preds = torch.nn.functional.adaptive_avg_pool2d(preds, output_size=(1, 1))
    preds = preds.squeeze(3).squeeze(2).cpu()
    return preds


def calculate_mu_sigma(all_activations):
    # Remove some zero values from the last incomplete batch
    all_activations = all_activations.to(dtype=torch.float64)
    all_activations = all_activations[torch.abs(all_activations).sum(1) > 0.5].numpy()
    mu = np.mean(all_activations, axis=0)
    sigma = np.cov(all_activations, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    score = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(score)


@measure_time
def compute_gen_statistics(nimg, img_size_yx, batch_size, device, data_format, force_fp32, resize_in_model, resize_method,
                           inception_model, inception_size_yx, inception_dim, G_model, G_inputs_sampler):
    G_model = deepcopy(G_model.eval())
    n_steps = nimg // batch_size + 1
    all_activations = torch.zeros((nimg + batch_size, inception_dim), device='cpu')
    start_idx = 0
    for _ in tqdm(range(n_steps), 'FID_gen'):
        with torch.no_grad():
            gen_z, gen_c = G_inputs_sampler(batch_size)
            # Note: not sure about noise_mode here
            gen_imgs = G_model(gen_z, gen_c, noise_mode='const', force_fp32=force_fp32)
            gen_imgs = postprocess_imgs(gen_imgs, src_range=(-1, 1), should_adjust_dynamic_range=True,
                                        to_numpy=(not resize_in_model))
            if not resize_in_model:
                gen_imgs = [resize_img(img, inception_size_yx, resize_method) for img in gen_imgs]
                gen_imgs = np.array(gen_imgs, dtype=np.uint8)
                gen_imgs = torch.from_numpy(gen_imgs).to(device)
            if data_format == 'NHWC':
                gen_imgs = gen_imgs.to(memory_format=torch.channels_last)
            with torch.autocast('cuda', enabled=(not force_fp32)):
                # Note: model can output activations from different layers, for now only 1 is used
                activations = inception_model(gen_imgs)[0]
        activations = process_activations(activations)
        all_activations[start_idx: (start_idx + batch_size)] = activations
        start_idx += batch_size
    mu, sigma = calculate_mu_sigma(all_activations)
    return mu, sigma


@measure_time
def compute_real_statistics(nimg, img_size_yx, batch_size, device, data_format, force_fp32, resize_in_model,
                            resize_method, inception_model, inception_size_yx, inception_dim, data_loader):
    data_loader_iter = iter(data_loader)
    n_steps = nimg // batch_size + 1
    all_activations = torch.zeros((nimg + batch_size, inception_dim), device='cpu')
    start_idx = 0
    for _ in tqdm(range(n_steps), 'FID_real'):
        real_imgs, real_labels = next(data_loader_iter)
        # Data from iterator is on cpu, RGB, CHW, UINT8
        assert 0 <= real_imgs.min() and real_imgs.max() <= 255
        assert real_imgs.dtype == torch.uint8 and real_imgs.shape[1] in [1, 3], \
            f'real_imgs.dtype={real_imgs.dtype}, real_imgs.shape={real_imgs.shape}'
        # NCHW => NHWC
        NCHW_to_NHWC_idxs = [0, 2, 3, 1]
        if resize_in_model:
            # Note: much faster without contiguous(). Is it safe to skip it?
            real_imgs = real_imgs.permute(NCHW_to_NHWC_idxs).contiguous()
        else:
            real_imgs = np.transpose(real_imgs.numpy(), NCHW_to_NHWC_idxs)
            real_imgs = [resize_img(img, inception_size_yx, resize_method) for img in real_imgs]
            real_imgs = np.array(real_imgs, dtype=np.uint8)
            real_imgs = torch.from_numpy(real_imgs)
        real_imgs = real_imgs.to(device)
        if data_format == 'NHWC':
            real_imgs = real_imgs.to(memory_format=torch.channels_last)
        n_samples = real_imgs.shape[0]
        with torch.no_grad():
            with torch.autocast('cuda', enabled=(not force_fp32)):
                # Note: model can output activations from different layers, for now only 1 is used
                activations = inception_model(real_imgs)[0]
        activations = process_activations(activations)
        all_activations[start_idx : (start_idx + n_samples)] = activations
        start_idx += n_samples
    mu, sigma = calculate_mu_sigma(all_activations)
    return mu, sigma


# ----- Final metric class -----

class FIDMetric(torch.nn.Module):
    def __init__(self, device, metric_config, config, ckpt_dir, rank=0, logger=None):
        super().__init__()
        # Note: Inception inputs are uint8/float32, RGB images in range [0, 255]
        self.device = device
        self.inception_block_dims = 3
        self.inception_size_yx = (299, 299)
        self.inception_dim = InceptionV3.BLOCK_DIM_BY_INDEX[self.inception_block_dims]
        self.inception_output_blocks = [self.inception_block_dims]
        assert len(self.inception_output_blocks) == 1, \
            f'{self.__class__.__name__} only supports 1 output block now'
        self.resize_in_model = metric_config['resize_in_model']
        self.resize_method = metric_config['resize_method']
        inception_model = InceptionV3(self.inception_output_blocks,
                                      input_mode=RGB_255,
                                      resize_input=self.resize_in_model,
                                      resize_method=self.resize_method,
                                      normalize_input=True,
                                      requires_grad=False,
                                      use_fid_inception=True).to(self.device).eval()
        self.data_format = metric_config['data_format']
        if self.data_format == 'NHWC':
            inception_model = inception_model.to(memory_format=torch.channels_last)
            print(f'Moved model for {self.__class__.__name__} to NHWC format')
        self.use_compilation = check_compilation_state(config['general_params']['use_compilation'], rank, logger)
        if self.use_compilation:
            print(f'Compiling model for {self.__class__.__name__}')
            inception_model = torch.compile(inception_model)
        self.use_data_parallel = metric_config.get('use_data_parallel', False)
        if self.use_data_parallel:
            inception_model, num_gpus = to_data_parallel(inception_model, config)
        else:
            num_gpus = 1
        self.inception_model = inception_model
        self.m_real, self.s_real = None, None
        self.nimg = get_nimg(metric_config['kimg'], metric_config['ds_size'])
        self.img_size_yx = None  # not used for now
        self.num_classes = config['general_params']['num_classes']
        # Distributed => use batch_size_per_gpu, DataParallel or default => batch_size
        metric_batch_size = metric_config.get('batch_size', None)
        if metric_batch_size is not None:
            batch_size = metric_batch_size
        else:
            mult = 2
            batch_size = get_metric_batch_size(mult=mult, config=config)
            message = f'batch_size is not provided for {self.__class__.__name__}, ' \
                      f'using {batch_size} (mult={mult} x model_batch_size)'
            log_message(message, rank, logger)
        self.batch_size = batch_size * num_gpus
        message = f'{self.__class__.__name__}: use_data_parallel={self.use_data_parallel}, num_gpus={num_gpus}, ' \
                  f'batch_size={self.batch_size}'
        log_message(message, rank, logger)
        # dataloader_config = config['dataloader_params']
        self.valid_dataloader_config = {
            'batch_size': self.batch_size,
            'pin_memory': True, #ataloader_config['pin_memory'],
            # In DDP use all available threads
            'num_workers': 'AUTO' # dataloader_config['num_workers'],
        }
        self.stats_fname = os.path.join(ckpt_dir, f'fid_stats_{self.nimg // 1000}k.pt')

    def save_real_stats(self):
        state_dict = {
            'm_real': torch.tensor(self.m_real, dtype=torch.float32, device='cpu'),
            's_real': torch.tensor(self.s_real, dtype=torch.float32, device='cpu')
        }
        torch.save(state_dict, self.stats_fname)
        print(f'Saved FID real stats to {self.stats_fname}')

    def load_real_stats(self):
        status = False
        if os.path.exists(self.stats_fname):
            state_dict = torch.load(self.stats_fname, weights_only=True)
            self.m_real = state_dict['m_real'].numpy()
            self.s_real = state_dict['s_real'].numpy()
            print(f'Loaded FID real stats from {self.stats_fname}')
            status = True
        return status

    def run(self, G_model, G_inputs_sampler, ds, force_fp32=False):
        torch.cuda.empty_cache()
        shared_kwargs = dict(nimg=self.nimg, img_size_yx=self.img_size_yx,
            batch_size=self.batch_size, device=self.device, data_format=self.data_format, force_fp32=force_fp32,
            resize_in_model=self.resize_in_model, resize_method=self.resize_method,
            inception_model=self.inception_model, inception_size_yx=self.inception_size_yx,
            inception_dim=self.inception_dim)
        if (self.m_real is None) and (self.s_real is None):
            status = self.load_real_stats()
            if not status:
                # Enable validation mode for dataset. For some data center crop can be used
                ds = deepcopy(ds)
                ds.enable_mode(False)
                data_loader = prepare_valid_dataloader(ds, self.valid_dataloader_config, logger=None)
                self.m_real, self.s_real = compute_real_statistics(data_loader=data_loader, **shared_kwargs)
                self.save_real_stats()
        self.m_gen, self.s_gen = compute_gen_statistics(G_model=G_model, G_inputs_sampler=G_inputs_sampler,
                                                        **shared_kwargs)
        fid_value = calculate_frechet_distance(self.m_real, self.s_real, self.m_gen, self.s_gen)
        torch.cuda.empty_cache()
        return fid_value
