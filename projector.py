import os
import time
from glob import glob

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from models.networks import prepare_G_model
from models.utils import GenInputsSampler
from wavelets.utils import extract_coeffs_from_channels, NCHW_FORMAT
from shared_utils import read_yaml, format_time, postprocess_imgs


# Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
SIZE_THR = 256

# ----- Utils -----

def load_target_img(img_path):
    img = cv2.imread(img_path)
    img = np.ascontiguousarray(img[..., ::-1]) # BGR => RGB
    return img


def load_feature_extractor(device):
    # Load VGG16 feature detector.
    local_path = os.path.join('pretrained_models', 'vgg16.pt')
    if os.path.exists(local_path):
        print(f'Using local pretrained model: {local_path}')
        vgg16_model = torch.jit.load(local_path, map_location=device).eval()
    else:
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        print(f'Downloading pretrained model from url: {url}')
        assert False, 'Downloading is not implemented'
        with dnnlib.util.open_url(url) as f:
            vgg16_model = torch.jit.load(f).eval().to(device)
    return vgg16_model


# ----- Projection -----

def compute_noise_loss(noise_buffers):
    reg_loss = 0.0
    for v in noise_buffers.values():
        noise = v[None, None, :, :] # must be [1, 1, H, W] for F.avg_pool2d()
        while True:
            reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
            reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
            if noise.shape[2] <= 8:
                break
            noise = F.avg_pool2d(noise, kernel_size=2)
    return reg_loss


def compute_wavelet_loss(G, target_imgs, gen_imgs):
    loss_weights = [0, 0.25, 0.25, 0.25]
    # loss_weights = [0.25, 0, 0, 0]
    assert len(loss_weights) == 4
    num_steps = 3
    target_LL, gen_LL = target_imgs, gen_imgs
    loss = 0
    for _ in range(num_steps):
        target_output = G.synthesis.output_dwt(target_LL)
        target_coeffs = extract_coeffs_from_channels(target_output, data_format=NCHW_FORMAT)
        target_LL = target_coeffs[0]
        gen_output = G.synthesis.output_dwt(gen_LL)
        gen_coeffs = extract_coeffs_from_channels(gen_output, data_format=NCHW_FORMAT)
        gen_LL = gen_coeffs[0]
        for t, g, w in zip(target_coeffs, gen_coeffs, loss_weights):
            loss += w * F.l1_loss(g, t, reduction='mean')
    return loss


def project(
    G,
    features_model,
    target,
    clip_imgs               = True,
    use_wavelet_loss        = False,
    num_steps               = 1000,
    w_avg_samples           = 10000,
    initial_learning_rate   = 0.1,
    initial_noise_factor    = 0.05,
    lr_rampdown_length      = 0.25,
    lr_rampup_length        = 0.05,
    noise_ramp_length       = 0.75,
    regularize_noise_weight = 1e5,
    device                  = None,
    config                  = None
):
    target_shape = tuple(target.shape)
    resolution_scale = 2 if G.synthesis.use_wavelet else 1
    G_target_y = G.synthesis.img_target_resolution * resolution_scale
    G_target_x = G.synthesis.img_target_resolution * resolution_scale
    G_target_shape = (G.synthesis.img_channels, G_target_y, G_target_x)
    if target_shape[0] < G_target_y and target_shape[1] < G_target_y:
        target = Image.fromarray(target)
        target = target.resize((G_target_x, G_target_y), resample=Image.BICUBIC)
        target = np.array(target)
        print(f'Resized from {target_shape[:2]} to {G_target_shape[1:]}')
    target = torch.tensor(target, dtype=torch.uint8).permute(2, 0, 1)  # HWC => CHW
    target_shape = tuple(target.shape)
    assert target_shape == G_target_shape, f'Wrong projection shape: img={target_shape}, G={G_target_shape}'

    # Compute w stats.
    print(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    gen_inputs_sampler = GenInputsSampler(device, config)
    z_samples, _ = gen_inputs_sampler(w_avg_samples)
    w_samples = G.mapping(z_samples, None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_buffers = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > SIZE_THR:
        target_images = F.interpolate(target_images, size=(SIZE_THR, SIZE_THR), mode='area')
    target_features = features_model(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_buffers.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_buffers.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const', apply_idwt=True, force_fp32=True)

        synth_images = (synth_images + 1) * (255 / 2)
        if clip_imgs:
            synth_images = torch.clip(synth_images, 0, 255)
        if synth_images.shape[2] > SIZE_THR:
            synth_images = F.interpolate(synth_images, size=(SIZE_THR, SIZE_THR), mode='area')

        # Features for synth images.
        synth_features = features_model(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        noise_loss = compute_noise_loss(noise_buffers)

        # Wavelet regularization
        if use_wavelet_loss:
            wavelet_loss = compute_wavelet_loss(G, target_imgs=target_images, gen_imgs=synth_images)
        else:
            wavelet_loss = 0.0

        loss = dist + regularize_noise_weight * noise_loss + wavelet_loss

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f'Step {step+1:>4d}/{num_steps}: dist={dist:<4.2f}, noise_loss={noise_loss:<4.2f}, '
              f'wavelet_loss={wavelet_loss:<4.2f}, total_loss={float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_buffers.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, G.mapping.num_ws, 1])


def run_projection(config_path, img_path, weights_path, device, use_wavelet_loss, clip_imgs, num_steps, initial_learning_rate):
    config = read_yaml(config_path)

    G_model = prepare_G_model(weights_path, config, device)
    features_model = load_feature_extractor(device)
    img = load_target_img(img_path)

    print('Ready for projection...')
    start_time = time.time()
    w_projected = project(G=G_model,
                          features_model=features_model,
                          target=img,
                          use_wavelet_loss=use_wavelet_loss,
                          clip_imgs=clip_imgs,
                          num_steps=num_steps,
                          initial_learning_rate=initial_learning_rate,
                          device=device,
                          config=config)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start_time
    speed = num_steps / total_time
    print(f'Projection finished in {format_time(total_time)}. Speed: {speed:.2f} steps/sec')

    # w_projected: [num_steps, num_ws, w_dim]
    final_ws = w_projected[-1][None, ...]
    img_last_level_folder = os.path.split(os.path.split(img_path)[0])[1]
    dst_folder = os.path.join('results', 'projection', config_name.rsplit('.', 1)[0], img_last_level_folder)
    os.makedirs(dst_folder, exist_ok=True)
    projected_img = G_model.synthesis(final_ws, noise_mode='const', apply_idwt=True)
    projected_img = postprocess_imgs(projected_img, src_range=(-1, 1))[0]
    img_fname = os.path.split(img_path)[-1].rsplit('.', 1)[0] + f'_clip{clip_imgs}_steps{num_steps}.png'
    dst_img_path = os.path.join(dst_folder, img_fname)
    if img.shape != projected_img.shape:
        size_xy = (projected_img.shape[1], projected_img.shape[0])
        img = np.array(Image.fromarray(img).resize(size_xy, resample=Image.BICUBIC))
    stacked_img = np.hstack([img, projected_img])[..., ::-1]
    status = cv2.imwrite(dst_img_path, stacked_img)
    print(f'Saved projected image (status={status}) to: {dst_img_path}')


def run_projection_multiple_data(config_path, all_img_paths, weights_path, device, use_wavelet_loss, clip_imgs_params, num_steps,
                                 initial_learning_rate):
    start_time = time.time()
    num_iters = 0
    if not isinstance(clip_imgs_params, list):
        assert isinstance(clip_imgs_params, bool)
        clip_imgs_params = [clip_imgs_params]
    for clip_imgs in clip_imgs_params:
        for p in all_img_paths:
            run_projection(config_path=config_path,
                           img_path=p,
                           weights_path=weights_path,
                           device=device,
                           use_wavelet_loss=use_wavelet_loss,
                           clip_imgs=clip_imgs,
                           num_steps=num_steps,
                           initial_learning_rate=initial_learning_rate)
            torch.cuda.empty_cache()
            num_iters += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    total_time = time.time() - start_time
    print(f'Finished processing all data (num_iters={num_iters}) in {format_time(total_time)}')


if __name__ == '__main__':
    device = torch.device('cuda:1')
    initial_learning_rate = 0.1 # default is 0.1
    num_steps = 500 # default is 1000
    clip_imgs = False
    use_wavelet_loss = False

    config_name = 'FFHQ_v1.yaml'
    config_path = os.path.join('configs', config_name)
    weights_path = '.../G_ema_model.pt'

    run_single_projection = True
    if run_single_projection:
        name = ['name'][0]
        img_path = os.path.join('projection_samples', 'Upd_celebrity_samples', f'{name}_0.png')
        run_projection(config_path=config_path,
                       img_path=img_path,
                       weights_path=weights_path,
                       device=device,
                       use_wavelet_loss=use_wavelet_loss,
                       clip_imgs=clip_imgs,
                       num_steps=num_steps,
                       initial_learning_rate=initial_learning_rate)

    run_multiple_projections = False
    if run_multiple_projections:
        paths1 = glob(os.path.join('projection_samples', 'dir1', '*.png'))
        paths2 = glob(os.path.join('projection_samples', 'dir2', '*.png'))
        all_imgs_paths = paths1 + paths2
        clip_imgs_params = [True, False]
        num_steps = 500
        run_projection_multiple_data(config_path=config_path,
                                     all_img_paths=all_imgs_paths,
                                     weights_path=weights_path,
                                     device=device,
                                     use_wavelet_loss=False,
                                     clip_imgs_params=clip_imgs_params,
                                     num_steps=num_steps,
                                     initial_learning_rate=initial_learning_rate)

    print('\n\n--- Finished processing projection script ---')
