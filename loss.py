# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""
import copy

import numpy as np
import torch
import torch.nn.functional as F

# from torch_utils import training_stats
from models.ops import conv2d_gradfix
from models.ops import upfirdn2d
from shared_utils import report_stats


USE_TRAINING_STATS = False


def apply_scaler(x, scaler):
    return scaler.scale(x) if scaler is not None else x


#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg

    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        img = self.G.synthesis(ws, update_emas=update_emas)
        return img, ws

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['G_main', 'G_reg', 'G_both', 'D_main', 'D_reg', 'D_both']
        if self.pl_weight == 0:
            phase = {'G_reg': 'none', 'G_both': 'G_main'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'D_reg': 'none', 'D_both': 'D_main'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # G_main: Maximize logits for generated images.
        if phase in ['G_main', 'G_both']:
            with torch.autograd.profiler.record_function('G_main_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                if USE_TRAINING_STATS:
                    report_stats('Loss/scores/fake', gen_logits)
                    report_stats('Loss/signs/fake', gen_logits.sign())
                loss_G_main = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                if USE_TRAINING_STATS:
                    report_stats('Loss/G/loss', loss_G_main)
            with torch.autograd.profiler.record_function('G_main_backward'):
                loss_G_main.mean().mul(gain).backward()

        # G_pl: Apply path length regularization.
        if phase in ['G_reg', 'G_both']:
            with torch.autograd.profiler.record_function('G_pl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size])
                print(f'loss ws shape: {gen_ws.shape}')
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                if USE_TRAINING_STATS:
                    report_stats('Loss/pl_penalty', pl_penalty)
                loss_G_pl = pl_penalty * self.pl_weight
                if USE_TRAINING_STATS:
                    report_stats('Loss/G/reg', loss_G_pl)
            with torch.autograd.profiler.record_function('G_pl_backward'):
                loss_G_pl.mean().mul(gain).backward()

        # D_main: Minimize logits for generated images.
        loss_D_gen = 0
        if phase in ['D_main', 'D_both']:
            with torch.autograd.profiler.record_function('D_gen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                if USE_TRAINING_STATS:
                    report_stats('Loss/scores/fake', gen_logits)
                    report_stats('Loss/signs/fake', gen_logits.sign())
                loss_D_gen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('D_gen_backward'):
                loss_D_gen.mean().mul(gain).backward()

        # D_main: Maximize logits for real images.
        # D_r1: Apply R1 regularization.
        if phase in ['D_main', 'D_reg', 'D_both']:
            name = 'D_real' if phase == 'D_main' else 'D_r1' if phase == 'D_reg' else 'D_real_D_r1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(phase in ['D_reg', 'D_both'])
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                if USE_TRAINING_STATS:
                    report_stats('Loss/scores/real', real_logits)
                    report_stats('Loss/signs/real', real_logits.sign())

                loss_D_real = 0
                if phase in ['D_main', 'D_both']:
                    loss_D_real = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    if USE_TRAINING_STATS:
                        report_stats('Loss/D/loss', loss_D_gen + loss_D_real)

                loss_D_r1 = 0
                if phase in ['D_reg', 'D_both']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_D_r1 = r1_penalty * (self.r1_gamma / 2)
                    if USE_TRAINING_STATS:
                        report_stats('Loss/r1_penalty', r1_penalty)
                        report_stats('Loss/D/reg', loss_D_r1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_D_real + loss_D_r1).mean().mul(gain).backward()


def get_loss_fn(loss_name):
    loss_name = loss_name.lower()
    reduction = 'none'
    if loss_name == 'l1':
        loss_fn = torch.nn.L1Loss(reduction=reduction)
    elif loss_name == 'l2':
        loss_fn = torch.nn.MSELoss(reduction=reduction)
    elif loss_name == 'smooth_l1':
        beta = 1.0
        loss_fn = torch.nn.SmoothL1Loss(reduction=reduction, beta=beta)
    else:
        assert False, f'loss={loss_name} is not supported'
    return loss_fn


class StyleWaveGANLoss(torch.nn.Module):
    # Note: models are provided as inputs for forward
    def __init__(self, device, is_data_parallel, loss_config, config):
        # Note: maybe remove device from args and just always move class instance after initialization
        super().__init__()
        self.device = device
        self.is_data_parallel = is_data_parallel
        # Read config params
        self.r1_gamma            = loss_config['r1_gamma']  #.get('r1_gamma', 10)
        self.style_mixing_prob   = loss_config['style_mixing_prob']  #.get('style_mixing_prob', 0)
        self.pl_weight           = loss_config['pl_weight']  #.get('pl_weight', 0)
        self.pl_batch_shrink     = loss_config['pl_batch_shrink']  #.get('pl_batch_shrink', 2)
        self.pl_decay            = loss_config['pl_decay']  #.get('pl_decay', 0.01)
        self.pl_no_weight_grad   = loss_config['pl_no_weight_grad']  #.get('pl_no_weight_grad', False)
        self.pl_mean             = torch.zeros([], device=device)
        self.blur_init_sigma     = loss_config['blur_init_sigma']  #.get('blur_init_sigma', 0)
        self.blur_fade_kimg      = loss_config['blur_fade_kimg']  #.get('blur_fade_kimg', 0)
        self.G_reg_fade_kimg     = loss_config['G_reg_fade_kimg']
        self.use_projection      = loss_config['use_projection']
        self.use_clip_loss       = loss_config['use_clip_loss']
        self.clip_weight         = loss_config['clip_weight']
        if self.use_clip_loss:
            assert self.clip_weight > 0

    def run_G(self, G_model, z, c, update_emas=False, apply_idwt=True):
        if self.is_data_parallel:
            ws = G_model.module.mapping(z, c, update_emas=update_emas)
        else:
            ws = G_model.mapping(z, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob,
                    cutoff, torch.full_like(cutoff, ws.shape[1]))
                if self.is_data_parallel:
                    ws[:, cutoff:] = G_model.module.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
                else:
                    ws[:, cutoff:] = G_model.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        if self.is_data_parallel:
            img = G_model.module.synthesis(ws, update_emas=update_emas, apply_idwt=apply_idwt)
        else:
            img = G_model.synthesis(ws, update_emas=update_emas, apply_idwt=apply_idwt)
        return img, ws

    def run_D(self, D_model, augment_pipe, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if augment_pipe is not None:
            img = augment_pipe(img)
        logits = D_model(img, c, update_emas=update_emas)
        return logits
    def G_main_loss(self, G_model, D_model, augment_pipe, grad_scaler, phase, real_img, gen_z, gen_c, gain, blur_sigma):
        # G_main: Maximize logits for generated images.
        if phase in ['G_main', 'G_both']:
            with torch.autograd.profiler.record_function('G_main_forward'):
                gen_img, _gen_ws = self.run_G(G_model, gen_z, gen_c)
                gen_logits = self.run_D(D_model, augment_pipe, gen_img, gen_c, blur_sigma=blur_sigma)
                if USE_TRAINING_STATS:
                    report_stats('Loss/scores/fake', gen_logits)
                    report_stats('Loss/signs/fake', gen_logits.sign())
                if self.use_projection:
                    loss_G_main = -gen_logits
                else:
                    loss_G_main = F.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                if self.use_clip_loss:
                    loss_G_main += self.clip_weight * F.relu(torch.abs(gen_img) - 1).mean(dim=[1, 2, 3]).unsqueeze(1)
                if USE_TRAINING_STATS:
                    report_stats('Loss/G/loss', loss_G_main)
            with torch.autograd.profiler.record_function('G_main_backward'):
                apply_scaler(loss_G_main.mean().mul(gain), grad_scaler).backward()

    def G_reg_loss(self, G_model, grad_scaler, phase, gen_z, gen_c, gain):
        # G_pl: Apply path length regularization.
        if phase in ['G_reg', 'G_both']:
            with torch.autograd.profiler.record_function('G_pl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(G_model, gen_z[:batch_size], gen_c[:batch_size], apply_idwt=True)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws],
                                                   create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                if USE_TRAINING_STATS:
                    report_stats('Loss/pl_penalty', pl_penalty)
                loss_G_pl = pl_penalty * self.pl_weight
                if USE_TRAINING_STATS:
                    report_stats('Loss/G/reg', loss_G_pl)
            with torch.autograd.profiler.record_function('G_pl_backward'):
                apply_scaler(loss_G_pl.mean().mul(gain), grad_scaler).backward()

    def G_loss(self, G_model, D_model, augment_pipe, grad_scaler, phase, real_img, gen_z, gen_c, gain, blur_sigma):
        self.G_main_loss(G_model=G_model, D_model=D_model, augment_pipe=augment_pipe, grad_scaler=grad_scaler,
            phase=phase, real_img=real_img, gen_z=gen_z, gen_c=gen_c, gain=gain, blur_sigma=blur_sigma)
        self.G_reg_loss(G_model=G_model, grad_scaler=grad_scaler, phase=phase, gen_z=gen_z, gen_c=gen_c, gain=gain)

    def D_loss(self, G_model, D_model, augment_pipe, grad_scaler, phase, real_img, real_c, gen_z, gen_c, gain, blur_sigma):
        # return 0  # for debugging
        # D_main: Minimize logits for generated images.
        loss_D_gen = 0
        if phase in ['D_main', 'D_both']:
            with torch.autograd.profiler.record_function('D_gen_forward'):
                gen_img, _gen_ws = self.run_G(G_model, gen_z, gen_c, update_emas=True, apply_idwt=True)
                gen_logits = self.run_D(D_model, augment_pipe, gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                if USE_TRAINING_STATS:
                    report_stats('Loss/scores/fake', gen_logits)
                    report_stats('Loss/signs/fake', gen_logits.sign())
                if self.use_projection:
                    loss_D_gen = F.relu(torch.ones_like(gen_logits) + gen_logits)
                else:
                    loss_D_gen = F.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('D_gen_backward'):
                apply_scaler(loss_D_gen.mean().mul(gain), grad_scaler).backward()

        # D_main: Maximize logits for real images.
        # D_r1: Apply R1 regularization.
        if phase in ['D_main', 'D_reg', 'D_both']:
            name = 'D_real' if phase == 'D_main' else 'D_r1' if phase == 'D_reg' else 'D_real_D_r1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(phase in ['D_reg', 'D_both'])
                real_logits = self.run_D(D_model, augment_pipe, real_img_tmp, real_c, blur_sigma=blur_sigma)
                if USE_TRAINING_STATS:
                    report_stats('Loss/scores/real', real_logits)
                    report_stats('Loss/signs/real', real_logits.sign())

                loss_D_real = 0
                if phase in ['D_main', 'D_both']:
                    if self.use_projection:
                        loss_D_real = F.relu(torch.ones_like(real_logits) - real_logits)
                    else:
                        loss_D_real = F.softplus(-real_logits) # -log(sigmoid(real_logits))
                    if USE_TRAINING_STATS:
                        report_stats('Loss/D/loss', loss_D_gen + loss_D_real)

                loss_D_r1 = 0
                if phase in ['D_reg', 'D_both']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp],
                                                       create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_D_r1 = r1_penalty * (self.r1_gamma / 2)
                    if USE_TRAINING_STATS:
                        report_stats('Loss/r1_penalty', r1_penalty)
                        report_stats('Loss/D/reg', loss_D_r1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                apply_scaler((loss_D_real + loss_D_r1).mean().mul(gain), grad_scaler).backward()

    def forward(self, G_model, D_model, augment_pipe, grad_scaler, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['G_main', 'G_reg', 'G_both', 'D_main', 'D_reg', 'D_both']
        src_phase = copy.deepcopy(phase)
        G_phase_dict = {'G_reg': 'none', 'G_both': 'G_main'}
        D_phase_dict = {'D_reg': 'none', 'D_both': 'D_main'}
        if self.pl_weight == 0:
            phase = G_phase_dict.get(phase, phase)
        if self.r1_gamma == 0:
            phase = D_phase_dict.get(phase, phase)

        if self.G_reg_fade_kimg > 0:
            if cur_nimg < self.G_reg_fade_kimg * 1000:
                phase = G_phase_dict.get(phase, phase)

        if self.blur_fade_kimg > 0:
            base_blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1000), 0)
            blur_sigma = base_blur_sigma * self.blur_init_sigma
        else:
            blur_sigma = 0

        self.G_loss(G_model=G_model, D_model=D_model, augment_pipe=augment_pipe, grad_scaler=grad_scaler,
            phase=phase, real_img=real_img, gen_z=gen_z, gen_c=gen_c, gain=gain, blur_sigma=blur_sigma)
        self.D_loss(G_model=G_model, D_model=D_model, augment_pipe=augment_pipe, grad_scaler=grad_scaler,
            phase=phase, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, gain=gain, blur_sigma=blur_sigma)

        should_run_optimizer = phase != 'none'
        return should_run_optimizer

#----------------------------------------------------------------------------
