"""
Key changes made by Claude 3.5 Sonnet to workaround Out-Of-Memory error:

- Added CUDA memory optimization settings at the start
- Converted model to half precision (FP16)
- Enabled attention slicing, CPU offloading, and gradient checkpointing if available
- Added automatic mixed precision (amp) context
- Reduced resolution if needed
- Reduced number of steps if memory is tight
- Added torch.no_grad() for inference
- Added strategic torch.cuda.empty_cache() calls
- Added error handling for OOM errors
- Converted input images to half precision

Additional steps if needed:

- Further reduce the max_resolution value (currently set to 512)
- Reduce the number of steps (currently limited to 20)
- Add model = model.eval() before inference to ensure training mode is off

"""
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


import random
import sys
from argparse import ArgumentParser

import json
import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
from k_diffusion import utils
from k_diffusion.sampling import default_noise_sampler, to_d, get_ancestral_step
from tqdm.auto import trange
import torch.nn.functional as F

# Add memory optimization settings
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config
from stable_diffusion.ldm.hfp import extract_high_freq_component, sobel_operator


def sample_euler_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    extra_args = extra_args or {}
    noise_sampler = noise_sampler or default_noise_sampler(x)
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        d = to_d(x, sigmas[i], denoised)
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x

def resize_image_to_resolution(input_image, resolution, reverse=True):
    width, height = input_image.size
    scale = resolution / min(width, height) if reverse else resolution / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    new_width = (new_width // 64) * 64
    new_height = (new_height // 64) * 64
    return ImageOps.fit(input_image, (new_width, new_height), method=Image.Resampling.LANCZOS)

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def _get_hfp_loss(self, x_pred, x_original, t, lambdas=[0.001]):
        fft_pred, sobel_pred = self._get_hf_map(x_pred)
        fft_original, sobel_original = self._get_hf_map(x_original.to(x_pred.device))
        loss = (F.mse_loss(fft_pred, fft_original, reduction='none') + F.mse_loss(sobel_pred, sobel_original, reduction='none')) / 2
        loss = torch.mean(torch.flatten(loss, 1), dim=1) * torch.exp(-lambdas[0] * t)
        return torch.mean(loss)

    def _get_hf_map(self, rgb):
        rgb = (rgb + 1) / 2
        rgb = rgb.to(torch.float32)
        assert rgb.dim() == 4, "Input must be 4D tensor"
        return extract_high_freq_component(rgb, cutoff=30), sobel_operator(rgb)
    
    def _pred_xstart_from_eps(self, x_t, t, epilson):
        sqrt_one_minus_at = self.inner_model.inner_model.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1).to(x_t.device)
        a_t = self.inner_model.inner_model.alphas_cumprod[t].view(-1, 1, 1, 1).to(x_t.device)
        return (x_t - sqrt_one_minus_at * epilson) / a_t.sqrt()

    def _get_eps(self, input, sigma, **kwargs):
        c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.inner_model.get_scalings(sigma)]
        return self.inner_model.get_eps(input * c_in, self.inner_model.sigma_to_t(sigma), **kwargs)

    def hf_guidance(self, z, sigma, cond, x_original):
        t = self.inner_model.sigma_to_t(sigma).long()
        z = z.requires_grad_(True)
        eps = self._get_eps(z, sigma, cond=cond)
        pred_x0 = self._pred_xstart_from_eps(x_t=z / sigma, t=t, epilson=eps)
        x_pred = torch.clamp(self.inner_model.inner_model.differentiable_decode_first_stage(pred_x0), min=-1.0, max=1.0)
        loss_hfp = self._get_hfp_loss(x_pred, x_original, t)
        print(loss_hfp)
        torch.autograd.set_detect_anomaly(True)
        with torch.autograd.detect_anomaly():
            grad_cond = torch.autograd.grad(loss_hfp.requires_grad_(True), [z])[0]
        z = z - grad_cond
        torch.cuda.empty_cache()
        return z
        
    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale, input_image=None):
        if input_image is not None:
            for _ in range(5):
                z = self.hf_guidance(z, sigma, cond, x_original=input_image)
        cfg_z = einops.repeat(z, "b ... -> (repeat b) ...", repeat=3)
        cfg_sigma = einops.repeat(sigma, "b ... -> (repeat b) ...", repeat=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], cond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_txt_cond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return 0.5 * (out_img_cond + out_txt_cond) + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_cond - out_txt_cond)


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    model = instantiate_from_config(config.model)
    pl_sd = torch.load(ckpt, map_location="cpu")
    if 'state_dict' in pl_sd:
        pl_sd = pl_sd['state_dict']
    m, u = model.load_state_dict(pl_sd, strict=False)
    print(m, u)
    # Convert model to half precision
    model = model.half()
    return model

def main():
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=640, type=int)
    parser.add_argument("--steps", default=20, type=int)
    parser.add_argument("--config", default="configs/promptfix.yaml", type=str)
    parser.add_argument("--ckpt", default="./checkpoints/promptfix.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--indir", default='./examples/validation', type=str)
    parser.add_argument("--outdir", default="validation_results/", type=str)
    parser.add_argument("--cfg-text", default=6.5, type=float)
    parser.add_argument("--cfg-image", default=1.25, type=float)
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--disable_hf_guidance", type=bool, default=True)
    parser.add_argument("--enable-flaw-prompt", type=bool, default=True)
    # Add these arguments to workaround OOM error
    max_resolution_default = 384  # 512
    parser.add_argument("--max-resolution", default=max_resolution_default, type=int, help="Maximum resolution for processing")
    max_steps_default = 10  # 15 # 20
    parser.add_argument("--max-steps", default=max_steps_default, type=int, help="Maximum number of steps")    
    args = parser.parse_args()
    SEP = 80*"="
    config = OmegaConf.load(args.config)
    os.makedirs(args.outdir, exist_ok=True)
    
    # Clear CUDA cache and reset GPU
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available!")

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.set_device(0)  # Explicitly set GPU device

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

    # Load model with optimizations
    try:
        model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
        model = model.half().eval().cuda()
    except RuntimeError as e:
        print("Error loading model to GPU:", e)
        print("Attempting to reset CUDA...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
        model = model.half().eval().cuda()

   
    # Enable memory optimizations
    if hasattr(model, 'enable_attention_slicing'):
        model.enable_attention_slicing()
    if hasattr(model, 'enable_sequential_cpu_offload'):
        model.enable_sequential_cpu_offload()
    if hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()

    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    seed = args.seed if args.seed is not None else random.randint(0, 100000)
    
    instruct_dic = json.load(open(os.path.join(args.indir, 'instructions.json')))

    
    for val_img_idx, image_path in enumerate(instruct_dic):
        print("image_path and prompts:")
        print(SEP)
        print(image_path, instruct_dic[image_path])
        
        try:
            # Process images in smaller resolution if needed
            input_image = Image.open(os.path.join(args.indir, image_path)).convert("RGB")
            # Reduce resolution if memory is an issue
            max_resolution = min(args.resolution, args.max_resolution)  # Limit maximum resolution
            input_image = resize_image_to_resolution(input_image, max_resolution, 'inpaint' not in image_path)
            input_image_pil = input_image
            
            with autocast("cuda"):  # Enable automatic mixed precision
                with torch.cuda.amp.autocast():  # Further optimize memory usage
                    cond = {}
                    inst_prompt, flaw_prompt = instruct_dic[image_path]
                    cond["c_crossattn"] = [model.get_learned_conditioning([inst_prompt, flaw_prompt] if args.enable_flaw_prompt else [inst_prompt])]
                    input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
                    input_image = rearrange(input_image, "h w c -> 1 c h w").to(next(model.parameters()).device)
                    
                    # Convert to half precision
                    input_image = input_image.half()
                    
                    cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

                    uncond = {
                        "c_crossattn": [torch.cat([null_token, null_token], 0) if args.enable_flaw_prompt else null_token],
                        "c_concat": [torch.zeros_like(cond["c_concat"][0])]
                    }

                    # Reduce number of steps if memory is tight
                    actual_steps = min(args.steps, args.max_steps) if 'inpaint' not in image_path else args.max_steps
                    sigmas = model_wrap.get_sigmas(actual_steps)

                    extra_args = {
                        "cond": cond,
                        "uncond": uncond,
                        "text_cfg_scale": args.cfg_text,
                        "image_cfg_scale": args.cfg_image,
                    }
                    
                    if not args.disable_hf_guidance:
                        extra_args["input_image"] = input_image

                    torch.manual_seed(seed)
                    print(f"input_image.shape: {input_image.shape}")
                    
                    # Clear cache before heavy operations
                    torch.cuda.empty_cache()
                    
                    _, skip_connect_hs = model.first_stage_model.encoder(input_image)
                    z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
                    
                    # Use no_grad for inference
                    with torch.no_grad():
                        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
                        x = model.decode_first_stage(z, skip_connect_hs=skip_connect_hs)
                    
                    x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
                    x = 255.0 * rearrange(x, "1 c h w -> h w c")
                    edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())

                    out_img_path = image_path.split('.')[0] + '.jpg'
                    print(f"Save images to {out_img_path}")
                    edited_image.save(os.path.join(args.outdir, f"{image_path.split('.')[0]}.jpg"))
                    input_image_pil.save(os.path.join(args.outdir, f"{image_path.split('.')[0]}_input.jpg"))
                    
                    # Clear cache after processing each image
                    torch.cuda.empty_cache()
                    
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                print(f"Skipping {image_path} due to Out-Of-Memory (OOM) error. Try reducing resolution or batch size.")
                continue
            else:
                raise e

if __name__ == "__main__":
    main()