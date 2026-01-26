"""
PCam Dataset - FGSM Attack + Guided-Diffusion (ImageNet Pretrained) Defense (ViT Classifier)
"""

"""
# 基本実行
python imagenet_fgsm_eval.py

# パラメータ指定
python imagenet_fgsm_eval.py --epsilon 0.03137 --start_t 80 --T_purify 50 --gpu 0
"""

import os
import sys
import argparse
import random
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import save_image, make_grid
from sklearn.metrics import confusion_matrix
from pathlib import Path
import numpy as np
from datetime import datetime
from tqdm.auto import tqdm

sys.path.insert(0, '/mnt/data1/gotou/kaggle/guided-diffusion')
from guided_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion


def parse_args():
    parser = argparse.ArgumentParser(description='PCam FGSM + Guided-Diffusion Defense (ViT)')
    parser.add_argument('--epsilon', type=float, default=8/255)
    parser.add_argument('--start_t', type=int, default=280)
    parser.add_argument('--T_purify', type=int, default=300)
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/vit/pcam/correct_samples_balanced_500_vit.pt')
    parser.add_argument('--diffusion_ckpt', type=str, 
                        default='/mnt/data1/gotou/kaggle/guided-diffusion/256x256_diffusion_uncond.pt')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/vit/classifiers/checkpoints/pcam/20260117_210505/best_vit_pcam.pth')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/vit/pcam/imagenet/fgsm/results')
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ViTClassifierWrapper(nn.Module):
    def __init__(self, classifier, mean, std, input_size=224):
        super().__init__()
        self.classifier = classifier
        self.input_size = input_size
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        return self.classifier((x - self.mean) / self.std)


class GuidedDiffusionPurifier(nn.Module):
    def __init__(self, diffusion_model, diffusion, device, start_t=80, T_purify=50, eta=0.0):
        super().__init__()
        self.model = diffusion_model
        self.diffusion = diffusion
        self.device = device
        self.start_t = start_t
        self.T_purify = T_purify
        self.eta = eta
    
    @torch.no_grad()
    def purify(self, x_pixel):
        b = x_pixel.size(0)
        original_size = x_pixel.shape[-2:]
        
        if original_size != (256, 256):
            x_pixel = F.interpolate(x_pixel, size=(256, 256), mode='bilinear', align_corners=False)
        
        x_diff = x_pixel * 2.0 - 1.0
        
        t = torch.full((b,), self.start_t, device=self.device, dtype=torch.long)
        noise = torch.randn_like(x_diff)
        x_t = self.diffusion.q_sample(x_diff, t, noise=noise)
        
        end_t = max(self.start_t - self.T_purify, 0)
        for i in range(self.start_t, end_t, -1):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            out = self.diffusion.p_mean_variance(self.model, x_t, t, clip_denoised=True, model_kwargs={})
            if i > 0:
                nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
                x_t = out["mean"] + nonzero_mask * (self.eta * torch.sqrt(out["variance"])) * torch.randn_like(x_t)
            else:
                x_t = out["mean"]
        
        x_purified = torch.clamp((torch.clamp(x_t, -1.0, 1.0) + 1.0) / 2.0, 0, 1)
        
        if original_size != (256, 256):
            x_purified = F.interpolate(x_purified, size=original_size, mode='bilinear', align_corners=False)
        
        return x_purified
    
    def forward(self, x):
        return self.purify(x)


class GuidedDiffusionDefenseWrapper(nn.Module):
    def __init__(self, purifier, classifier, mean, std, input_size=224):
        super().__init__()
        self.purifier = purifier
        self.classifier = classifier
        self.input_size = input_size
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        x_purified = self.purifier(x)
        if x_purified.shape[-1] != self.input_size:
            x_purified = F.interpolate(x_purified, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        return self.classifier((x_purified - self.mean) / self.std)


def fgsm_attack(model, x, y, epsilon, device):
    x = x.clone().detach().to(device).requires_grad_(True)
    y = y.clone().detach().to(device)
    outputs = model(x)
    loss = F.cross_entropy(outputs, y)
    model.zero_grad()
    loss.backward()
    x_adv = torch.clamp(x + epsilon * x.grad.data.sign(), 0.0, 1.0).detach()
    return x_adv


def load_cached_samples(cached_path):
    print(f"\nLoading cached samples from: {cached_path}")
    cached = torch.load(cached_path, map_location='cpu')
    return cached['x_test'], cached['y_test'], cached.get('classes', ['normal', 'tumor'])


def load_models(args, device):
    classifier = models.vit_b_16(weights=None)
    in_features = classifier.heads.head.in_features
    classifier.heads.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(in_features, 2))
    
    ckpt = torch.load(args.clf_ckpt, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        classifier.load_state_dict(ckpt['model_state_dict'])
    else:
        classifier.load_state_dict(ckpt)
    classifier = classifier.to(device).eval()
    
    model_config = model_and_diffusion_defaults()
    model_config.update({
        'attention_resolutions': '32,16,8',
        'class_cond': False,
        'diffusion_steps': 1000,
        'rescale_timesteps': True,
        'timestep_respacing': '1000',
        'image_size': 256,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 256,
        'num_head_channels': 64,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_fp16': False,
        'use_scale_shift_norm': True,
    })
    
    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(torch.load(args.diffusion_ckpt, map_location=device))
    model = model.to(device).eval()
    
    print(f"Loaded ViT classifier and Guided-Diffusion model")
    return classifier, model, diffusion


def get_predictions_and_accuracy(model, x, y, bs=32, device=None):
    if device is None:
        device = next(model.parameters()).device
    n_batches = (len(x) + bs - 1) // bs
    preds, correct = [], 0
    with torch.no_grad():
        for i in range(n_batches):
            x_batch = x[i*bs:(i+1)*bs].to(device)
            y_batch = y[i*bs:(i+1)*bs].to(device)
            outputs = model(x_batch)
            batch_preds = outputs.argmax(dim=1)
            preds.append(batch_preds.cpu())
            correct += (batch_preds == y_batch).sum().item()
    return torch.cat(preds).numpy(), correct / len(x)


def run_fgsm_attack(model, x_test, y_test, epsilon, device, batch_size=32):
    print(f"\nRunning FGSM attack with epsilon={epsilon:.4f}...")
    n_batches = (len(x_test) + batch_size - 1) // batch_size
    x_adv_list = []
    for i in tqdm(range(n_batches), desc="FGSM Attack"):
        x_batch = x_test[i*batch_size:(i+1)*batch_size].to(device)
        y_batch = y_test[i*batch_size:(i+1)*batch_size].to(device)
        x_adv_list.append(fgsm_attack(model, x_batch, y_batch, epsilon, device).cpu())
    return torch.cat(x_adv_list, dim=0)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    timestamp = datetime.now().strftime("%m%d%H%M")
    log_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    classifier, diffusion_model, diffusion = load_models(args, device)
    
    purifier = GuidedDiffusionPurifier(diffusion_model, diffusion, device, args.start_t, args.T_purify, args.eta)
    
    classifier_model = ViTClassifierWrapper(classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    defense_model = GuidedDiffusionDefenseWrapper(purifier, classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    
    x_test, y_test, classes = load_cached_samples(args.cached_samples)
    
    print(f"\n{'='*70}")
    print("PCam - FGSM + Guided-Diffusion (ImageNet) Defense (ViT)")
    print(f"{'='*70}")
    
    results = {}
    
    print("\n[1/4] Clean accuracy...")
    _, clean_acc = get_predictions_and_accuracy(classifier_model, x_test, y_test, args.batch_size, device)
    results['clean_acc_classifier'] = clean_acc
    
    print("\n[2/4] Clean + Diffusion...")
    _, clean_purified_acc = get_predictions_and_accuracy(defense_model, x_test, y_test, args.batch_size, device)
    results['clean_acc_with_diffusion'] = clean_purified_acc
    
    print("\n[3/4] FGSM attack...")
    start_time = time.time()
    x_adv = run_fgsm_attack(classifier_model, x_test, y_test, args.epsilon, device, args.batch_size)
    attack_time = time.time() - start_time
    
    _, adv_acc_no_defense = get_predictions_and_accuracy(classifier_model, x_adv, y_test, args.batch_size, device)
    results['adv_acc_no_defense'] = adv_acc_no_defense
    results['attack_time'] = attack_time
    
    print("\n[4/4] Adversarial + Diffusion...")
    _, adv_defended_acc = get_predictions_and_accuracy(defense_model, x_adv, y_test, args.batch_size, device)
    results['adv_acc_with_diffusion'] = adv_defended_acc
    results['defense_improvement'] = adv_defended_acc - adv_acc_no_defense
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Clean: {clean_acc:.4f} | Clean+Diffusion: {clean_purified_acc:.4f}")
    print(f"FGSM: {adv_acc_no_defense:.4f} | FGSM+Diffusion: {adv_defended_acc:.4f}")
    print(f"Defense improvement: {results['defense_improvement']:+.4f}")
    
    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        json.dump({'dataset': 'PCam', 'classifier': 'ViT-B/16', 'defense': 'Guided-Diffusion', 'args': vars(args), **results}, f, indent=2)
    
    print(f"\n✅ Results saved to: {log_dir}")


if __name__ == '__main__':
    main()
