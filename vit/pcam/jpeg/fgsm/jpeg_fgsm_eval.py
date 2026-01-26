"""
PCam Dataset - FGSM Attack + JPEG Compression Defense (ViT Classifier)
"""

import os
import sys
import io
import argparse
import random
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import save_image, make_grid
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='PCam FGSM + JPEG Defense (ViT)')
    parser.add_argument('--epsilon', type=float, default=8/255)
    parser.add_argument('--quality', type=int, default=11)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/vit/pcam/correct_samples_balanced_500_vit.pt')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/vit/classifiers/checkpoints/pcam/20260117_210505/best_vit_pcam.pth')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/vit/pcam/jpeg/fgsm/results')
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ViTClassifierWrapper(nn.Module):
    def __init__(self, classifier, mean, std):
        super().__init__()
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        return self.classifier((x - self.mean) / self.std)


class JPEGDefense(nn.Module):
    def __init__(self, quality=11):
        super().__init__()
        self.quality = quality
    
    def compress_single(self, img_tensor):
        img = img_tensor.detach().clamp(0, 1).cpu()
        arr = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        pil = Image.fromarray(arr)
        buf = io.BytesIO()
        pil.save(buf, format='JPEG', quality=self.quality, subsampling=0, optimize=True)
        buf.seek(0)
        pil_j = Image.open(buf).convert('RGB')
        arr_j = np.array(pil_j).astype(np.float32) / 255.0
        return torch.from_numpy(arr_j).permute(2, 0, 1)
    
    def forward(self, x):
        device = x.device
        return torch.stack([self.compress_single(x[i]) for i in range(x.size(0))], dim=0).to(device)


class JPEGDefenseWrapper(nn.Module):
    def __init__(self, jpeg_defense, classifier, mean, std):
        super().__init__()
        self.jpeg_defense = jpeg_defense
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        x_compressed = self.jpeg_defense(x)
        return self.classifier((x_compressed - self.mean) / self.std)


def fgsm_attack(model, x, y, epsilon, device):
    x = x.clone().detach().to(device).requires_grad_(True)
    y = y.clone().detach().to(device)
    loss = F.cross_entropy(model(x), y)
    model.zero_grad()
    loss.backward()
    return torch.clamp(x + epsilon * x.grad.data.sign(), 0.0, 1.0).detach()


def load_classifier(args, device):
    classifier = models.vit_b_16(weights=None)
    classifier.heads.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(classifier.heads.head.in_features, 2))
    ckpt = torch.load(args.clf_ckpt, map_location=device)
    classifier.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    return classifier.to(device).eval()


def load_cached_samples(cached_path):
    cached = torch.load(cached_path, map_location='cpu')
    return cached['x_test'], cached['y_test'], cached.get('classes', ['normal', 'tumor'])


def get_predictions_and_accuracy(model, x, y, bs=32, device=None):
    if device is None:
        device = next(model.parameters()).device
    preds, correct = [], 0
    with torch.no_grad():
        for i in range((len(x) + bs - 1) // bs):
            x_batch = x[i*bs:(i+1)*bs].to(device)
            y_batch = y[i*bs:(i+1)*bs].to(device)
            batch_preds = model(x_batch).argmax(dim=1)
            preds.append(batch_preds.cpu())
            correct += (batch_preds == y_batch).sum().item()
    return torch.cat(preds).numpy(), correct / len(x)


def run_fgsm_attack(model, x_test, y_test, epsilon, device, batch_size):
    x_adv_list = []
    for i in tqdm(range((len(x_test) + batch_size - 1) // batch_size), desc="FGSM Attack"):
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
    
    timestamp = datetime.now().strftime("%m%d%H%M")
    log_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    classifier = load_classifier(args, device)
    jpeg_defense = JPEGDefense(quality=args.quality)
    
    classifier_model = ViTClassifierWrapper(classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    defense_model = JPEGDefenseWrapper(jpeg_defense, classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    
    x_test, y_test, classes = load_cached_samples(args.cached_samples)
    
    print(f"\n{'='*70}")
    print("PCam - FGSM + JPEG Defense (ViT)")
    print(f"{'='*70}")
    
    results = {}
    _, results['clean_acc_classifier'] = get_predictions_and_accuracy(classifier_model, x_test, y_test, args.batch_size, device)
    _, results['clean_acc_with_jpeg'] = get_predictions_and_accuracy(defense_model, x_test, y_test, args.batch_size, device)
    
    start_time = time.time()
    x_adv = run_fgsm_attack(classifier_model, x_test, y_test, args.epsilon, device, args.batch_size)
    results['attack_time'] = time.time() - start_time
    
    _, results['adv_acc_no_defense'] = get_predictions_and_accuracy(classifier_model, x_adv, y_test, args.batch_size, device)
    _, results['adv_acc_with_jpeg'] = get_predictions_and_accuracy(defense_model, x_adv, y_test, args.batch_size, device)
    results['defense_improvement'] = results['adv_acc_with_jpeg'] - results['adv_acc_no_defense']
    
    print(f"Clean: {results['clean_acc_classifier']:.4f} | Clean+JPEG: {results['clean_acc_with_jpeg']:.4f}")
    print(f"FGSM: {results['adv_acc_no_defense']:.4f} | FGSM+JPEG: {results['adv_acc_with_jpeg']:.4f}")
    print(f"Defense improvement: {results['defense_improvement']:+.4f}")
    
    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        json.dump({'dataset': 'PCam', 'classifier': 'ViT-B/16', 'defense': 'JPEG', 'args': vars(args), **results}, f, indent=2)
    
    print(f"\n✅ Results saved to: {log_dir}")


if __name__ == '__main__':
    main()
