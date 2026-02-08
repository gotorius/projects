"""
Defense-GAN Quick Visual Test - 数枚のサンプルで浄化を確認

このスクリプトは、Defense-GANが実際に敵対的画像を浄化できているか
数枚のサンプルで視覚的に確認するためのテストです。

実行例:
python gan_fgsm_quick_test.py --n_samples 4
python gan_fgsm_quick_test.py --n_samples 8 --epsilon 0.031 --rec_iters 300
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import save_image, make_grid
import numpy as np
from datetime import datetime

# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='Defense-GAN Quick Visual Test')
    
    parser.add_argument('--n_samples', type=int, default=4,
                        help='Number of samples to test')
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='FGSM perturbation epsilon')
    parser.add_argument('--rec_iters', type=int, default=20,
                        help='Number of reconstruction iterations (推奨: 10-50)')
    parser.add_argument('--rec_lr', type=float, default=0.01,
                        help='Learning rate for reconstruction')
    parser.add_argument('--rec_rr', type=int, default=1,
                        help='Number of random restarts (推奨: 1-2. 元画像から大きく離れないようにするため削減)')
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/pcam/ddpm/correct_samples_balanced_500.pt',
                        help='Path to cached correct samples')
    parser.add_argument('--gan_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/pcam/gan/checkpoints_v3/20251225_230534/checkpoint_epoch_0020.pth',
                        help='GAN checkpoint path')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/pcam/resnet/checkpoints/best_resnet50_pcam.pth',
                        help='Classifier checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/pcam/gan/fgsm/quick_test',
                        help='Output directory')
    parser.add_argument('--gpu', type=int, default=2,
                        help='GPU ID')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


# ========== 定数 ==========
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========== Self-Attention ==========
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.query = nn.utils.spectral_norm(self.query)
        self.key = nn.utils.spectral_norm(self.key)
        self.value = nn.utils.spectral_norm(self.value)
    
    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H * W)
        value = self.value(x).view(batch_size, -1, H * W)
        
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        return self.gamma * out + x


# ========== ResNet Blocks ==========
class ResBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in [self.conv1, self.conv2, self.shortcut]:
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        h = self.bn1(x)
        h = F.relu(h)
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = self.conv1(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.conv2(h)
        
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.shortcut(x)
        
        return h + x


# ========== Generator ==========
class GeneratorV3(nn.Module):
    def __init__(self, latent_dim=512, ngf=64, nc=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.init_size = 7
        
        self.fc = nn.Linear(latent_dim, ngf * 8 * self.init_size * self.init_size)
        
        self.block1 = ResBlockUp(ngf * 8, ngf * 8)
        self.block2 = ResBlockUp(ngf * 8, ngf * 4)
        self.block3 = ResBlockUp(ngf * 4, ngf * 2)
        self.attention = SelfAttention(ngf * 2)
        self.block4 = ResBlockUp(ngf * 2, ngf)
        self.block5 = ResBlockUp(ngf, ngf // 2)
        
        self.bn_out = nn.BatchNorm2d(ngf // 2)
        self.conv_out = nn.Conv2d(ngf // 2, nc, 3, 1, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.orthogonal_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
    
    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 512, self.init_size, self.init_size)
        
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.attention(h)
        h = self.block4(h)
        h = self.block5(h)
        
        h = self.bn_out(h)
        h = F.relu(h)
        h = self.conv_out(h)
        h = torch.tanh(h)
        
        return h


# ========== Defense-GAN ==========
class DefenseGAN:
    def __init__(self, generator, latent_dim=512, rec_iters=500, rec_lr=0.01, 
                 rec_rr=5, device='cuda'):
        self.generator = generator
        self.generator.eval()
        self.latent_dim = latent_dim
        self.rec_iters = rec_iters
        self.rec_lr = rec_lr
        self.rec_rr = rec_rr
        self.device = device
        
        for param in self.generator.parameters():
            param.requires_grad = False
    
    def _denormalize_from_classifier(self, x):
        """ImageNet正規化を解除"""
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        x = x * std + mean
        x = x.clamp(0, 1)
        return x * 2 - 1
    
    def _normalize_for_classifier(self, x):
        """GAN出力をImageNet正規化に変換"""
        x = (x + 1) / 2
        x = x.clamp(0, 1)
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        return (x - mean) / std
    
    def reconstruct(self, x):
        """バッチ画像を再構成"""
        batch_size = x.size(0)
        x_target = self._denormalize_from_classifier(x)
        
        best_z_list = []
        best_loss_list = [float('inf')] * batch_size
        
        for _ in range(batch_size):
            best_z_list.append(None)
        
        for r in range(self.rec_rr):
            z = torch.randn(batch_size, self.latent_dim, device=self.device, requires_grad=True)
            optimizer = torch.optim.Adam([z], lr=self.rec_lr)
            
            for _ in range(self.rec_iters):
                optimizer.zero_grad()
                x_gen = self.generator(z)
                loss_per_sample = F.mse_loss(x_gen, x_target, reduction='none')
                loss_per_sample = loss_per_sample.view(batch_size, -1).mean(dim=1)
                total_loss = loss_per_sample.sum()
                total_loss.backward()
                optimizer.step()
            
            with torch.no_grad():
                x_gen = self.generator(z)
                final_loss = F.mse_loss(x_gen, x_target, reduction='none')
                final_loss = final_loss.view(batch_size, -1).mean(dim=1)
                
                for i in range(batch_size):
                    if final_loss[i].item() < best_loss_list[i]:
                        best_loss_list[i] = final_loss[i].item()
                        best_z_list[i] = z[i].detach().clone()
        
        best_z = torch.stack(best_z_list)
        
        with torch.no_grad():
            x_rec = self.generator(best_z)
            x_rec = self._normalize_for_classifier(x_rec)
        
        return x_rec


# ========== モデル読み込み ==========
def load_classifier(args, device):
    data = torch.load(args.cached_samples, map_location='cpu')
    num_classes = len(data['classes'])
    
    classifier = models.resnet50(weights=None)
    num_features = classifier.fc.in_features
    classifier.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    
    checkpoint = torch.load(args.clf_ckpt, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    fixed_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('fc.1.'):
            new_k = k.replace('fc.1.', 'fc.')
            fixed_state_dict[new_k] = v
        elif k.startswith('fc.0.') or k == 'fc.weight' or k == 'fc.bias':
            fixed_state_dict[k] = v
        else:
            fixed_state_dict[k] = v
    
    try:
        classifier.load_state_dict(fixed_state_dict)
    except RuntimeError:
        classifier.fc = nn.Linear(num_features, num_classes)
        simple_state_dict = {}
        for k, v in state_dict.items():
            if k == 'fc.1.weight':
                simple_state_dict['fc.weight'] = v
            elif k == 'fc.1.bias':
                simple_state_dict['fc.bias'] = v
            elif not k.startswith('fc.'):
                simple_state_dict[k] = v
            else:
                simple_state_dict[k] = v
        classifier.load_state_dict(simple_state_dict)
    
    classifier = classifier.to(device).eval()
    return classifier


def load_generator(args, device):
    checkpoint = torch.load(args.gan_ckpt, map_location=device)
    
    ckpt_args = checkpoint.get('args', {})
    latent_dim = ckpt_args.get('latent_dim', 512)
    ngf = ckpt_args.get('ngf', 64)
    
    generator = GeneratorV3(latent_dim=latent_dim, ngf=ngf, nc=3).to(device)
    
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
        
        if 'ema_state_dict' in checkpoint:
            ema_state_dict = checkpoint['ema_state_dict']
            current_state = generator.state_dict()
            for name, param in ema_state_dict.items():
                if name in current_state:
                    current_state[name] = param
            generator.load_state_dict(current_state)
    else:
        raise ValueError(f"Cannot find generator weights in checkpoint")
    
    generator.eval()
    return generator, latent_dim


def load_cached_samples(path):
    data = torch.load(path, map_location='cpu')
    x_test = data['x_test']
    y_test = data['y_test']
    classes = data['classes']
    return x_test, y_test, classes


# ========== FGSM攻撃 ==========
def fgsm_attack(model, x, y, epsilon, device):
    x = x.clone().to(device)
    x.requires_grad = True
    
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    x_norm = (x - mean) / std
    
    outputs = model(x_norm)
    loss = F.cross_entropy(outputs, y.to(device))
    loss.backward()
    
    x_adv = x + epsilon * x.grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1)
    
    return x_adv.detach()


# ========== 分類 ==========
def classify(model, x, device):
    """分類実行、予測ラベルと信頼度を返す"""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    with torch.no_grad():
        x_norm = (x - mean) / std
        outputs = model(x_norm)
        probs = F.softmax(outputs, dim=1)
        pred_labels = outputs.argmax(dim=1)
        pred_confs = probs.max(dim=1)[0]
    
    return pred_labels, pred_confs


# ========== メイン ==========
def main():
    args = parse_args()
    
    # シード設定
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # デバイス
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Defense-GAN Quick Visual Test")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
    
    # 出力ディレクトリ
    timestamp = datetime.now().strftime("%m%d%H%M%S")
    log_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output: {log_dir}\n")
    
    # モデル読み込み
    print("Loading models...")
    classifier = load_classifier(args, device)
    print("✓ Loaded classifier")
    
    generator, latent_dim = load_generator(args, device)
    print("✓ Loaded generator")
    
    # Defense-GAN初期化
    defense_gan = DefenseGAN(
        generator=generator,
        latent_dim=latent_dim,
        rec_iters=args.rec_iters,
        rec_lr=args.rec_lr,
        rec_rr=args.rec_rr,
        device=device
    )
    print(f"✓ Defense-GAN initialized (iters={args.rec_iters}, rr={args.rec_rr})\n")
    
    # データ読み込み
    x_test, y_test, classes = load_cached_samples(args.cached_samples)
    print(f"Loaded {len(x_test)} samples, {len(classes)} classes: {classes}\n")
    
    # テスト用に数枚抽出
    n_samples = min(args.n_samples, len(x_test))
    x_sample = x_test[:n_samples].to(device)
    y_sample = y_test[:n_samples].to(device)
    
    print(f"{'='*70}")
    print(f"Testing with {n_samples} samples")
    print(f"{'='*70}\n")
    
    # ステップ1: クリーン画像で分類
    print("[1/4] Classifying clean images...")
    clean_preds, clean_confs = classify(classifier, x_sample, device)
    print("✓ Done\n")
    
    for i in range(n_samples):
        pred_label = classes[clean_preds[i].item()]
        true_label = classes[y_sample[i].item()]
        conf = clean_confs[i].item()
        match = "✓" if clean_preds[i] == y_sample[i] else "✗"
        print(f"  Sample {i+1}: True={true_label:8s} | Pred={pred_label:8s} | Conf={conf:.4f} {match}")
    
    # ステップ2: FGSM攻撃
    print(f"\n[2/4] Generating FGSM adversarial examples (epsilon={args.epsilon:.4f})...")
    start_time = time.time()
    x_adv = fgsm_attack(classifier, x_sample, y_sample, args.epsilon, device)
    attack_time = time.time() - start_time
    print(f"✓ Done ({attack_time:.2f}s)\n")
    
    # ステップ3: 敵対的画像で分類
    print("[3/4] Classifying adversarial images (no defense)...")
    adv_preds, adv_confs = classify(classifier, x_adv, device)
    print("✓ Done\n")
    
    for i in range(n_samples):
        pred_label = classes[adv_preds[i].item()]
        true_label = classes[y_sample[i].item()]
        conf = adv_confs[i].item()
        match = "✓" if adv_preds[i] == y_sample[i] else "✗"
        print(f"  Sample {i+1}: True={true_label:8s} | Pred={pred_label:8s} | Conf={conf:.4f} {match}")
    
    # ステップ4: Defense-GAN浄化
    print(f"\n[4/4] Purifying adversarial images with Defense-GAN...")
    start_time = time.time()
    x_purified = defense_gan.reconstruct(x_adv)
    purify_time = time.time() - start_time
    print(f"✓ Done ({purify_time:.2f}s, {purify_time/n_samples*1000:.0f}ms/sample)\n")
    
    # 浄化後の分類
    print("[5/5] Classifying purified images...")
    purified_preds, purified_confs = classify(classifier, x_purified, device)
    print("✓ Done\n")
    
    for i in range(n_samples):
        pred_label = classes[purified_preds[i].item()]
        true_label = classes[y_sample[i].item()]
        conf = purified_confs[i].item()
        match = "✓" if purified_preds[i] == y_sample[i] else "✗"
        print(f"  Sample {i+1}: True={true_label:8s} | Pred={pred_label:8s} | Conf={conf:.4f} {match}")
    
    # 結果サマリー
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    
    clean_acc = (clean_preds == y_sample).float().mean().item()
    adv_acc = (adv_preds == y_sample).float().mean().item()
    purified_acc = (purified_preds == y_sample).float().mean().item()
    
    print(f"Clean accuracy:            {clean_acc*100:6.2f}%")
    print(f"Adversarial accuracy:      {adv_acc*100:6.2f}%  (attack success: {(clean_acc-adv_acc)*100:6.2f}%)")
    print(f"Purified accuracy:         {purified_acc*100:6.2f}%  (improvement: {(purified_acc-adv_acc)*100:+6.2f}%)")
    print(f"\nTotal time: {attack_time + purify_time:.2f}s")
    
    # 画像の可視化・保存
    print(f"\n{'='*70}")
    print("Saving visualizations...")
    print(f"{'='*70}\n")
    
    # 1. 画像正規化の解除（表示用）
    def denormalize_for_display(x):
        mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        x_display = x * std + mean
        return x_display.clamp(0, 1)
    
    x_clean_display = denormalize_for_display(x_sample)
    x_adv_display = denormalize_for_display(x_adv)
    x_purified_display = denormalize_for_display(x_purified)
    
    # 2. グリッド画像を作成
    # 行ごとに: クリーン, 敵対的, 浄化後
    for i in range(n_samples):
        images = torch.stack([
            x_clean_display[i],
            x_adv_display[i],
            x_purified_display[i]
        ])
        grid = make_grid(images, nrow=3, padding=10, normalize=False)
        true_label = classes[y_sample[i].item()]
        clean_label = classes[clean_preds[i].item()]
        adv_label = classes[adv_preds[i].item()]
        purified_label = classes[purified_preds[i].item()]
        
        filename = f"sample_{i+1:02d}_{true_label}_clean-{clean_label}_adv-{adv_label}_purif-{purified_label}.png"
        save_image(grid, os.path.join(log_dir, filename))
        print(f"  [{i+1}] Saved: {filename}")
    
    # 3. 全体グリッド（クリーン、敵対的、浄化後を3行で）
    grid_clean = make_grid(x_clean_display, nrow=n_samples, padding=5, normalize=False)
    grid_adv = make_grid(x_adv_display, nrow=n_samples, padding=5, normalize=False)
    grid_purified = make_grid(x_purified_display, nrow=n_samples, padding=5, normalize=False)
    
    grid_all = torch.cat([grid_clean, grid_adv, grid_purified], dim=1)
    save_image(grid_all, os.path.join(log_dir, 'all_samples_comparison.png'))
    print(f"\n  [Grid] Saved: all_samples_comparison.png")
    
    # 4. テキストレポート保存
    report_path = os.path.join(log_dir, 'test_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Defense-GAN Quick Visual Test Report\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)\n")
        f.write(f"  Reconstruction iterations: {args.rec_iters}\n")
        f.write(f"  Learning rate: {args.rec_lr}\n")
        f.write(f"  Random restarts: {args.rec_rr}\n")
        f.write(f"  Samples: {n_samples}\n\n")
        
        f.write(f"Results:\n")
        f.write(f"  Clean accuracy: {clean_acc*100:.2f}%\n")
        f.write(f"  Adversarial accuracy (no defense): {adv_acc*100:.2f}%\n")
        f.write(f"  Adversarial accuracy (with Defense-GAN): {purified_acc*100:.2f}%\n")
        f.write(f"  Defense improvement: {(purified_acc-adv_acc)*100:+.2f}%\n")
        f.write(f"  Attack success rate: {(clean_acc-adv_acc)*100:.2f}%\n\n")
        
        f.write(f"Timings:\n")
        f.write(f"  Attack time: {attack_time:.2f}s\n")
        f.write(f"  Purification time: {purify_time:.2f}s ({purify_time/n_samples*1000:.0f}ms/sample)\n")
        f.write(f"  Total time: {attack_time + purify_time:.2f}s\n\n")
        
        f.write(f"Sample Details:\n")
        f.write(f"{'-'*70}\n")
        f.write(f"{'Sample':<8} {'True':<12} {'Clean':<12} {'Adversarial':<15} {'Purified':<12} {'Improvement':<12}\n")
        f.write(f"{'-'*70}\n")
        for i in range(n_samples):
            true_label = classes[y_sample[i].item()]
            clean_label = f"{classes[clean_preds[i].item()]}({clean_confs[i]:.2f})"
            adv_label = f"{classes[adv_preds[i].item()]}({adv_confs[i]:.2f})"
            purified_label = f"{classes[purified_preds[i].item()]}({purified_confs[i]:.2f})"
            improvement = "✓" if purified_preds[i] == y_sample[i] and adv_preds[i] != y_sample[i] else "-"
            f.write(f"{i+1:<8} {true_label:<12} {clean_label:<12} {adv_label:<15} {purified_label:<12} {improvement:<12}\n")
    
    print(f"\n  [Report] Saved: test_report.txt\n")
    
    print(f"{'='*70}")
    print(f"✓ All visualizations saved to: {log_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
