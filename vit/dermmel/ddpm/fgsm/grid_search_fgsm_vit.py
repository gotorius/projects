"""
DDPM FGSM防御 グリッドサーチ: start_t と t_purify の最適パラメータ探索 (ViT)

- start_t: 0〜300 (10間隔) → 31値
- t_purify: 0〜300 (10間隔) → 31値
- 制約: t_purify <= start_t (start_t より多くのステップは逆拡散できない)
- データ: 各クラス5枚 × 2クラス = 10枚

実行例:
python grid_search_fgsm_vit.py
python grid_search_fgsm_vit.py --gpu 0 --epsilon 0.031
"""

import os
import sys
import argparse
import time
import json
import csv
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from tqdm.auto import tqdm

# プロジェクトのルートを追加
sys.path.insert(0, '/mnt/data1/gotou/projects/resnet/dermmel/ddpm')
from ddpm_train_v2 import SimpleUNet


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='Grid Search: start_t & t_purify for DDPM FGSM Defense (ViT)')

    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='FGSM perturbation epsilon')

    # グリッドサーチ範囲
    parser.add_argument('--start_t_min', type=int, default=0)
    parser.add_argument('--start_t_max', type=int, default=300)
    parser.add_argument('--start_t_step', type=int, default=10)
    parser.add_argument('--purify_min', type=int, default=0)
    parser.add_argument('--purify_max', type=int, default=300)
    parser.add_argument('--purify_step', type=int, default=10)

    # DDPM設定
    parser.add_argument('--eta', type=float, default=1.0,
                        help='Stochasticity parameter for DDIM')

    # パス設定
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/vit/dermmel/vit/correct_samples_balanced_500_vit.pt')
    parser.add_argument('--ddpm_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/resnet/dermmel/ddpm/ddpm_out2/best_model.pth')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/vit/classifiers/checkpoints/dermmel/20260118_175806/best_vit_dermmel.pth')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/vit/dermmel/ddpm/fgsm/results')

    # 実行設定
    parser.add_argument('--n_samples_per_class', type=int, default=5,
                        help='Number of samples per class for evaluation')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for evaluation')
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


# ========== 定数 ==========
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========== ViTモデル構築 ==========
def get_vit_model(model_name='vit_b_16', num_classes=2, dropout=0.1):
    if model_name == 'vit_b_16':
        model = models.vit_b_16(weights=None)
    elif model_name == 'vit_b_32':
        model = models.vit_b_32(weights=None)
    elif model_name == 'vit_l_16':
        model = models.vit_l_16(weights=None)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes)
    )
    return model


# ========== GaussianDiffusion クラス ==========
class GaussianDiffusion:
    def __init__(self, timesteps=1000, device='cuda', beta_schedule='cosine'):
        self.timesteps = timesteps
        self.device = device

        if beta_schedule == 'linear':
            betas = torch.linspace(1e-4, 0.02, timesteps)
        elif beta_schedule == 'cosine':
            import math
            s = 0.008
            t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
            f = (t / timesteps + s) / (1 + s)
            alphas_bar = torch.cos(f * math.pi / 2) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
            betas = betas.clamp(min=1e-8, max=0.999).to(torch.float32)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.betas = betas.to(device)
        self.alphas = (1.0 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)

        alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), self.alphas_cumprod[:-1]], dim=0)
        self.posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod + 1e-8)
        self.posterior_variance[0] = 1e-8


# ========== DDPM Purifier ==========
class DDPMPurifierImproved(nn.Module):
    def __init__(self, unet, diffusion, device, t_purify=50, start_t=80, eta=0.0):
        super().__init__()
        self.unet = unet
        self.diffusion = diffusion
        self.device = device
        self.t_purify = t_purify
        self.start_t = start_t
        self.eta = eta

    def forward(self, x):
        # [0,1] → [-1,1]
        x_minus1to1 = x * 2.0 - 1.0

        batch_size = x.size(0)
        t0 = torch.full((batch_size,), self.start_t, device=self.device, dtype=torch.long)
        noise = torch.randn_like(x_minus1to1)

        sqrt_alpha_bar_t0 = self.diffusion.sqrt_alphas_cumprod[t0].view(-1, 1, 1, 1)
        sqrt_1m_alpha_bar_t0 = self.diffusion.sqrt_one_minus_alphas_cumprod[t0].view(-1, 1, 1, 1)
        x_t = sqrt_alpha_bar_t0 * x_minus1to1 + sqrt_1m_alpha_bar_t0 * noise

        eps_pred_final = None
        t_final = self.start_t

        for i in range(self.t_purify):
            curr_t = self.start_t - i
            if curr_t < 0:
                break

            t_batch = torch.full((batch_size,), curr_t, device=self.device, dtype=torch.long)
            eps_pred = self.unet(x_t, t_batch)

            alpha_t = self.diffusion.alphas[curr_t]
            alpha_bar_t = self.diffusion.alphas_cumprod[curr_t]

            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_pred
            )

            if curr_t > 0:
                z = torch.randn_like(x_t)
                sigma = self.eta * torch.sqrt(self.diffusion.posterior_variance[curr_t])
                x_t = mean + sigma * z
            else:
                x_t = mean

            x_t = torch.clamp(x_t, -1.0, 1.0)
            eps_pred_final = eps_pred
            t_final = curr_t

        alpha_bar_tf = self.diffusion.alphas_cumprod[t_final]
        x0_hat = (x_t - torch.sqrt(1 - alpha_bar_tf) * eps_pred_final) / torch.sqrt(alpha_bar_tf + 1e-12)
        x0_hat = torch.clamp(x0_hat, -1.0, 1.0)

        x_purified = (x0_hat + 1.0) / 2.0
        x_purified = torch.clamp(x_purified, 0, 1)

        return x_purified


# ========== モデル読み込み ==========
def load_models(args, device):
    data = torch.load(args.cached_samples, map_location='cpu')
    num_classes = len(data['classes'])

    classifier = get_vit_model(model_name='vit_b_16', num_classes=num_classes, dropout=0.1)
    checkpoint = torch.load(args.clf_ckpt, map_location=device)
    if 'model_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        classifier.load_state_dict(checkpoint)
    classifier = classifier.to(device).eval()
    print(f"Loaded ViT classifier from {args.clf_ckpt}")

    ddpm_ckpt = torch.load(args.ddpm_ckpt, map_location=device)
    ddpm_args = ddpm_ckpt.get('args', {})
    base_ch = ddpm_args.get('base_channels', 128)
    timesteps = ddpm_args.get('timesteps', 1000)

    unet = SimpleUNet(in_ch=3, base_ch=base_ch, time_emb_dim=256, attn_heads=4).to(device)
    if 'ema_state_dict' in ddpm_ckpt and ddpm_ckpt['ema_state_dict'] is not None:
        print("Using EMA weights")
        unet.load_state_dict(ddpm_ckpt['ema_state_dict'])
    elif 'model_state_dict' in ddpm_ckpt:
        unet.load_state_dict(ddpm_ckpt['model_state_dict'])
    else:
        unet.load_state_dict(ddpm_ckpt)
    unet.eval()

    diffusion = GaussianDiffusion(timesteps=timesteps, device=device)
    print(f"Loaded DDPM (base_ch={base_ch}, timesteps={timesteps})")

    return classifier, unet, diffusion


# ========== データ読み込み（各クラスN枚ずつ） ==========
def load_subset_samples(path, n_per_class=5, seed=42):
    """各クラスからn_per_class枚ずつサンプルを取得"""
    data = torch.load(path, map_location='cpu')
    x_all = data['x_test']
    y_all = data['y_test']
    classes = data['classes']

    torch.manual_seed(seed)
    np.random.seed(seed)

    x_subset = []
    y_subset = []

    for cls_idx in range(len(classes)):
        mask = (y_all == cls_idx)
        x_cls = x_all[mask]
        y_cls = y_all[mask]

        indices = torch.randperm(len(x_cls))[:n_per_class]
        x_subset.append(x_cls[indices])
        y_subset.append(y_cls[indices])

    x_subset = torch.cat(x_subset, dim=0)
    y_subset = torch.cat(y_subset, dim=0)

    print(f"Loaded subset: {len(x_subset)} samples ({n_per_class}/class)")
    for i, c in enumerate(classes):
        cnt = (y_subset == i).sum().item()
        print(f"  {c}: {cnt} samples")

    return x_subset, y_subset, classes


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


# ========== 評価関数 ==========
def evaluate_accuracy(model, x, y, device):
    """精度を計算（少量データ用、バッチ分割なし）"""
    model.eval()
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        x_dev = x.to(device)
        y_dev = y.to(device)
        x_norm = (x_dev - mean) / std
        outputs = model(x_norm)
        _, predicted = outputs.max(1)
        correct = (predicted == y_dev).sum().item()

    return correct / len(y), predicted.cpu().numpy()


def evaluate_with_purification(purifier, classifier, x, y, device):
    """DDPM浄化後の精度を計算"""
    purifier.eval()
    classifier.eval()

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        x_dev = x.to(device)
        y_dev = y.to(device)

        x_purified = purifier(x_dev)

        x_norm = (x_purified - mean) / std
        outputs = classifier(x_norm)
        _, predicted = outputs.max(1)
        correct = (predicted == y_dev).sum().item()

    return correct / len(y), predicted.cpu().numpy()


# ========== メイン ==========
def main():
    args = parse_args()

    # シード
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 出力ディレクトリ
    timestamp = datetime.now().strftime("%m%d%H%M")
    log_dir = os.path.join(args.output_dir, f'grid_search_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {log_dir}")

    # モデル読み込み
    classifier, unet, diffusion = load_models(args, device)

    # データ読み込み (各クラス5枚 = 10枚)
    x_test, y_test, classes = load_subset_samples(
        args.cached_samples, n_per_class=args.n_samples_per_class, seed=args.seed
    )

    # ベースライン: クリーン精度
    clean_acc, _ = evaluate_accuracy(classifier, x_test, y_test, device)
    print(f"\nClean accuracy (no attack, no defense): {clean_acc:.4f}")

    # FGSM攻撃（一度だけ実行）
    print(f"Running FGSM attack (epsilon={args.epsilon:.4f})...")
    x_adv = fgsm_attack(classifier, x_test, y_test, args.epsilon, device)
    adv_acc, _ = evaluate_accuracy(classifier, x_adv, y_test, device)
    print(f"Adversarial accuracy (no defense): {adv_acc:.4f}")

    # グリッドサーチ範囲
    start_t_values = list(range(args.start_t_min, args.start_t_max + 1, args.start_t_step))
    purify_values = list(range(args.purify_min, args.purify_max + 1, args.purify_step))

    # start_t=0 は浄化なしと同じなので除外
    if 0 in start_t_values:
        start_t_values.remove(0)
    # t_purify=0 も浄化なしなので除外
    if 0 in purify_values:
        purify_values.remove(0)

    # 有効な組み合わせ数を計算 (t_purify <= start_t)
    valid_combos = [(st, tp) for st in start_t_values for tp in purify_values if tp <= st]
    total_combos = len(valid_combos)
    print(f"\nGrid search: {len(start_t_values)} start_t × {len(purify_values)} t_purify")
    print(f"Valid combinations (t_purify <= start_t): {total_combos}")
    print(f"start_t range: {start_t_values[0]}〜{start_t_values[-1]} (step {args.start_t_step})")
    print(f"t_purify range: {purify_values[0]}〜{purify_values[-1]} (step {args.purify_step})")

    # 結果格納
    results = []
    best_adv_acc = -1
    best_params = {}
    best_combined_score = -1  # adv_acc_with_ddpm - clean_acc_drop のバランス
    best_balanced_params = {}

    # CSVヘッダー
    csv_path = os.path.join(log_dir, 'grid_search_results.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'start_t', 't_purify', 'eta',
            'clean_acc', 'adv_acc_no_defense',
            'clean_acc_with_ddpm', 'adv_acc_with_ddpm',
            'defense_improvement', 'clean_acc_drop',
            'combined_score', 'time_sec'
        ])

    # ログファイル
    log_path = os.path.join(log_dir, 'grid_search_log.txt')
    log_file = open(log_path, 'w')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f"{'='*80}")
    log(f"Grid Search: start_t & t_purify for DDPM FGSM Defense (ViT) - DermMel")
    log(f"{'='*80}")
    log(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Classifier: ViT-B/16")
    log(f"Attack: FGSM, epsilon={args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    log(f"Samples: {len(x_test)} ({args.n_samples_per_class}/class)")
    log(f"Classes: {classes}")
    log(f"Clean accuracy: {clean_acc:.4f}")
    log(f"Adversarial accuracy (no defense): {adv_acc:.4f}")
    log(f"Grid: start_t [{start_t_values[0]}..{start_t_values[-1]}] step {args.start_t_step}")
    log(f"Grid: t_purify [{purify_values[0]}..{purify_values[-1]}] step {args.purify_step}")
    log(f"Valid combinations: {total_combos}")
    log(f"eta: {args.eta}")
    log(f"{'='*80}\n")

    total_start_time = time.time()

    # グリッドサーチ実行
    pbar = tqdm(valid_combos, desc="Grid Search", ncols=120)
    for idx, (st, tp) in enumerate(pbar):
        iter_start = time.time()

        # シード固定（再現性のため）
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

        # Purifier作成
        purifier = DDPMPurifierImproved(
            unet, diffusion, device,
            t_purify=tp, start_t=st, eta=args.eta
        )

        # クリーン画像 + DDPM浄化
        clean_acc_ddpm, _ = evaluate_with_purification(
            purifier, classifier, x_test, y_test, device
        )

        # 敵対的画像 + DDPM浄化
        adv_acc_ddpm, _ = evaluate_with_purification(
            purifier, classifier, x_adv, y_test, device
        )

        iter_time = time.time() - iter_start

        # メトリクス計算
        defense_improvement = adv_acc_ddpm - adv_acc
        clean_acc_drop = clean_acc - clean_acc_ddpm
        # 結合スコア: 敵対的精度を最大化しつつ、クリーン精度低下を最小化
        combined_score = adv_acc_ddpm - 0.5 * max(0, clean_acc_drop)

        result_entry = {
            'start_t': st,
            't_purify': tp,
            'eta': args.eta,
            'clean_acc': clean_acc,
            'adv_acc_no_defense': adv_acc,
            'clean_acc_with_ddpm': clean_acc_ddpm,
            'adv_acc_with_ddpm': adv_acc_ddpm,
            'defense_improvement': defense_improvement,
            'clean_acc_drop': clean_acc_drop,
            'combined_score': combined_score,
            'time_sec': iter_time,
        }
        results.append(result_entry)

        # CSVに追記
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                st, tp, args.eta,
                f"{clean_acc:.4f}", f"{adv_acc:.4f}",
                f"{clean_acc_ddpm:.4f}", f"{adv_acc_ddpm:.4f}",
                f"{defense_improvement:+.4f}", f"{clean_acc_drop:+.4f}",
                f"{combined_score:.4f}", f"{iter_time:.2f}"
            ])

        # ベスト更新チェック
        is_best_adv = adv_acc_ddpm > best_adv_acc
        is_best_combined = combined_score > best_combined_score

        if is_best_adv:
            best_adv_acc = adv_acc_ddpm
            best_params = {'start_t': st, 't_purify': tp}

        if is_best_combined:
            best_combined_score = combined_score
            best_balanced_params = {'start_t': st, 't_purify': tp}

        # プログレスバー更新
        pbar.set_postfix({
            'st': st, 'tp': tp,
            'adv_ddpm': f'{adv_acc_ddpm:.2f}',
            'clean_ddpm': f'{clean_acc_ddpm:.2f}',
            'best_adv': f'{best_adv_acc:.2f}',
        })

        # 定期ログ (20組み合わせごと)
        if (idx + 1) % 20 == 0 or is_best_adv:
            marker = " ★BEST" if is_best_adv else ""
            log(f"[{idx+1}/{total_combos}] start_t={st:3d}, t_purify={tp:3d} | "
                f"clean_ddpm={clean_acc_ddpm:.4f}, adv_ddpm={adv_acc_ddpm:.4f}, "
                f"defense={defense_improvement:+.4f} | {iter_time:.1f}s{marker}")

    total_time = time.time() - total_start_time

    # ========== 結果サマリー ==========
    log(f"\n{'='*80}")
    log(f"GRID SEARCH COMPLETE")
    log(f"{'='*80}")
    log(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    log(f"Combinations evaluated: {total_combos}")
    log(f"Average time per combo: {total_time/total_combos:.2f}s")

    # ベスト結果 (敵対的精度最大)
    log(f"\n--- Best by Adversarial Accuracy (with DDPM) ---")
    log(f"  start_t = {best_params['start_t']}, t_purify = {best_params['t_purify']}")
    best_entry = [r for r in results
                  if r['start_t'] == best_params['start_t']
                  and r['t_purify'] == best_params['t_purify']][0]
    log(f"  Adv acc (with DDPM):   {best_entry['adv_acc_with_ddpm']:.4f}")
    log(f"  Clean acc (with DDPM): {best_entry['clean_acc_with_ddpm']:.4f}")
    log(f"  Defense improvement:   {best_entry['defense_improvement']:+.4f}")
    log(f"  Clean acc drop:        {best_entry['clean_acc_drop']:+.4f}")

    # ベスト結果 (結合スコア)
    log(f"\n--- Best by Combined Score (adv_acc - 0.5*clean_drop) ---")
    log(f"  start_t = {best_balanced_params['start_t']}, t_purify = {best_balanced_params['t_purify']}")
    best_bal_entry = [r for r in results
                      if r['start_t'] == best_balanced_params['start_t']
                      and r['t_purify'] == best_balanced_params['t_purify']][0]
    log(f"  Adv acc (with DDPM):   {best_bal_entry['adv_acc_with_ddpm']:.4f}")
    log(f"  Clean acc (with DDPM): {best_bal_entry['clean_acc_with_ddpm']:.4f}")
    log(f"  Combined score:        {best_bal_entry['combined_score']:.4f}")

    # Top 10 (敵対的精度)
    sorted_results = sorted(results, key=lambda r: r['adv_acc_with_ddpm'], reverse=True)
    log(f"\n--- Top 10 by Adversarial Accuracy ---")
    log(f"{'Rank':>4} | {'start_t':>7} | {'t_purify':>8} | {'adv_ddpm':>8} | {'clean_ddpm':>10} | {'defense':>8} | {'combined':>8}")
    log("-" * 75)
    for rank, r in enumerate(sorted_results[:10], 1):
        log(f"{rank:>4} | {r['start_t']:>7} | {r['t_purify']:>8} | "
            f"{r['adv_acc_with_ddpm']:>8.4f} | {r['clean_acc_with_ddpm']:>10.4f} | "
            f"{r['defense_improvement']:>+8.4f} | {r['combined_score']:>8.4f}")

    # Top 10 (結合スコア)
    sorted_combined = sorted(results, key=lambda r: r['combined_score'], reverse=True)
    log(f"\n--- Top 10 by Combined Score ---")
    log(f"{'Rank':>4} | {'start_t':>7} | {'t_purify':>8} | {'adv_ddpm':>8} | {'clean_ddpm':>10} | {'defense':>8} | {'combined':>8}")
    log("-" * 75)
    for rank, r in enumerate(sorted_combined[:10], 1):
        log(f"{rank:>4} | {r['start_t']:>7} | {r['t_purify']:>8} | "
            f"{r['adv_acc_with_ddpm']:>8.4f} | {r['clean_acc_with_ddpm']:>10.4f} | "
            f"{r['defense_improvement']:>+8.4f} | {r['combined_score']:>8.4f}")

    log_file.close()

    # JSON保存
    summary = {
        'config': {
            'epsilon': args.epsilon,
            'eta': args.eta,
            'n_samples_per_class': args.n_samples_per_class,
            'total_samples': len(x_test),
            'classes': classes,
            'seed': args.seed,
            'grid': {
                'start_t': {'min': start_t_values[0], 'max': start_t_values[-1], 'step': args.start_t_step},
                't_purify': {'min': purify_values[0], 'max': purify_values[-1], 'step': args.purify_step},
            },
            'total_combinations': total_combos,
        },
        'baseline': {
            'clean_acc': clean_acc,
            'adv_acc_no_defense': adv_acc,
        },
        'best_by_adv_acc': {
            'params': best_params,
            'adv_acc_with_ddpm': best_entry['adv_acc_with_ddpm'],
            'clean_acc_with_ddpm': best_entry['clean_acc_with_ddpm'],
            'defense_improvement': best_entry['defense_improvement'],
        },
        'best_by_combined_score': {
            'params': best_balanced_params,
            'adv_acc_with_ddpm': best_bal_entry['adv_acc_with_ddpm'],
            'clean_acc_with_ddpm': best_bal_entry['clean_acc_with_ddpm'],
            'combined_score': best_bal_entry['combined_score'],
        },
        'total_time_sec': total_time,
        'all_results': results,
    }

    json_path = os.path.join(log_dir, 'grid_search_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {log_dir}/")
    print(f"  CSV:  {csv_path}")
    print(f"  Log:  {log_path}")
    print(f"  JSON: {json_path}")


if __name__ == '__main__':
    main()
