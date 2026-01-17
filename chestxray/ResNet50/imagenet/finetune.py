"""
ChestXray (肺炎分類) - ImageNet事前学習 Guided-Diffusion のドメイン適応ファインチューニング (本格版)

目的:
- 事前学習済み 256x256 unconditional Guided-Diffusion を ChestXray ドメインで本格的に微調整
- 出力された微調整モデルを adversarial defense (FGSM+浄化) 評価スクリプトで使用

改善点:
- 長期学習対応 (デフォルト100万ステップ)
- Cosine Annealing学習率スケジューラ + Warmup
- Gradient Accumulation対応
- Validation評価 + Best Model保存
- 医用画像に適したデータ拡張 (過度でない範囲で)
- 定期的なチェックポイント保存
- FP16混合精度学習
- 学習曲線プロット機能

特徴:
- torchvision ImageFolder による学習データ読み込み (train, val ディレクトリ)
- 画像は [-1,1] に正規化 (拡散モデル仕様)
- guided_diffusion の GaussianDiffusion.training_losses を使用
- AdamW + Cosine Annealing + EMA (decay=0.9999)
- 単GPU前提、分散/mpi 依存なし

使い方(例):
  python finetune.py \
    --data-dir /mnt/data1/Public/MedImages/CellData/chest_xray/train \
    --val-dir /mnt/data1/Public/MedImages/CellData/chest_xray/val \
    --pretrained /mnt/data1/gotou/kaggle/guided-diffusion/256x256_diffusion_uncond.pt \
    --out-dir /mnt/data1/gotou/projects/chestxray/imagenet/finetune_out \
    --total-steps 1000000 --batch-size 4 --accum-steps 4 \
    --lr 1e-4 --warmup-steps 5000 --fp16

学習後:
- 出力した best_ema_*.pt を imagenet_fgsm.py の model_path に設定して評価
"""

import os
import sys
import argparse
import json
from typing import Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import numpy as np

# guided-diffusion をパスに追加
sys.path.insert(0, '/mnt/data1/gotou/kaggle/guided-diffusion')
from guided_diffusion.script_util import create_model_and_diffusion
from guided_diffusion import logger


def to_minus1_1():
    """[0,1] -> [-1,1] に変換するTransform"""
    return transforms.Lambda(lambda x: x * 2.0 - 1.0)


class MedicalImageAugmentation:
    """医用画像(胸部X線)向けのデータ拡張 - 過度にならない範囲で"""
    @staticmethod
    def get_train_transforms(image_size: int, intensity: str = 'medium'):
        """
        Args:
            image_size: 出力画像サイズ
            intensity: 'light', 'medium', 'heavy' のいずれか
        """
        tfms = []
        
        # リサイズ+クロップ
        tfms.append(transforms.Resize(image_size + 16))
        tfms.append(transforms.RandomCrop(image_size))
        
        # 左右反転 (胸部X線では一般的に問題ない)
        tfms.append(transforms.RandomHorizontalFlip(p=0.5))
        
        if intensity in ['medium', 'heavy']:
            # 軽度の回転 (±5度以内、医用画像なので控えめ)
            tfms.append(transforms.RandomRotation(degrees=5))
            
            # 軽度のアフィン変換
            tfms.append(transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ))
        
        if intensity == 'heavy':
            # コントラスト・明度調整 (X線画像の撮影条件の違いをシミュレート)
            tfms.append(transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
            ))
        
        tfms.append(transforms.ToTensor())
        tfms.append(to_minus1_1())
        
        return transforms.Compose(tfms)
    
    @staticmethod
    def get_val_transforms(image_size: int):
        """検証用(拡張なし)"""
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            to_minus1_1(),
        ])


class WarmupCosineScheduler:
    """Warmup + Cosine Annealing学習率スケジューラ"""
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        base_lr: float,
        min_lr: float = 1e-7,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            # Warmup phase
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class EMA:
    """シンプルなEMA管理クラス (パラメータ影のコピーを保持)"""
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            new_avg = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
            self.shadow[name] = new_avg

    def copy_to(self, model: nn.Module):
        """EMA重みを一時的にモデルへ反映 (評価/保存時など)"""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self, model: nn.Module):
        """モデルのstate_dictを複製し、学習可能パラメータのみEMAで置き換え"""
        sd = model.state_dict()
        for name, buf in sd.items():
            if name in self.shadow:
                sd[name] = self.shadow[name].clone().detach().to(buf.device)
        return sd


def build_dataloader(
    data_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    is_train: bool = True,
    augmentation_intensity: str = 'medium',
) -> DataLoader:
    """データローダー構築
    
    Args:
        data_dir: データディレクトリ
        image_size: 画像サイズ
        batch_size: バッチサイズ
        num_workers: ワーカー数
        is_train: 学習用かどうか
        augmentation_intensity: データ拡張の強度 ('light', 'medium', 'heavy')
    """
    if is_train:
        transform = MedicalImageAugmentation.get_train_transforms(image_size, augmentation_intensity)
        shuffle = True
        drop_last = True
    else:
        transform = MedicalImageAugmentation.get_val_transforms(image_size)
        shuffle = False
        drop_last = False
    
    ds = datasets.ImageFolder(root=data_dir, transform=transform)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
    return dl


def save_checkpoint(state_dict: dict, out_dir: str, prefix: str, step: int):
    """チェックポイント保存"""
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"{prefix}_{step:07d}.pt")
    torch.save(state_dict, ckpt_path)
    return ckpt_path


def save_training_state(
    out_dir: str,
    step: int,
    model,
    ema,
    optimizer,
    scheduler,
    scaler,
    train_losses: list,
    val_losses: list,
    best_val_loss: float,
):
    """学習状態を保存 (resumeできるように)"""
    state = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'ema_shadow': ema.shadow,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state': {
            'current_step': scheduler.current_step,
        },
        'scaler_state_dict': scaler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
    }
    path = os.path.join(out_dir, f'training_state_{step:07d}.pt')
    torch.save(state, path)
    return path


def load_training_state(path: str, model, ema, optimizer, scheduler, scaler):
    """学習状態を読み込んでresume"""
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state['model_state_dict'])
    ema.shadow = state['ema_shadow']
    optimizer.load_state_dict(state['optimizer_state_dict'])
    scheduler.current_step = state['scheduler_state']['current_step']
    scaler.load_state_dict(state['scaler_state_dict'])
    return state['step'], state['train_losses'], state['val_losses'], state['best_val_loss']


@torch.no_grad()
def evaluate_model(model, diffusion, dataloader, device, max_batches: Optional[int] = None):
    """検証データでモデルを評価"""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_vb = 0.0
    num_batches = 0
    
    for i, (images, _) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        
        images = images.to(device, non_blocking=True)
        t = torch.randint(
            low=0,
            high=diffusion.num_timesteps,
            size=(images.size(0),),
            device=device,
            dtype=torch.long,
        )
        
        losses = diffusion.training_losses(model, images, t, model_kwargs={})
        total_loss += losses["loss"].mean().item()
        total_mse += losses.get("mse", torch.tensor(0.0)).mean().item()
        total_vb += losses.get("vb", torch.tensor(0.0)).mean().item()
        num_batches += 1
    
    model.train()
    
    if num_batches == 0:
        return 0.0, 0.0, 0.0
    
    return (
        total_loss / num_batches,
        total_mse / num_batches,
        total_vb / num_batches,
    )


def plot_training_curves(train_losses: list, val_losses: list, out_dir: str):
    """学習曲線をプロット"""
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        if train_losses:
            steps_train = [x[0] for x in train_losses]
            losses_train = [x[1] for x in train_losses]
            ax.plot(steps_train, losses_train, label='Train Loss', alpha=0.7)
        
        if val_losses:
            steps_val = [x[0] for x in val_losses]
            losses_val = [x[1] for x in val_losses]
            ax.plot(steps_val, losses_val, label='Val Loss', marker='o', alpha=0.7)
        
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_path = os.path.join(out_dir, 'training_curve.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training curve saved to: {plot_path}")
    except ImportError:
        print("matplotlib not available, skipping plot generation")
    except Exception as e:
        print(f"Failed to plot training curves: {e}")


def get_default_paths():
    """デフォルトパスを返す"""
    return {
        'data_dir': '/mnt/data1/Public/MedImages/CellData/chest_xray/train',
        'val_dir': '/mnt/data1/Public/MedImages/CellData/chest_xray/val',
        'pretrained': '/mnt/data1/gotou/kaggle/guided-diffusion/256x256_diffusion_uncond.pt',
        'out_dir': '/mnt/data1/gotou/projects/chestxray/imagenet/finetune_out',
    }


def interactive_setup():
    """対話的にパスを設定"""
    defaults = get_default_paths()
    
    print("="*80)
    print("Guided-Diffusion Fine-tuning Setup")
    print("="*80)
    print("\nPress Enter to use default values shown in [brackets]")
    print()
    
    data_dir = input(f"Training data directory [{defaults['data_dir']}]: ").strip()
    if not data_dir:
        data_dir = defaults['data_dir']
    
    val_dir = input(f"Validation data directory [{defaults['val_dir']}] (leave empty to skip validation): ").strip()
    if not val_dir:
        val_dir = defaults['val_dir']
    if val_dir.lower() in ['none', 'skip', 'no']:
        val_dir = None
    
    pretrained = input(f"Pretrained model path [{defaults['pretrained']}]: ").strip()
    if not pretrained:
        pretrained = defaults['pretrained']
    
    out_dir = input(f"Output directory [{defaults['out_dir']}]: ").strip()
    if not out_dir:
        out_dir = defaults['out_dir']
    
    print("\n" + "="*80)
    print("Configuration Summary:")
    print("="*80)
    print(f"Training data:    {data_dir}")
    print(f"Validation data:  {val_dir if val_dir else 'None (skipped)'}")
    print(f"Pretrained model: {pretrained}")
    print(f"Output directory: {out_dir}")
    print("="*80)
    
    confirm = input("\nProceed with these settings? [Y/n]: ").strip().lower()
    if confirm and confirm not in ['y', 'yes']:
        print("Aborted.")
        sys.exit(0)
    
    return data_dir, val_dir, pretrained, out_dir


def main():
    parser = argparse.ArgumentParser()
    # 必須引数をオプショナルに変更
    parser.add_argument('--data-dir', type=str, default=None, help='学習用データディレクトリ (ImageFolder形式)')
    parser.add_argument('--pretrained', type=str, default=None, help='ImageNet事前学習Guided-Diffusionのチェックポイント .pt')
    parser.add_argument('--out-dir', type=str, default=None, help='出力先ディレクトリ')
    parser.add_argument('--interactive', action='store_true', help='対話モードで実行')
    
    # データ関連
    parser.add_argument('--val-dir', type=str, default=None, help='検証用データディレクトリ (省略時は検証なし)')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=2, help='バッチサイズ (GPUメモリに応じて調整、デフォルト2)')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--augmentation', type=str, default='medium', choices=['light', 'medium', 'heavy'],
                        help='データ拡張の強度')
    
    # 学習ハイパーパラメータ
    parser.add_argument('--total-steps', type=int, default=1000000, help='総学習ステップ数')
    parser.add_argument('--accum-steps', type=int, default=8, help='Gradient Accumulation ステップ数 (デフォルト8)')
    parser.add_argument('--lr', type=float, default=1e-4, help='学習率')
    parser.add_argument('--min-lr', type=float, default=1e-7, help='最小学習率(cosine annealing)')
    parser.add_argument('--warmup-steps', type=int, default=5000, help='Warmupステップ数')
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--ema-decay', type=float, default=0.9999)
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping (0で無効化)')
    parser.add_argument('--use-checkpoint', action='store_true', help='Gradient checkpointing有効化(メモリ節約)')
    
    # ロギング・保存
    parser.add_argument('--save-interval', type=int, default=10000, help='チェックポイント保存間隔')
    parser.add_argument('--val-interval', type=int, default=5000, help='検証実行間隔')
    parser.add_argument('--val-batches', type=int, default=50, help='検証時の最大バッチ数')
    parser.add_argument('--log-interval', type=int, default=100, help='ログ出力間隔')
    
    # その他
    parser.add_argument('--resume', type=str, default=None, help='学習再開用のtraining_state_*.ptパス')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true', help='FP16混合精度学習を有効化')
    parser.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda'])

    args = parser.parse_args()

    # 引数が指定されていない場合は対話モードまたはデフォルト値を使用
    if args.data_dir is None or args.pretrained is None or args.out_dir is None or args.interactive:
        print("\nNo command-line arguments provided. Starting interactive setup...\n")
        data_dir, val_dir_interactive, pretrained, out_dir = interactive_setup()
        
        # コマンドライン引数が優先、なければ対話モードの値を使用
        args.data_dir = args.data_dir or data_dir
        args.pretrained = args.pretrained or pretrained
        args.out_dir = args.out_dir or out_dir
        if args.val_dir is None:
            args.val_dir = val_dir_interactive
    
    # パスの存在確認
    if not os.path.exists(args.data_dir):
        print(f"Error: Training data directory not found: {args.data_dir}")
        sys.exit(1)
    
    if args.val_dir and not os.path.exists(args.val_dir):
        print(f"Warning: Validation data directory not found: {args.val_dir}")
        print("Continuing without validation...")
        args.val_dir = None
    
    if not os.path.exists(args.pretrained):
        print(f"Error: Pretrained model not found: {args.pretrained}")
        sys.exit(1)

    # シード固定
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # デバイス設定
    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"FP16: {args.fp16 and device.type == 'cuda'}")

    # 出力ディレクトリ作成
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 設定をJSONで保存
    config_path = os.path.join(args.out_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to: {config_path}")

    # ロガー設定
    logger.configure(dir=args.out_dir)

    # データローダー
    print("Building data loaders...")
    train_dl = build_dataloader(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=True,
        augmentation_intensity=args.augmentation,
    )
    print(f"Train dataset size: {len(train_dl.dataset)}")
    
    val_dl = None
    if args.val_dir:
        val_dl = build_dataloader(
            data_dir=args.val_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            is_train=False,
        )
        print(f"Validation dataset size: {len(val_dl.dataset)}")

    # モデルと拡散プロセス (ImageNet 256x256 unconditional)
    print("Creating model and diffusion...")
    model_config = {
        'attention_resolutions': '32,16,8',
        'class_cond': False,
        'diffusion_steps': 1000,
        'image_size': args.image_size,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 256,
        'num_head_channels': 64,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_fp16': args.fp16,
        'use_scale_shift_norm': True,
    }

    model, diffusion = create_model_and_diffusion(
        **model_config,
        timestep_respacing='',
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        use_checkpoint=args.use_checkpoint,  # Gradient checkpointing
        use_new_attention_order=False,
        dropout=0.0,
        channel_mult='',
        num_heads=4,
        num_heads_upsample=-1,
    )

    # 事前学習重みロード
    print(f"Loading pretrained weights from: {args.pretrained}")
    state_dict = torch.load(args.pretrained, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.train()
    
    # パラメータ数とメモリ使用量表示
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

    # オプティマイザ & スケジューラ
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        base_lr=args.lr,
        min_lr=args.min_lr,
    )
    
    use_fp16 = bool(args.fp16 and device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    
    ema = EMA(model, decay=args.ema_decay)

    # Resume処理
    start_step = 0
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"Resuming from: {args.resume}")
        start_step, train_losses, val_losses, best_val_loss = load_training_state(
            args.resume, model, ema, optimizer, scheduler, scaler
        )
        print(f"Resumed from step {start_step}")

    # 学習ループ
    print(f"\nStarting training from step {start_step} to {args.total_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.accum_steps}")
    print(f"Effective batch size: {args.batch_size * args.accum_steps}")
    print(f"Gradient checkpointing: {args.use_checkpoint}")
    print(f"FP16: {use_fp16}")
    
    if torch.cuda.is_available():
        print(f"\nInitial GPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    step = start_step
    data_iter = iter(train_dl)
    pbar = tqdm(
        initial=start_step,
        total=args.total_steps,
        desc='Finetuning',
        dynamic_ncols=True,
    )
    
    accum_loss = 0.0
    accum_mse = 0.0
    accum_vb = 0.0

    while step < args.total_steps:
        # Gradient accumulation loop
        optimizer.zero_grad(set_to_none=True)
        
        try:
            for micro_step in range(args.accum_steps):
                try:
                    images, _ = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_dl)
                    images, _ = next(data_iter)

                images = images.to(device, non_blocking=True)
                t = torch.randint(
                    low=0,
                    high=diffusion.num_timesteps,
                    size=(images.size(0),),
                    device=device,
                    dtype=torch.long,
                )

                with torch.cuda.amp.autocast(enabled=use_fp16):
                    losses = diffusion.training_losses(model, images, t, model_kwargs={})
                    loss = losses["loss"].mean() / args.accum_steps

                scaler.scale(loss).backward()
                
                # 累積値記録
                accum_loss += loss.item() * args.accum_steps
                accum_mse += losses.get("mse", torch.tensor(0.0, device=device)).mean().item() / args.accum_steps
                accum_vb += losses.get("vb", torch.tensor(0.0, device=device)).mean().item() / args.accum_steps
                
                # メモリ解放
                del images, t, losses, loss
                if micro_step % 2 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n{'='*80}")
                print("GPU OUT OF MEMORY ERROR")
                print(f"{'='*80}")
                print(f"Current settings:")
                print(f"  Batch size: {args.batch_size}")
                print(f"  Accumulation steps: {args.accum_steps}")
                print(f"  Image size: {args.image_size}")
                print(f"  FP16: {use_fp16}")
                print(f"  Gradient checkpointing: {args.use_checkpoint}")
                print(f"\nSuggestions:")
                print(f"  1. Reduce batch size: --batch-size 1")
                print(f"  2. Enable gradient checkpointing: --use-checkpoint")
                print(f"  3. Enable FP16 if not already: --fp16")
                print(f"  4. Increase accumulation steps: --accum-steps 16")
                print(f"{'='*80}")
                if torch.cuda.is_available():
                    print(f"GPU Memory at error:")
                    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                    print(f"  Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
                raise e
            else:
                raise e

        # Gradient clipping
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Scheduler step
        current_lr = scheduler.step()
        
        # EMA更新
        ema.update(model)
        
        step += 1
        pbar.update(1)

        # ロギング
        if step % args.log_interval == 0:
            avg_loss = accum_loss / args.log_interval
            avg_mse = accum_mse / args.log_interval
            avg_vb = accum_vb / args.log_interval
            
            logger.logkv("step", step)
            logger.logkv("loss", avg_loss)
            logger.logkv("mse", avg_mse)
            logger.logkv("vb", avg_vb)
            logger.logkv("lr", current_lr)
            logger.dumpkvs()
            
            train_losses.append((step, avg_loss))
            
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}',
            })
            
            accum_loss = 0.0
            accum_mse = 0.0
            accum_vb = 0.0

        # 検証
        if val_dl is not None and step % args.val_interval == 0:
            print(f"\nRunning validation at step {step}...")
            val_loss, val_mse, val_vb = evaluate_model(
                model, diffusion, val_dl, device, max_batches=args.val_batches
            )
            print(f"Validation - Loss: {val_loss:.4f}, MSE: {val_mse:.4f}, VB: {val_vb:.4f}")
            
            logger.logkv("step", step)
            logger.logkv("val_loss", val_loss)
            logger.logkv("val_mse", val_mse)
            logger.logkv("val_vb", val_vb)
            logger.dumpkvs()
            
            val_losses.append((step, val_loss))
            
            # Best model保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ema_sd = ema.state_dict(model)
                best_path = os.path.join(args.out_dir, f'best_ema_{args.ema_decay}.pt')
                torch.save(ema_sd, best_path)
                print(f"New best model saved: {best_path} (val_loss: {val_loss:.4f})")

        # チェックポイント保存
        if step % args.save_interval == 0 or step == args.total_steps:
            print(f"\nSaving checkpoints at step {step}...")
            
            # 通常モデル
            path_model = save_checkpoint(
                model.state_dict(), args.out_dir, prefix="model", step=step
            )
            
            # EMAモデル
            ema_sd = ema.state_dict(model)
            path_ema = save_checkpoint(
                ema_sd, args.out_dir, prefix=f"ema_{args.ema_decay}", step=step
            )
            
            # 学習状態 (resume用)
            path_state = save_training_state(
                args.out_dir, step, model, ema, optimizer, scheduler, scaler,
                train_losses, val_losses, best_val_loss
            )
            
            print(f"Saved:\n  - {path_model}\n  - {path_ema}\n  - {path_state}")
            
            # 学習曲線プロット
            plot_training_curves(train_losses, val_losses, args.out_dir)

    pbar.close()
    
    # 最終プロット
    plot_training_curves(train_losses, val_losses, args.out_dir)
    
    print("\n" + "="*80)
    print("Finetuning completed!")
    print(f"Total steps: {args.total_steps}")
    print(f"Checkpoints saved to: {args.out_dir}")
    if val_dl is not None:
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best model: {os.path.join(args.out_dir, f'best_ema_{args.ema_decay}.pt')}")
    print("="*80)


if __name__ == '__main__':
    main()

