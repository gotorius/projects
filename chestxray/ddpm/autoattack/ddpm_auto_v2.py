"""
ChestXray - DiffPure方式のAutoAttack評価
eval_sde_adv.pyを参考に、2段階評価を実装:
1. 分類器のみへのAutoAttack
2. DDPM Purification + 分類器へのAutoAttack
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
import time
from pathlib import Path

# ========== 設定 ==========
DATA_DIR = '/mnt/data1/Public/MedImages/CellData/chest_xray'
TEST_DIR = os.path.join(DATA_DIR, 'test')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ========== データセット定義 ==========
class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit=None):
        self.transform = transform
        self.samples = []
        root_path = Path(root_dir)
        class_folders = sorted([d for d in root_path.iterdir() if d.is_dir()])
        self.classes = [d.name for d in class_folders]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        for cfold in class_folders:
            cidx = self.class_to_idx[cfold.name]
            for p in cfold.glob('*'):
                if p.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(p), cidx))
        
        if limit is not None and limit > 0:
            self.samples = self.samples[:limit]
        
        print(f"Loaded {len(self.samples)} test images from {root_dir}")
        print("Classes:", self.classes)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# ========== 変換 ==========
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ========== 分類器ロード ==========
def load_classifier(ckpt_path):
    print(f"Loading classifier: {ckpt_path}")
    clf = models.resnet50(pretrained=False)
    clf.fc = nn.Linear(clf.fc.in_features, 2)
    ckpt = torch.load(ckpt_path, map_location=device)
    clf.load_state_dict(ckpt['model_state_dict'])
    clf = clf.to(device)
    clf.eval()
    print(f"Classifier loaded. Best val acc: {ckpt.get('best_val_acc', 'N/A')}")
    return clf

# ========== DDPM定義 ==========
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch)
        self.norm2 = nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, out_ch),
                nn.SiLU()
            )
        else:
            self.time_mlp = None
        
        self.act = nn.SiLU()
    
    def forward(self, x, t_emb=None):
        h = self.norm1(self.conv1(x))
        if self.time_mlp is not None and t_emb is not None:
            h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.act(h)
        h = self.norm2(self.conv2(h))
        h = self.act(h)
        return h + self.skip(x)

class SimpleUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        # Encoder
        self.enc1 = ResidualBlock(in_ch, base_ch, time_emb_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1)
        
        self.enc2 = ResidualBlock(base_ch * 2, base_ch * 2, time_emb_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1)
        
        self.enc3 = ResidualBlock(base_ch * 4, base_ch * 4, time_emb_dim)
        self.down3 = nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1)
        
        self.enc4 = ResidualBlock(base_ch * 8, base_ch * 8, time_emb_dim)
        self.down4 = nn.Conv2d(base_ch * 8, base_ch * 8, 4, 2, 1)
        
        # Bottleneck
        self.bot1 = ResidualBlock(base_ch * 8, base_ch * 8, time_emb_dim)
        self.bot2 = ResidualBlock(base_ch * 8, base_ch * 8, time_emb_dim)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch * 8, base_ch * 8, 4, 2, 1)
        self.dec4 = ResidualBlock(base_ch * 16, base_ch * 8, time_emb_dim)
        
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, 2, 1)
        self.dec3 = ResidualBlock(base_ch * 8, base_ch * 4, time_emb_dim)
        
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1)
        self.dec2 = ResidualBlock(base_ch * 4, base_ch * 2, time_emb_dim)
        
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1)
        self.dec1 = ResidualBlock(base_ch * 2, base_ch, time_emb_dim)
        
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, in_ch, 3, padding=1)
        )
    
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        e1 = self.enc1(x, t_emb); d1 = self.down1(e1)
        e2 = self.enc2(d1, t_emb); d2 = self.down2(e2)
        e3 = self.enc3(d2, t_emb); d3 = self.down3(e3)
        e4 = self.enc4(d3, t_emb); d4 = self.down4(e4)
        
        b = self.bot1(d4, t_emb)
        b = self.bot2(b, t_emb)
        
        u4 = self.up4(b); u4 = torch.cat([u4, e4], dim=1); u4 = self.dec4(u4, t_emb)
        u3 = self.up3(u4); u3 = torch.cat([u3, e3], dim=1); u3 = self.dec3(u3, t_emb)
        u2 = self.up2(u3); u2 = torch.cat([u2, e2], dim=1); u2 = self.dec2(u2, t_emb)
        u1 = self.up1(u2); u1 = torch.cat([u1, e1], dim=1); u1 = self.dec1(u1, t_emb)
        
        return self.out_conv(u1)

def load_ddpm(ckpt_path):
    print(f"Loading DDPM: {ckpt_path}")
    ddpm = SimpleUNet().to(device)
    raw = torch.load(ckpt_path, map_location=device)
    if isinstance(raw, dict) and 'model_state_dict' in raw:
        ddpm.load_state_dict(raw['model_state_dict'])
    else:
        ddpm.load_state_dict(raw)
    ddpm.eval()
    print("DDPM loaded.")
    return ddpm

# ========== 拡散スケジュール ==========
T_steps = 1000
betas = torch.linspace(1e-4, 0.02, T_steps, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
posterior_variance = torch.zeros_like(betas)
posterior_variance[1:] = betas[1:] * (1.0 - alphas_cumprod[:-1]) / (1.0 - alphas_cumprod[1:])
posterior_variance[0] = 1e-8

# ========== 正規化ツール ==========
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

def denormalize(x):
    """ImageNet正規化を解除 -> [0,1]"""
    return x * imagenet_std + imagenet_mean

def renormalize(x):
    """[0,1] -> ImageNet正規化"""
    return (x - imagenet_mean) / imagenet_std

def to_ddpm_space(x_norm):
    """ImageNet正規化 -> DDPM空間 [-1,1]"""
    x_pix = denormalize(x_norm)  # [0,1]
    return (x_pix - 0.5) / 0.5   # [-1,1]

def from_ddpm_space(x_minus1):
    """DDPM空間 [-1,1] -> ImageNet正規化"""
    x_pix = x_minus1 * 0.5 + 0.5  # [0,1]
    x_pix = torch.clamp(x_pix, 0.0, 1.0)
    return renormalize(x_pix)

# ========== DDPM Purification ==========
def diffusion_purify(x_adv_minus1, model, start_t=80, steps=100, eta=0.0, enable_grad=False):
    """
    DiffPure式のpurification
    start_t: ノイズ追加のタイムステップ
    steps: デノイジングのステップ数 (T_purify)
    enable_grad: Trueの場合、勾配計算を有効化（AutoAttack用）
    """
    if not enable_grad:
        # 通常の評価時は勾配不要
        with torch.no_grad():
            return _diffusion_purify_impl(x_adv_minus1, model, start_t, steps, eta)
    else:
        # AutoAttack時は勾配必要
        return _diffusion_purify_impl(x_adv_minus1, model, start_t, steps, eta)

def _diffusion_purify_impl(x_adv_minus1, model, start_t=80, steps=100, eta=0.0):
    """Purificationの実装（勾配の有無は呼び出し元で制御）"""
    b = x_adv_minus1.size(0)
    
    # Forward: x_0 -> x_{start_t}
    t0 = torch.full((b,), start_t, device=device, dtype=torch.long)
    noise = torch.randn_like(x_adv_minus1)
    sqrt_alpha_bar = torch.sqrt(alphas_cumprod[t0]).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alphas_cumprod[t0]).view(-1, 1, 1, 1)
    x_t = sqrt_alpha_bar * x_adv_minus1 + sqrt_one_minus_alpha_bar * noise
    
    # Reverse: x_{start_t} -> x_0
    for t_ in range(start_t, max(start_t - steps, 0), -1):
        tb = torch.full((b,), t_, device=device, dtype=torch.long)
        eps_pred = model(x_t, tb)
        
        alpha_t = alphas[t_]
        alpha_bar_t = alphas_cumprod[t_]
        
        # 平均の計算
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_pred
        )
        
        if t_ > 0:
            z = torch.randn_like(x_t)
            sigma = eta * torch.sqrt(posterior_variance[t_])
            x_t = mean + sigma * z
        else:
            x_t = mean
        
        x_t = torch.clamp(x_t, -1.0, 1.0)
    
    return x_t

# ========== 防御モデル (DiffPure式) ==========
class DDPM_Defense_Model(nn.Module):
    """
    DiffPureのSDE_Adv_Modelに相当
    入力: ImageNet正規化された画像
    出力: 分類ロジット
    """
    def __init__(self, ddpm, classifier, start_t=80, purify_steps=100):
        super().__init__()
        self.ddpm = ddpm
        self.classifier = classifier
        self.start_t = start_t
        self.purify_steps = purify_steps
        self.counter = 0
        self.enable_grad = False  # デフォルトは勾配なし
    
    def reset_counter(self):
        self.counter = 0
    
    def set_grad_enabled(self, enable):
        """AutoAttack時はTrueに設定"""
        self.enable_grad = enable
    
    def forward(self, x_norm):
        """
        x_norm: ImageNet正規化された画像 (N, 3, 224, 224)
        """
        if self.counter % 5 == 0:
            print(f'Purification times: {self.counter}')
        
        # ImageNet正規化 -> DDPM空間 [-1,1]
        x_minus1 = to_ddpm_space(x_norm)
        
        # DDPM purification
        start_time = time.time()
        x_purified_minus1 = diffusion_purify(
            x_minus1, 
            self.ddpm, 
            start_t=self.start_t, 
            steps=self.purify_steps,
            enable_grad=self.enable_grad  # 勾配計算の有効/無効を制御
        )
        elapsed = time.time() - start_time
        
        if self.counter % 5 == 0:
            print(f'Purification time per batch: {elapsed:.2f}s')
        
        # DDPM空間 -> ImageNet正規化
        x_purified_norm = from_ddpm_space(x_purified_minus1)
        
        # 分類
        logits = self.classifier(x_purified_norm)
        
        self.counter += 1
        return logits

# ========== AutoAttack評価関数 (eval_sde_adv.py方式) ==========
from autoattack import AutoAttack

def get_accuracy(model, x, y, bs=32):
    """精度計算"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(0, len(x), bs):
            x_batch = x[i:i+bs]
            y_batch = y[i:i+bs]
            logits = model(x_batch)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    
    return correct / total if total > 0 else 0.0

def eval_autoattack(classifier, defense_model, x_val, y_val, args, log_dir):
    """
    DiffPureのeval_autoattack関数に相当
    1. 分類器のみへの攻撃
    2. 防御モデル(DDPM+分類器)への攻撃
    """
    attack_version = args['attack_version']  # 'standard', 'custom'
    
    if attack_version == 'standard':
        attack_list = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
    elif attack_version == 'custom':
        attack_list = args['attack_type'].split(',')
    else:
        raise NotImplementedError(f'Unknown attack version: {attack_version}')
    
    print(f'attack_version: {attack_version}, attack_list: {attack_list}')
    
    # ========== ① 分類器のみへの攻撃 ==========
    print(f"\n{'='*70}")
    print(f"Phase 1: Apply AutoAttack to CLASSIFIER ONLY [{args['lp_norm']}]")
    print(f"{'='*70}")
    
    # AutoAttackの初期化（standardではattacks_to_runを指定しない）
    if attack_version == 'standard':
        adversary_clf = AutoAttack(
            classifier, 
            norm=args['lp_norm'], 
            eps=args['adv_eps'],
            version=attack_version,
            log_path=f'{log_dir}/log_classifier.txt', 
            device=device,
            verbose=True
        )
    else:  # custom
        adversary_clf = AutoAttack(
            classifier, 
            norm=args['lp_norm'], 
            eps=args['adv_eps'],
            version=attack_version, 
            attacks_to_run=attack_list,
            log_path=f'{log_dir}/log_classifier.txt', 
            device=device,
            verbose=True
        )
        adversary_clf.apgd.n_restarts = 1
        adversary_clf.square.n_queries = 5000
    
    print(f'{args["lp_norm"]}, epsilon: {args["adv_eps"]}')
    print(f'Running AutoAttack with attacks: {attack_list}')
    print(f'This may take a while (especially with standard version)...')
    
    # 攻撃実行
    x_adv_clf = adversary_clf.run_standard_evaluation(x_val, y_val, bs=args['batch_size'])
    print(f'x_adv_classifier shape: {x_adv_clf.shape}')
    
    # 保存
    torch.save([x_adv_clf, y_val], f'{log_dir}/x_adv_classifier.pt')
    
    # ========== ② 防御モデル(DDPM+分類器)への攻撃 ==========
    print(f"\n{'='*70}")
    print(f"Phase 2: Apply AutoAttack to DEFENSE MODEL (DDPM+Classifier) [{args['lp_norm']}]")
    print(f"{'='*70}")
    
    defense_model.reset_counter()
    defense_model.set_grad_enabled(True)  # AutoAttack用に勾配を有効化
    
    # AutoAttackの初期化（standardではattacks_to_runを指定しない）
    if attack_version == 'standard':
        adversary_defense = AutoAttack(
            defense_model,
            norm=args['lp_norm'],
            eps=args['adv_eps'],
            version=attack_version,
            log_path=f'{log_dir}/log_defense.txt',
            device=device,
            verbose=True
        )
    else:  # custom
        adversary_defense = AutoAttack(
            defense_model,
            norm=args['lp_norm'],
            eps=args['adv_eps'],
            version=attack_version,
            attacks_to_run=attack_list,
            log_path=f'{log_dir}/log_defense.txt',
            device=device,
            verbose=True
        )
        adversary_defense.apgd.n_restarts = 1
        adversary_defense.square.n_queries = 5000
    
    print(f'{args["lp_norm"]}, epsilon: {args["adv_eps"]}')
    print(f'Running AutoAttack with attacks: {attack_list}')
    print(f'This may take a while (especially with standard version)...')
    
    # 攻撃実行
    x_adv_defense = adversary_defense.run_standard_evaluation(x_val, y_val, bs=args['batch_size'])
    print(f'x_adv_defense shape: {x_adv_defense.shape}')
    
    # 攻撃終了後は勾配を無効化
    defense_model.set_grad_enabled(False)
    
    # 保存
    torch.save([x_adv_defense, y_val], f'{log_dir}/x_adv_defense.pt')
    
    return x_adv_clf, x_adv_defense

# ========== メイン評価 ==========
def main():
    # パラメータ設定
    args = {
        'classifier_ckpt': '/mnt/data1/gotou/projects/chestxray/resnet/resnet50_best.pth',
        'ddpm_ckpt': '/mnt/data1/gotou/projects/chestxray/ddpm/ddpm_out/ddpm_epoch100.pth',
        'start_t': 80,           # グリッドサーチ結果の最適値
        'purify_steps': 50,      # T_purify (メモリ節約のため50に削減、本来は100)
        'adv_eps': 8/255.0,      # 医用画像用に調整可能 (0.01~0.03推奨)
        'lp_norm': 'Linf',
        'attack_version': 'standard',  # 'standard': 4つ全ての攻撃 (論文と同じ)
        'attack_type': 'apgd-ce,apgd-t,fab-t,square',  # standardの場合は使用されない
        'batch_size': 4,         # 勾配計算のためメモリ使用量大 → 小さく設定
        'num_samples': 200,      # 評価サンプル数 (全サンプルの場合は-1またはNone)
        'log_dir': '/mnt/data1/gotou/projects/chestxray/ddpm/autoattack/results_v2_standard'
    }
    
    os.makedirs(args['log_dir'], exist_ok=True)
    
    # メモリ最適化
    torch.cuda.empty_cache()
    print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    
    print(f"\n{'='*70}")
    print("ChestXray - DiffPure-style AutoAttack Evaluation")
    print(f"{'='*70}")
    print(f"Classifier: {args['classifier_ckpt']}")
    print(f"DDPM: {args['ddpm_ckpt']}")
    print(f"Purification: start_t={args['start_t']}, steps={args['purify_steps']}")
    attack_desc = 'standard (apgd-ce, apgd-t, fab-t, square)' if args['attack_version'] == 'standard' else f"{args['attack_version']} ({args['attack_type']})"
    print(f"Attack: {attack_desc}, eps={args['adv_eps']:.4f}")
    print(f"Samples: {args['num_samples'] if args['num_samples'] else 'ALL'}")
    print(f"{'='*70}\n")
    
    # データロード
    test_dataset = ChestXrayDataset(
        TEST_DIR, 
        transform=test_transform, 
        limit=args['num_samples'] if args['num_samples'] and args['num_samples'] > 0 else None
    )
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=4)
    
    # 全データをメモリに
    print("Loading all test data...")
    x_all, y_all = [], []
    for x, y in test_loader:
        x_all.append(x)
        y_all.append(y)
    x_val = torch.cat(x_all, dim=0).to(device)
    y_val = torch.cat(y_all, dim=0).to(device)
    print(f"Total test samples: {len(x_val)}")
    
    # モデルロード
    classifier = load_classifier(args['classifier_ckpt'])
    ddpm = load_ddpm(args['ddpm_ckpt'])
    
    # 防御モデル構築
    defense_model = DDPM_Defense_Model(
        ddpm, 
        classifier, 
        start_t=args['start_t'], 
        purify_steps=args['purify_steps']
    ).to(device)
    defense_model.eval()
    
    # ========== クリーン精度の測定 ==========
    print("\n" + "="*70)
    print("Measuring Clean Accuracy")
    print("="*70)
    
    clean_acc_clf = get_accuracy(classifier, x_val, y_val, bs=args['batch_size'])
    print(f"Classifier (clean): {clean_acc_clf:.4f}")
    
    defense_model.reset_counter()
    clean_acc_defense = get_accuracy(defense_model, x_val, y_val, bs=args['batch_size'])
    print(f"Defense Model (clean): {clean_acc_defense:.4f}")
    
    # ========== AutoAttack評価 ==========
    x_adv_clf, x_adv_defense = eval_autoattack(
        classifier, 
        defense_model, 
        x_val, 
        y_val, 
        args, 
        args['log_dir']
    )
    
    # ========== ロバスト精度の測定 ==========
    print("\n" + "="*70)
    print("Measuring Robust Accuracy")
    print("="*70)
    
    robust_acc_clf = get_accuracy(classifier, x_adv_clf, y_val, bs=args['batch_size'])
    print(f"Classifier (adversarial): {robust_acc_clf:.4f}")
    
    defense_model.reset_counter()
    robust_acc_defense = get_accuracy(defense_model, x_adv_defense, y_val, bs=args['batch_size'])
    print(f"Defense Model (adversarial): {robust_acc_defense:.4f}")
    
    # ========== 結果まとめ ==========
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Total samples: {len(x_val)}")
    print(f"\nClassifier Only:")
    print(f"  Clean Accuracy:      {clean_acc_clf:.4f}")
    print(f"  Robust Accuracy:     {robust_acc_clf:.4f}")
    print(f"  Attack Success Rate: {1 - robust_acc_clf/clean_acc_clf:.4f}")
    
    print(f"\nDefense Model (DDPM + Classifier):")
    print(f"  Clean Accuracy:      {clean_acc_defense:.4f}")
    print(f"  Robust Accuracy:     {robust_acc_defense:.4f}")
    print(f"  Attack Success Rate: {1 - robust_acc_defense/clean_acc_defense:.4f}")
    
    print(f"\nDefense Improvement:")
    print(f"  Robust Acc Gain:     {robust_acc_defense - robust_acc_clf:+.4f}")
    print(f"  Relative Improvement: {(robust_acc_defense/robust_acc_clf - 1)*100:+.2f}%")
    
    # ========== サマリー保存 ==========
    summary_path = os.path.join(args['log_dir'], 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ChestXray - DiffPure-style AutoAttack Evaluation\n")
        f.write("="*70 + "\n\n")
        f.write(f"Classifier: {args['classifier_ckpt']}\n")
        f.write(f"DDPM: {args['ddpm_ckpt']}\n")
        f.write(f"Purification: start_t={args['start_t']}, steps={args['purify_steps']}\n")
        attack_desc = 'standard (apgd-ce, apgd-t, fab-t, square)' if args['attack_version'] == 'standard' else f"{args['attack_version']} ({args['attack_type']})"
        f.write(f"Attack: {attack_desc}, eps={args['adv_eps']:.4f}\n")
        f.write(f"Total samples: {len(x_val)}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("Classifier Only:\n")
        f.write(f"  Clean Accuracy:      {clean_acc_clf:.4f}\n")
        f.write(f"  Robust Accuracy:     {robust_acc_clf:.4f}\n")
        f.write(f"  Attack Success Rate: {1 - robust_acc_clf/clean_acc_clf:.4f}\n\n")
        
        f.write("Defense Model (DDPM + Classifier):\n")
        f.write(f"  Clean Accuracy:      {clean_acc_defense:.4f}\n")
        f.write(f"  Robust Accuracy:     {robust_acc_defense:.4f}\n")
        f.write(f"  Attack Success Rate: {1 - robust_acc_defense/clean_acc_defense:.4f}\n\n")
        
        f.write("Defense Improvement:\n")
        f.write(f"  Robust Acc Gain:     {robust_acc_defense - robust_acc_clf:+.4f}\n")
        f.write(f"  Relative Improvement: {(robust_acc_defense/robust_acc_clf - 1)*100:+.2f}%\n")
    
    print(f"\nResults saved to: {args['log_dir']}")
    print(f"Summary: {summary_path}")
    print("\nEvaluation complete!")

if __name__ == '__main__':
    main()
