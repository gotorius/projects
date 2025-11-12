"""
DermMel (Melanoma分類) - FGSM攻撃 + JPEG圧縮防御検証スクリプト
JPEG圧縮による敵対的画像の浄化と防御性能評価
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix
import io

# ========== 設定 ==========
DATA_DIR = '/mnt/data1/Public/MedImages/DermMel'
TRAIN_DIR = os.path.join(DATA_DIR, 'train_sep')
VALID_DIR = os.path.join(DATA_DIR, 'valid')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ========== データセット定義 ==========
class DermMelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        DermMelデータセット（ImageFolder形式）
        root_dir: Melanoma/ と NotMelanoma/ を含むディレクトリ
        """
        from pathlib import Path
        self.transform = transform
        self.samples = []
        
        # クラスフォルダを探索
        root_path = Path(root_dir)
        class_folders = sorted([d for d in root_path.iterdir() if d.is_dir()])
        
        # クラス名とインデックスのマッピング
        self.classes = [d.name for d in class_folders]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 画像ファイルを収集
        for class_folder in class_folders:
            class_idx = self.class_to_idx[class_folder.name]
            for img_path in class_folder.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                    self.samples.append((str(img_path), class_idx))
        
        print(f"Found {len(self.samples)} images in {root_dir}")
        print(f"Classes: {self.classes}")
        print(f"Class to index: {self.class_to_idx}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# データ変換（検証データ用）
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 検証データセット
val_dataset = DermMelDataset(VALID_DIR, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

print(f"Validation samples: {len(val_dataset)}")

# ========== 分類器の読み込み ==========
print("\n" + "="*70)
print("Loading ResNet50 classifier...")
print("="*70)

clf_ckpt = "/mnt/data1/gotou/projects/dermmel/resnet50_models/resnet50_best.pth"

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2クラス分類
model = model.to(device)

# 重みロード
checkpoint = torch.load(clf_ckpt, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded classifier from {clf_ckpt}")
print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
print(f"Classes: {checkpoint.get('class_names', ['Melanoma', 'NotMelanoma'])}")

# ========== 正規化パラメータ ==========
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

def denormalize(x_norm, mean, std):
    """正規化された画像をピクセル空間[0,1]に戻す"""
    return x_norm * std + mean

def renormalize(x_pixel, mean, std):
    """ピクセル空間[0,1]の画像を正規化"""
    return (x_pixel - mean) / std

# ========== FGSM攻撃関数 ==========
def fgsm_attack_improved(model, images, labels, epsilon_pixel, device,
                         mean_tensor, std_tensor, return_preds=True):
    """
    正規化を考慮したFGSM攻撃
    
    Args:
        model: 分類モデル
        images: 正規化済み画像 [B, C, H, W]
        labels: ラベル [B]
        epsilon_pixel: 摂動の大きさ（ピクセルスケール 0-1）
        device: デバイス
        mean_tensor: 正規化平均 [1, C, 1, 1]
        std_tensor: 正規化標準偏差 [1, C, 1, 1]
        return_preds: 攻撃後の予測を返すか
    
    Returns:
        adv_images: 敵対的画像（正規化済み）
        adv_preds: 攻撃後の予測（オプション）
    """
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    images.requires_grad = True
    
    # Forward pass
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    grad = images.grad.data
    
    # 勾配の符号
    grad_sign = grad.sign()
    
    # εをチャネルごとに正規化空間に変換
    if not torch.is_tensor(epsilon_pixel):
        eps_pixel_tensor = torch.tensor(epsilon_pixel, dtype=images.dtype, device=device)
    else:
        eps_pixel_tensor = epsilon_pixel.to(device).to(images.dtype)
    
    eps_norm = (eps_pixel_tensor / std_tensor).view(1, -1, 1, 1)
    
    # 敵対的画像を生成（正規化空間）
    adv_images = images + eps_norm * grad_sign
    
    # ピクセル空間でクリッピング
    adv_pixel = denormalize(adv_images, mean_tensor, std_tensor)
    adv_pixel = torch.clamp(adv_pixel, 0.0, 1.0)
    
    # 正規化空間に戻す
    adv_images = renormalize(adv_pixel, mean_tensor, std_tensor).detach()
    adv_images.requires_grad = False
    
    if return_preds:
        with torch.no_grad():
            adv_outputs = model(adv_images)
            adv_preds = torch.argmax(adv_outputs, dim=1)
        
        # メモリ解放
        del grad, grad_sign, adv_pixel, eps_norm, loss
        torch.cuda.empty_cache()
        return adv_images, adv_preds
    
    del grad, grad_sign, adv_pixel, eps_norm, loss
    torch.cuda.empty_cache()
    return adv_images

# ========== JPEG圧縮防御関数 ==========
def jpeg_compress_defense(images_norm, quality=75, mean_tensor=None, std_tensor=None):
    """
    JPEG圧縮による防御
    
    Args:
        images_norm: 正規化済み画像 [B, C, H, W]
        quality: JPEG品質 (0-100, 高いほど高品質)
        mean_tensor: 正規化平均
        std_tensor: 正規化標準偏差
    
    Returns:
        compressed_images_norm: JPEG圧縮後の正規化画像
    """
    if mean_tensor is None:
        mean_tensor = imagenet_mean
    if std_tensor is None:
        std_tensor = imagenet_std
    
    # ピクセル空間に変換
    images_pixel = denormalize(images_norm, mean_tensor, std_tensor)
    
    batch_size = images_pixel.size(0)
    compressed_images = []
    
    for i in range(batch_size):
        # テンソルをPIL画像に変換
        img_np = images_pixel[i].cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        
        # JPEG圧縮
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_pil = Image.open(buffer)
        
        # テンソルに戻す
        compressed_np = np.array(compressed_pil).astype(np.float32) / 255.0
        compressed_tensor = torch.from_numpy(compressed_np).permute(2, 0, 1)
        compressed_images.append(compressed_tensor)
    
    # バッチテンソルに結合
    compressed_batch = torch.stack(compressed_images).to(images_norm.device)
    
    # 正規化空間に戻す
    compressed_norm = renormalize(compressed_batch, mean_tensor, std_tensor)
    
    return compressed_norm

# ========== 評価ループ ==========
print("\n" + "="*70)
print("Starting evaluation...")
print("="*70)

# 実験パラメータ
epsilon_pixel = 8/255.0
jpeg_qualities = [11]  # 複数のJPEG品質でテスト
save_examples_dir = "/mnt/data1/gotou/projects/dermmel/jpeg/fgsm/defense_results"
os.makedirs(save_examples_dir, exist_ok=True)

# 各品質レベルでの結果を保存
all_results = {}

for jpeg_quality in jpeg_qualities:
    print(f"\n{'='*70}")
    print(f"Testing JPEG Quality: {jpeg_quality}")
    print(f"{'='*70}")
    
    # 保存用ディレクトリ
    quality_dir = os.path.join(save_examples_dir, f"quality_{jpeg_quality}")
    save_triplets_dir = os.path.join(quality_dir, "triplets")
    save_clean_dir = os.path.join(quality_dir, "clean")
    save_adv_dir = os.path.join(quality_dir, "adversarial")
    save_compressed_dir = os.path.join(quality_dir, "compressed")
    for d in [quality_dir, save_triplets_dir, save_clean_dir, save_adv_dir, save_compressed_dir]:
        os.makedirs(d, exist_ok=True)
    
    MAX_IMAGES_TO_SAVE = 20
    saved_image_count = 0
    
    # 統計変数
    total = 0
    correct_clean = 0
    correct_adv = 0
    correct_compressed = 0
    
    all_labels = []
    all_preds_clean = []
    all_preds_adv = []
    all_preds_compressed = []
    
    l2_norms_adv = []
    linf_norms_adv = []
    l2_norms_compressed = []
    linf_norms_compressed = []
    
    # 評価ループ
    for batch_idx, (images_norm, labels) in enumerate(tqdm(val_loader, desc=f"Q={jpeg_quality}")):
        images_norm = images_norm.to(device)
        labels = labels.to(device).long()
        b = images_norm.size(0)
        
        # 1) Clean prediction
        with torch.no_grad():
            logits_clean = model(images_norm)
            preds_clean = torch.argmax(logits_clean, dim=1)
        
        # 元画像で正解した画像のみフィルタリング
        correct_mask = (preds_clean == labels)
        correct_indices = torch.where(correct_mask)[0]
        
        if len(correct_indices) == 0:
            continue
        
        images_norm_correct = images_norm[correct_indices]
        labels_correct = labels[correct_indices]
        preds_clean_correct = preds_clean[correct_indices]
        
        total += len(correct_indices)
        correct_clean += len(correct_indices)
        all_labels.extend(labels_correct.cpu().numpy())
        all_preds_clean.extend(preds_clean_correct.cpu().numpy())
        
        # 2) FGSM攻撃
        adv_images_norm, adv_preds = fgsm_attack_improved(
            model=model,
            images=images_norm_correct,
            labels=labels_correct,
            epsilon_pixel=epsilon_pixel,
            device=device,
            mean_tensor=imagenet_mean,
            std_tensor=imagenet_std,
            return_preds=True
        )
        
        correct_adv += (adv_preds == labels_correct).sum().item()
        all_preds_adv.extend(adv_preds.cpu().numpy())
        
        # L2/L∞ノルム計算（敵対的画像）
        clean_pixel = denormalize(images_norm_correct, imagenet_mean, imagenet_std)
        adv_pixel = denormalize(adv_images_norm, imagenet_mean, imagenet_std)
        diff_adv = (adv_pixel - clean_pixel).view(len(correct_indices), -1)
        l2_adv = torch.norm(diff_adv, p=2, dim=1).cpu().numpy()
        linf_adv = torch.norm(diff_adv, p=float('inf'), dim=1).cpu().numpy()
        l2_norms_adv.extend(l2_adv)
        linf_norms_adv.extend(linf_adv)
        
        # 3) JPEG圧縮防御
        compressed_images_norm = jpeg_compress_defense(
            adv_images_norm, 
            quality=jpeg_quality,
            mean_tensor=imagenet_mean,
            std_tensor=imagenet_std
        )
        
        # 4) 圧縮画像の分類
        with torch.no_grad():
            logits_compressed = model(compressed_images_norm)
            preds_compressed = torch.argmax(logits_compressed, dim=1)
            correct_compressed += (preds_compressed == labels_correct).sum().item()
            all_preds_compressed.extend(preds_compressed.cpu().numpy())
            
            # 圧縮後のノルム計算
            compressed_pixel = denormalize(compressed_images_norm, imagenet_mean, imagenet_std)
            diff_compressed = (compressed_pixel - clean_pixel).view(len(correct_indices), -1)
            l2_compressed = torch.norm(diff_compressed, p=2, dim=1).cpu().numpy()
            linf_compressed = torch.norm(diff_compressed, p=float('inf'), dim=1).cpu().numpy()
            l2_norms_compressed.extend(l2_compressed)
            linf_norms_compressed.extend(linf_compressed)
        
        # 5) 画像保存（最初のバッチのみ）
        if saved_image_count < MAX_IMAGES_TO_SAVE:
            from torchvision.utils import save_image, make_grid
            for i in range(min(len(correct_indices), MAX_IMAGES_TO_SAVE - saved_image_count)):
                idx = saved_image_count
                
                # 個別画像保存
                save_image(clean_pixel[i], os.path.join(save_clean_dir, f"clean_{idx:04d}.png"))
                save_image(adv_pixel[i], os.path.join(save_adv_dir, f"adv_{idx:04d}.png"))
                save_image(compressed_pixel[i], os.path.join(save_compressed_dir, f"compressed_{idx:04d}.png"))
                
                # トリプレット画像
                triplet = torch.stack([clean_pixel[i], adv_pixel[i], compressed_pixel[i]], dim=0)
                grid = make_grid(triplet, nrow=3, padding=5, pad_value=1.0)
                save_image(grid, os.path.join(save_triplets_dir, f"triplet_{idx:04d}.png"))
                
                saved_image_count += 1
                if saved_image_count >= MAX_IMAGES_TO_SAVE:
                    break
    
    # ========== 結果計算 ==========
    clean_acc = correct_clean / total if total > 0 else 0.0
    adv_acc = correct_adv / total if total > 0 else 0.0
    compressed_acc = correct_compressed / total if total > 0 else 0.0
    
    l2_norms_adv = np.array(l2_norms_adv)
    linf_norms_adv = np.array(linf_norms_adv)
    l2_norms_compressed = np.array(l2_norms_compressed)
    linf_norms_compressed = np.array(linf_norms_compressed)
    
    # 結果を保存
    all_results[jpeg_quality] = {
        'total': total,
        'correct_clean': correct_clean,
        'correct_adv': correct_adv,
        'correct_compressed': correct_compressed,
        'clean_acc': clean_acc,
        'adv_acc': adv_acc,
        'compressed_acc': compressed_acc,
        'defense_improvement': compressed_acc - adv_acc,
        'l2_adv_mean': l2_norms_adv.mean(),
        'l2_adv_std': l2_norms_adv.std(),
        'linf_adv_mean': linf_norms_adv.mean(),
        'linf_adv_std': linf_norms_adv.std(),
        'l2_compressed_mean': l2_norms_compressed.mean(),
        'l2_compressed_std': l2_norms_compressed.std(),
        'linf_compressed_mean': linf_norms_compressed.mean(),
        'linf_compressed_std': linf_norms_compressed.std(),
        'all_labels': all_labels,
        'all_preds_clean': all_preds_clean,
        'all_preds_adv': all_preds_adv,
        'all_preds_compressed': all_preds_compressed,
    }
    
    print(f"\n{'='*70}")
    print(f"==== Results for JPEG Quality {jpeg_quality} ====")
    print(f"{'='*70}")
    print(f"Total samples: {total}")
    print(f"Clean accuracy:      {clean_acc:.4f} ({correct_clean}/{total})")
    print(f"Adversarial accuracy:{adv_acc:.4f} ({correct_adv}/{total})")
    print(f"Compressed accuracy: {compressed_acc:.4f} ({correct_compressed}/{total})")
    print(f"Defense improvement: {compressed_acc - adv_acc:+.4f}")
    print(f"{'='*70}")
    
    # 混同行列を生成
    def plot_confusion_matrix(y_true, y_pred, title, filename):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Melanoma', 'NotMelanoma'],
                    yticklabels=['Melanoma', 'NotMelanoma'])
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        tn, fp, fn, tp = cm.ravel()
        precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
        recall = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = (2*precision*recall)/(precision+recall) if (precision+recall)>0 else 0.0
        
        return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                'precision': precision, 'recall': recall, 'f1': f1}
    
    # 混同行列を保存
    cm_clean = plot_confusion_matrix(all_labels, all_preds_clean, 
                                     f"Clean Images (Q={jpeg_quality})",
                                     os.path.join(quality_dir, "cm_clean.png"))
    cm_adv = plot_confusion_matrix(all_labels, all_preds_adv, 
                                   f"Adversarial FGSM (Q={jpeg_quality})",
                                   os.path.join(quality_dir, "cm_adversarial.png"))
    cm_compressed = plot_confusion_matrix(all_labels, all_preds_compressed, 
                                         f"JPEG Compressed (Q={jpeg_quality})",
                                         os.path.join(quality_dir, "cm_compressed.png"))
    
    # 詳細結果をCSVに保存
    stats_df = pd.DataFrame({
        'true_label': all_labels,
        'pred_clean': all_preds_clean,
        'pred_adv': all_preds_adv,
        'pred_compressed': all_preds_compressed,
        'l2_norm_adv': l2_norms_adv,
        'linf_norm_adv': linf_norms_adv,
        'l2_norm_compressed': l2_norms_compressed,
        'linf_norm_compressed': linf_norms_compressed,
    })
    
    stats_df['attack_success'] = (stats_df['pred_adv'] != stats_df['true_label']).astype(int)
    stats_df['defense_success'] = (stats_df['pred_compressed'] == stats_df['true_label']).astype(int)
    stats_df['defense_recovery'] = ((stats_df['attack_success'] == 1) & (stats_df['defense_success'] == 1)).astype(int)
    
    csv_path = os.path.join(quality_dir, 'detailed_results.csv')
    stats_df.to_csv(csv_path, index=False)
    
    # サマリー統計
    summary_path = os.path.join(quality_dir, 'summary_statistics.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"DermMel - FGSM Attack + JPEG Compression Defense (Quality={jpeg_quality})\n")
        f.write("="*70 + "\n\n")
        f.write(f"Dataset: DermMel (Melanoma vs NotMelanoma)\n")
        f.write(f"Attack: FGSM, epsilon={epsilon_pixel:.4f} ({epsilon_pixel*255:.1f}/255)\n")
        f.write(f"Defense: JPEG Compression, quality={jpeg_quality}\n")
        f.write(f"Classifier: {clf_ckpt}\n")
        f.write(f"  Best Val Acc: {checkpoint.get('best_val_acc', 'N/A')}\n\n")
        f.write("-"*70 + "\n")
        f.write(f"Results on {total} correctly classified images:\n")
        f.write("-"*70 + "\n")
        f.write(f"Clean Accuracy:      {clean_acc:.4f}\n")
        f.write(f"Adversarial Accuracy:{adv_acc:.4f}\n")
        f.write(f"Compressed Accuracy: {compressed_acc:.4f}\n")
        f.write(f"Defense Improvement: {compressed_acc - adv_acc:+.4f}\n")
        f.write(f"Attack Success Rate: {1 - adv_acc:.4f}\n")
        if (total - correct_adv) > 0:
            defense_rate = (correct_compressed - correct_adv) / (total - correct_adv)
            f.write(f"Defense Success Rate:{defense_rate:.4f}\n")
        f.write("\n" + "-"*70 + "\n")
        f.write("Perturbation Norms:\n")
        f.write("-"*70 + "\n")
        f.write("Adversarial Perturbations:\n")
        f.write(f"  L2:   mean={l2_norms_adv.mean():.4f}, std={l2_norms_adv.std():.4f}\n")
        f.write(f"  L∞:   mean={linf_norms_adv.mean():.4f}, std={linf_norms_adv.std():.4f}\n")
        f.write("\nCompressed Images (vs Clean):\n")
        f.write(f"  L2:   mean={l2_norms_compressed.mean():.4f}, std={l2_norms_compressed.std():.4f}\n")
        f.write(f"  L∞:   mean={linf_norms_compressed.mean():.4f}, std={linf_norms_compressed.std():.4f}\n")
        f.write("\n" + "-"*70 + "\n")
        f.write("Confusion Matrix Metrics:\n")
        f.write("-"*70 + "\n")
        f.write(f"Clean: Precision={cm_clean['precision']:.4f}, Recall={cm_clean['recall']:.4f}, F1={cm_clean['f1']:.4f}\n")
        f.write(f"Adversarial: Precision={cm_adv['precision']:.4f}, Recall={cm_adv['recall']:.4f}, F1={cm_adv['f1']:.4f}\n")
        f.write(f"Compressed: Precision={cm_compressed['precision']:.4f}, Recall={cm_compressed['recall']:.4f}, F1={cm_compressed['f1']:.4f}\n")
    
    print(f"✅ Saved results for quality {jpeg_quality} to {quality_dir}")

# ========== 全体サマリー ==========
print("\n" + "="*70)
print("==== Overall Summary ====")
print("="*70)

summary_df = pd.DataFrame([
    {
        'JPEG_Quality': q,
        'Clean_Acc': r['clean_acc'],
        'Adv_Acc': r['adv_acc'],
        'Compressed_Acc': r['compressed_acc'],
        'Defense_Improvement': r['defense_improvement'],
        'L2_Adv_Mean': r['l2_adv_mean'],
        'Linf_Adv_Mean': r['linf_adv_mean'],
        'L2_Compressed_Mean': r['l2_compressed_mean'],
        'Linf_Compressed_Mean': r['linf_compressed_mean'],
    }
    for q, r in all_results.items()
])

print(summary_df.to_string(index=False))

# サマリーをCSVに保存
summary_csv_path = os.path.join(save_examples_dir, 'overall_summary.csv')
summary_df.to_csv(summary_csv_path, index=False)
print(f"\n✅ Overall summary saved to: {summary_csv_path}")

# 品質vs精度のグラフを生成
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(summary_df['JPEG_Quality'], summary_df['Clean_Acc'], 'o-', label='Clean', linewidth=2, markersize=8)
plt.plot(summary_df['JPEG_Quality'], summary_df['Adv_Acc'], 's-', label='Adversarial', linewidth=2, markersize=8)
plt.plot(summary_df['JPEG_Quality'], summary_df['Compressed_Acc'], '^-', label='JPEG Compressed', linewidth=2, markersize=8)
plt.xlabel('JPEG Quality', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Accuracy vs JPEG Quality', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(summary_df['JPEG_Quality'], summary_df['Defense_Improvement'], 'o-', linewidth=2, markersize=8, color='green')
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
plt.xlabel('JPEG Quality', fontsize=12)
plt.ylabel('Defense Improvement', fontsize=12)
plt.title('Defense Improvement vs JPEG Quality', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(save_examples_dir, 'quality_comparison.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Comparison plot saved to: {plot_path}")

print("\n" + "="*70)
print("Evaluation completed successfully!")
print("="*70)
print(f"\nAll results saved in: {save_examples_dir}")
print(f"  - Individual quality results: {save_examples_dir}/quality_XX/")
print(f"  - Overall summary: {summary_csv_path}")
print(f"  - Comparison plot: {plot_path}")
