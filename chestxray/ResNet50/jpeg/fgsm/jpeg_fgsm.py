"""
ChestXray (肺炎分類) - FGSM攻撃 + JPEG圧縮防御検証スクリプト
PCamやDDPM版の出力フォーマットに合わせて、統計/CSV/トリプレット画像を保存
"""
import os
import io
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
from sklearn.metrics import confusion_matrix

# ========== 設定 ==========
DATA_DIR = '/mnt/data1/Public/MedImages/CellData/chest_xray'
TEST_DIR = os.path.join(DATA_DIR, 'test')  # NORMAL/ と PNEUMONIA/

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ========== データセット定義 ==========
class ChestXrayDataset(Dataset):
	def __init__(self, root_dir, transform=None):
		from pathlib import Path
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
		print(f"Collected {len(self.samples)} test images from {root_dir}")
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

test_dataset = ChestXrayDataset(TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
print(f"Test samples: {len(test_dataset)}")

# ========== 分類器ロード ==========
clf_ckpt = '/mnt/data1/gotou/projects/chestxray/resnet/resnet50_best.pth'
print("Loading classifier:", clf_ckpt)
clf = models.resnet50(pretrained=False)
clf.fc = nn.Linear(clf.fc.in_features, 2)
ckpt = torch.load(clf_ckpt, map_location=device)
clf.load_state_dict(ckpt['model_state_dict'])
clf = clf.to(device)
clf.eval()
print("Classifier loaded. Best val acc:", ckpt.get('best_val_acc', 'N/A'))

# ========== 正規化ツール ==========
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

def denormalize(x):
	return x * imagenet_std + imagenet_mean  # [0,1] へ

def renormalize(x):
	return (x - imagenet_mean) / imagenet_std

# ========== 攻撃 (FGSM) ==========
def _forward_logits(base_model, x_pix):
	"""
	x_pix: [0,1] tensor -> 正規化してモデルに通す
	return: logits
	"""
	x_norm = renormalize(x_pix)
	return base_model(x_norm)

def run_fgsm(base_model, x_norm, y, eps):
	"""
	base_model: 正規化入力想定の分類器
	x_norm: 正規化済み入力 [-] (B,3,H,W)
	y: 正解ラベル
	eps: ピクセル空間のLinf半径 (例: 8/255)

	返り値: adv_norm (正規化空間)
	"""
	base_model.eval()
	x_pix = denormalize(x_norm).detach().clone()
	x_pix.requires_grad_(True)

	logits = _forward_logits(base_model, x_pix)
	loss = F.cross_entropy(logits, y)
	base_model.zero_grad(set_to_none=True)
	loss.backward()
	grad_sign = x_pix.grad.detach().sign()

	x_adv_pix = x_pix + eps * grad_sign
	x_adv_pix = torch.clamp(x_adv_pix, 0.0, 1.0)
	adv_norm = renormalize(x_adv_pix).detach()
	return adv_norm

# ========== JPEG圧縮 (防御) ==========
def jpeg_compress_batch(x_pix, quality=75):
	"""
	x_pix: [0,1] float tensor (B,3,H,W)
	quality: JPEG品質 (1-100)。値が高いほど高品質/低圧縮。

	返り値: JPEG圧縮後の [0,1] float tensor (同型状)
	"""
	x_list = []
	B = x_pix.size(0)
	for i in range(B):
		img = x_pix[i].detach().clamp(0, 1).cpu()
		arr = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)  # HWC, uint8
		pil = Image.fromarray(arr)
		buf = io.BytesIO()
		# subsampling=0 で 4:4:4 を指定 (Pillow>=9)
		pil.save(buf, format='JPEG', quality=quality, subsampling=0, optimize=True)
		buf.seek(0)
		pil_j = Image.open(buf).convert('RGB')
		arr_j = np.array(pil_j).astype(np.float32) / 255.0
		ten_j = torch.from_numpy(arr_j).permute(2, 0, 1)  # CHW
		x_list.append(ten_j)
	x_j = torch.stack(x_list, dim=0).to(x_pix.device)
	return x_j

# ========== 評価設定 ==========
epsilon_pixel = 8 / 255.0
JPEG_QUALITY = 75

out_dir = f'/mnt/data1/gotou/projects/chestxray/jpeg/fgsm/results_q{JPEG_QUALITY}'
os.makedirs(out_dir, exist_ok=True)
triplet_dir = os.path.join(out_dir, 'triplets'); os.makedirs(triplet_dir, exist_ok=True)
MAX_SAVE = 3
saved = 0

# 統計
all_labels = []
all_clean = []
all_adv = []
all_pur = []
correct_clean = 0
correct_adv = 0
correct_pur = 0
total = 0
l2_adv = []
linf_adv = []
l2_pur = []
linf_pur = []

print(f"\n======================================")
print("Starting FGSM + JPEG purification evaluation (ChestXray)")
print("[Evaluation policy] Use only samples correctly classified by the clean model")
print("======================================")

for batch_idx, (x_norm, y) in enumerate(tqdm(test_loader, desc='Eval (FGSM->JPEG)')):
	x_norm = x_norm.to(device); y = y.to(device)

	# Clean preds
	with torch.no_grad():
		logits_clean = clf(x_norm)
		preds_clean = torch.argmax(logits_clean, dim=1)

	# Filter to only correctly classified clean samples
	correct_mask = (preds_clean == y)
	num_correct = int(correct_mask.sum().item())
	if num_correct == 0:
		continue

	x_norm = x_norm[correct_mask]
	y = y[correct_mask]
	preds_clean = preds_clean[correct_mask]

	# Update totals (clean subset only)
	total += x_norm.size(0)
	correct_clean += x_norm.size(0)  # all are correct by construction

	# FGSM adversarial (on filtered subset)
	adv_norm = run_fgsm(clf, x_norm, y, epsilon_pixel)
	with torch.no_grad():
		adv_logits = clf(adv_norm)
		adv_preds = torch.argmax(adv_logits, dim=1)
	correct_adv += (adv_preds == y).sum().item()

	# Purify via JPEG compression
	clean_pix = denormalize(x_norm)
	adv_pix = denormalize(adv_norm)
	pur_pix = jpeg_compress_batch(adv_pix, quality=JPEG_QUALITY)
	pur_norm = renormalize(pur_pix)
	with torch.no_grad():
		pur_logits = clf(pur_norm)
		pur_preds = torch.argmax(pur_logits, dim=1)
	correct_pur += (pur_preds == y).sum().item()

	# Norms (pixel space)
	diff_adv = (adv_pix - clean_pix).view(x_norm.size(0), -1)
	diff_pur = (pur_pix - clean_pix).view(x_norm.size(0), -1)
	l2_adv.extend(torch.norm(diff_adv, p=2, dim=1).cpu().numpy())
	linf_adv.extend(torch.norm(diff_adv, p=float('inf'), dim=1).cpu().numpy())
	l2_pur.extend(torch.norm(diff_pur, p=2, dim=1).cpu().numpy())
	linf_pur.extend(torch.norm(diff_pur, p=float('inf'), dim=1).cpu().numpy())

	# accumulate labels/preds (filtered subset only)
	all_labels.extend(y.cpu().numpy())
	all_clean.extend(preds_clean.cpu().numpy())
	all_adv.extend(adv_preds.cpu().numpy())
	all_pur.extend(pur_preds.cpu().numpy())

	# save triplets (filtered subset only)
	if saved < MAX_SAVE:
		for i in range(x_norm.size(0)):
			if saved >= MAX_SAVE: break
			row = torch.cat([clean_pix[i], adv_pix[i], pur_pix[i]], dim=2)
			save_image(row, os.path.join(triplet_dir, f'{saved:05d}_triplet.png'))
			saved += 1

# 結果集計
if total == 0:
	print("No samples were correctly classified by the clean model. Evaluation aborted.")
else:
	clean_acc = correct_clean / total
	adv_acc = correct_adv / total
	pur_acc = correct_pur / total
	l2_adv = np.array(l2_adv); linf_adv = np.array(linf_adv); l2_pur = np.array(l2_pur); linf_pur = np.array(linf_pur)

	print("\n======================================")
	print("Results (ChestXray FGSM + JPEG)")
	print("======================================")
	print(f'Total images (clean-correct only): {total}')
	print(f'Clean accuracy:      {clean_acc:.4f}')
	print(f'Adversarial accuracy: {adv_acc:.4f}')
	print(f'Purified accuracy:   {pur_acc:.4f}')
	print(f'Defense improvement: {(pur_acc - adv_acc):+.4f}')
	print("-" * 40)
	print("Perturbation Norms (Adv vs Clean):")
	print(f'  L2 mean={l2_adv.mean():.4f} std={l2_adv.std():.4f}')
	print(f'  Linf mean={linf_adv.mean():.4f} std={linf_adv.std():.4f}')
	print("Purified (vs Clean):")
	print(f'  L2 mean={l2_pur.mean():.4f} std={l2_pur.std():.4f}')
	print(f'  Linf mean={linf_pur.mean():.4f} std={linf_pur.std():.4f}')

	# 混同行列（テキスト）
	def print_cm(y_true, y_pred, title, labels=('NORMAL', 'PNEUMONIA')):
		cm = confusion_matrix(y_true, y_pred)
		tn, fp, fn, tp = cm.ravel()
		precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
		recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
		f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
		specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
		print(f"\n{title}:")
		print("  Confusion Matrix:")
		print(f"                Predicted")
		print(f"                {labels[0]:6s} {labels[1]:8s}")
		print(f"  Actual {labels[0]:6s}  {tn:5d}  {fp:5d}")
		print(f"         {labels[1]:6s}  {fn:5d}  {tp:5d}")
		print(f"  Precision:   {precision:.4f}")
		print(f"  Recall:      {recall:.4f}")
		print(f"  F1-Score:    {f1:.4f}")
		print(f"  Specificity: {specificity:.4f}")

	print_cm(all_labels, all_clean, 'Clean Images')
	print_cm(all_labels, all_adv, 'Adversarial (FGSM)')
	print_cm(all_labels, all_pur, 'Purified (JPEG) Images')

	# CSV / summary保存
	summary_txt = os.path.join(out_dir, 'summary_statistics.txt')
	df = pd.DataFrame({
		'true_label': all_labels,
		'pred_clean': all_clean,
		'pred_adv': all_adv,
		'pred_purified': all_pur,
		'l2_norm_adv': l2_adv,
		'linf_norm_adv': linf_adv,
		'l2_norm_purified': l2_pur,
		'linf_norm_purified': linf_pur,
	})
	df['attack_success'] = (df['pred_adv'] != df['true_label']).astype(int)
	df['purify_success'] = (df['pred_purified'] == df['true_label']).astype(int)
	df['defense_recovery'] = ((df['attack_success'] == 1) & (df['purify_success'] == 1)).astype(int)
	csv_path = os.path.join(out_dir, 'detailed_results.csv')
	df.to_csv(csv_path, index=False)

	with open(summary_txt, 'w') as f:
		f.write('=' * 70 + '\n')
		f.write('ChestXray - FGSM + JPEG Purification Summary\n')
		f.write('=' * 70 + '\n\n')
		f.write(f'Dataset: ChestXray test set (NORMAL vs PNEUMONIA)\n')
		f.write(f'Attack: FGSM, epsilon={epsilon_pixel:.4f} ({epsilon_pixel * 255:.1f}/255)\n')
		f.write(f'Purification: JPEG quality={JPEG_QUALITY}\n')
		f.write(f'Classifier ckpt: {clf_ckpt}\n')
		f.write(f'Output dir: {out_dir}\n\n')
		f.write('-' * 70 + '\n')
		f.write(f'Total images (clean-correct only): {total}\n')
		f.write(f'Clean Acc:       {clean_acc:.4f}\n')
		f.write(f'Adversarial Acc: {adv_acc:.4f}\n')
		f.write(f'Purified Acc:    {pur_acc:.4f}\n')
		f.write(f'Defense Improvement: {pur_acc - adv_acc:+.4f}\n')
		f.write('-' * 70 + '\n')
		f.write('Perturbation Norms (Adv vs Clean):\n')
		f.write(f'  L2 mean={l2_adv.mean():.6f} std={l2_adv.std():.6f}\n')
		f.write(f'  Linf mean={linf_adv.mean():.6f} std={linf_adv.std():.6f}\n')
		f.write('Purified (vs Clean):\n')
		f.write(f'  L2 mean={l2_pur.mean():.6f} std={l2_pur.std():.6f}\n')
		f.write(f'  Linf mean={linf_pur.mean():.6f} std={linf_pur.std():.6f}\n')
		f.write('-' * 70 + '\n')
		for name, preds in [('Clean', all_clean), ('Adversarial', all_adv), ('Purified', all_pur)]:
			cm = confusion_matrix(all_labels, preds); tn, fp, fn, tp = cm.ravel()
			precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
			recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
			f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
			specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
			f.write(f'\n{name} Images:\n')
			f.write(f'  TN:{tn:4d} FP:{fp:4d} FN:{fn:4d} TP:{tp:4d}\n')
			f.write(f'  Precision:{precision:.4f} Recall:{recall:.4f} F1:{f1:.4f} Specificity:{specificity:.4f}\n')

	print(f"Saved triplets -> {triplet_dir}")
	print(f"Saved stats CSV -> {csv_path}")
	print(f"Saved summary -> {summary_txt}")
	print("Evaluation complete.")

