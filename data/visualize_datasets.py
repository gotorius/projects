"""
3つの医療画像データセットから3×3のグリッド画像を生成するスクリプト
"""

import os
import random
import matplotlib.pyplot as plt
from PIL import Image
import glob

# データセットのパス
datasets = {
    "PCam": "/mnt/data1/Public/MedImages/PCam_ImageFolder/train",
    "ChestXray": "/mnt/data1/Public/MedImages/CellData/chest_xray/train",
    "DermMel": "/mnt/data1/Public/MedImages/DermMel/train_sep"
}

def get_images_from_dataset(dataset_path, num_images=9):
    """データセットからランダムに画像を取得"""
    all_images = []
    
    # サブディレクトリ（クラス）を取得
    for class_dir in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_dir)
        if os.path.isdir(class_path):
            # 画像ファイルを取得
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.bmp']:
                all_images.extend(glob.glob(os.path.join(class_path, ext)))
                all_images.extend(glob.glob(os.path.join(class_path, ext.upper())))
    
    # ランダムに選択
    if len(all_images) >= num_images:
        selected = random.sample(all_images, num_images)
    else:
        selected = all_images[:num_images]
    
    return selected

def create_3x3_grid(images_paths, title, output_path):
    """3×3のグリッド画像を作成して保存（224x224リサイズ、余白なし）"""
    fig, axes = plt.subplots(3, 3, figsize=(6.72, 6.72), dpi=100)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(images_paths):
            img_path = images_paths[idx]
            img = Image.open(img_path)
            # 224x224にリサイズ
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            # グレースケールかRGBかを確認
            if img.mode == 'L':
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
        ax.axis('off')
    
    # 余白なしで設定
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved: {output_path}")

def main():
    random.seed(42)  # 再現性のため
    
    output_dir = "/mnt/data1/gotou/dataset_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    for name, path in datasets.items():
        print(f"\nProcessing {name} dataset...")
        images = get_images_from_dataset(path, num_images=9)
        print(f"  Found {len(images)} images")
        
        if images:
            output_path = os.path.join(output_dir, f"{name}_3x3_grid.png")
            create_3x3_grid(images, f"{name} Dataset - Sample Images", output_path)
        else:
            print(f"  No images found in {path}")

if __name__ == "__main__":
    main()
