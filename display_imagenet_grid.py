"""
ImageNet画像を4x4グリッドで表示・保存するスクリプト
スライド用の見栄えの良い画像を生成

使用方法:
1. デフォルト（GitHubからImageNetサンプル画像をダウンロード）:
   python display_imagenet_grid.py

2. ImageNetデータセットがある場合:
   python display_imagenet_grid.py --imagenet_dir /path/to/imagenet/val

3. ラベル付きで表示:
   python display_imagenet_grid.py --show_labels
"""

import os
import argparse
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torchvision.models as models
from torchvision import datasets
import urllib.request
from io import BytesIO
import ssl


def parse_args():
    parser = argparse.ArgumentParser(description='Display ImageNet images in 4x4 grid')
    parser.add_argument('--imagenet_dir', type=str, default=None,
                        help='Path to ImageNet validation directory')
    parser.add_argument('--output', type=str, default='imagenet_grid_4x4.png',
                        help='Output filename')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Size of each image')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--title', type=str, default='ImageNet Sample Images',
                        help='Title for the figure')
    parser.add_argument('--show_labels', action='store_true',
                        help='Show class labels on images')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for output image')
    return parser.parse_args()


# ImageNetサンプル画像のURL（GitHub EliSchwartz/imagenet-sample-imagesより）
# 様々なカテゴリの画像を選択
SAMPLE_IMAGES = [
    # 動物
    ("n02124075_Egyptian_cat", "Egyptian Cat"),
    ("n02106662_German_shepherd", "German Shepherd"),
    ("n02129165_lion", "Lion"),
    ("n01443537_goldfish", "Goldfish"),
    ("n02504458_African_elephant", "African Elephant"),
    ("n01558993_robin", "Robin"),
    ("n02123045_tabby", "Tabby Cat"),
    ("n02129604_tiger", "Tiger"),
    # 乗り物・物体
    ("n02701002_ambulance", "Ambulance"),
    ("n04285008_sports_car", "Sports Car"),
    ("n02690373_airliner", "Airliner"),
    ("n04147183_schooner", "Schooner"),
    # 食べ物・植物
    ("n07747607_orange", "Orange"),
    ("n07753592_banana", "Banana"),
    ("n11939491_daisy", "Daisy"),
    ("n07720875_bell_pepper", "Bell Pepper"),
    # 追加（予備）
    ("n02099601_golden_retriever", "Golden Retriever"),
    ("n02102040_English_springer", "English Springer"),
    ("n01882714_koala", "Koala"),
    ("n02391049_zebra", "Zebra"),
]

BASE_URL = "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"


def download_imagenet_samples(num_images=16, image_size=224):
    """
    GitHubからImageNetのサンプル画像をダウンロード
    """
    # SSL証明書の検証を無効化（一部環境で必要）
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    
    images = []
    labels = []
    
    print("Downloading ImageNet sample images from GitHub...")
    for filename, label in SAMPLE_IMAGES:
        if len(images) >= num_images:
            break
        url = f"{BASE_URL}{filename}.JPEG"
        try:
            print(f"  Downloading {label}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ssl_context, timeout=10) as response:
                img_data = response.read()
                img = Image.open(BytesIO(img_data)).convert('RGB')
                img_tensor = transform(img)
                images.append(img_tensor)
                labels.append(label)
        except Exception as e:
            print(f"  Failed to download {label}: {e}")
    
    if len(images) == 0:
        return None, None
    
    # 必要な枚数に足りない場合でもそのまま返す
    return torch.stack(images), labels


def download_from_pytorch_hub(num_images=16, image_size=224):
    """
    PyTorchのサンプル画像URLからダウンロード（バックアップ）
    """
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    # 様々なソースからの画像URL
    sample_urls = [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg", "Cat"),
        ("https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg", "Dog"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/7/73/Lion_waiting_in_Namibia.jpg/1200px-Lion_waiting_in_Namibia.jpg", "Lion"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/Siegaue%2C_Pair_of_Mute_Swans.jpg/1200px-Siegaue%2C_Pair_of_Mute_Swans.jpg", "Swan"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Sunflower_from_Silesia.jpg/800px-Sunflower_from_Silesia.jpg", "Sunflower"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Red_Apple.jpg/800px-Red_Apple.jpg", "Apple"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Banana-Single.jpg/800px-Banana-Single.jpg", "Banana"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Big_Orange.jpg/800px-Big_Orange.jpg", "Orange"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Collage_of_Nine_Dogs.jpg/1200px-Collage_of_Nine_Dogs.jpg", "Dogs"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png/1200px-Image_created_with_a_mobile_phone.png", "Bird"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg", "Cat 2"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Dog_Breeds.jpg/1200px-Dog_Breeds.jpg", "Dog 2"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/1200px-Camponotus_flavomarginatus_ant.jpg", "Ant"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Phoenicopterus_ruber_in_S%C3%A3o_Paulo_Zoo.jpg/800px-Phoenicopterus_ruber_in_S%C3%A3o_Paulo_Zoo.jpg", "Flamingo"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/A_small_cup_of_coffee.JPG/800px-A_small_cup_of_coffee.JPG", "Coffee"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Good_Food_Display_-_NCI_Visuals_Online.jpg/800px-Good_Food_Display_-_NCI_Visuals_Online.jpg", "Food"),
    ]
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    
    images = []
    labels = []
    
    print("Downloading sample images from Wikipedia...")
    for url, label in sample_urls[:num_images]:
        try:
            print(f"  Downloading {label}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ssl_context, timeout=10) as response:
                img_data = response.read()
                img = Image.open(BytesIO(img_data)).convert('RGB')
                img_tensor = transform(img)
                images.append(img_tensor)
                labels.append(label)
        except Exception as e:
            print(f"  Failed to download {label}: {e}")
    
    if len(images) == 0:
        return None, None
    
    return torch.stack(images), labels


def load_imagenet_images(imagenet_dir, num_images=16, image_size=224, seed=42):
    """
    ImageNetディレクトリから画像を読み込む
    """
    random.seed(seed)
    np.random.seed(seed)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    
    imagenet_path = Path(imagenet_dir)
    
    # クラスフォルダを取得
    class_dirs = sorted([d for d in imagenet_path.iterdir() if d.is_dir()])
    
    if len(class_dirs) == 0:
        print(f"No class directories found in {imagenet_dir}")
        return None, None
    
    # ランダムにクラスを選択
    selected_classes = random.sample(class_dirs, min(num_images, len(class_dirs)))
    
    images = []
    labels = []
    
    for class_dir in selected_classes:
        # クラス内の画像を取得
        image_files = list(class_dir.glob('*.JPEG')) + list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        
        if len(image_files) > 0:
            # ランダムに1枚選択
            img_path = random.choice(image_files)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                images.append(img_tensor)
                labels.append(class_dir.name)
            except Exception as e:
                print(f"Failed to load {img_path}: {e}")
    
    # 足りない場合は追加で読み込み
    while len(images) < num_images:
        class_dir = random.choice(class_dirs)
        image_files = list(class_dir.glob('*.JPEG')) + list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        if len(image_files) > 0:
            img_path = random.choice(image_files)
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                images.append(img_tensor)
                labels.append(class_dir.name)
            except:
                pass
    
    return torch.stack(images[:num_images]), labels[:num_images]


def create_grid_figure(images, labels=None, title='ImageNet Sample Images', show_labels=False, figsize=(12, 12)):
    """
    4x4グリッドの図を作成
    """
    fig, axes = plt.subplots(4, 4, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            if isinstance(images[idx], torch.Tensor):
                # Tensor → numpy
                img = images[idx].permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
            else:
                img = np.array(images[idx]) / 255.0
            
            ax.imshow(img)
            
            if show_labels and labels is not None and idx < len(labels):
                ax.set_title(labels[idx][:15], fontsize=8)
        
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def create_grid_with_torchvision(images, output_path, padding=2):
    """
    torchvisionのmake_gridを使用してグリッド画像を作成
    """
    if isinstance(images, list):
        # PIL画像のリストをテンソルに変換
        transform = transforms.ToTensor()
        images = torch.stack([transform(img) for img in images])
    
    grid = make_grid(images, nrow=4, padding=padding, normalize=False)
    
    # 保存
    grid_np = grid.permute(1, 2, 0).numpy()
    grid_np = np.clip(grid_np * 255, 0, 255).astype(np.uint8)
    Image.fromarray(grid_np).save(output_path)
    print(f"Grid image saved to {output_path}")


def main():
    args = parse_args()
    
    # シード設定
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    images = None
    labels = None
    
    # 画像の読み込み
    if args.imagenet_dir and os.path.exists(args.imagenet_dir):
        print(f"Loading images from {args.imagenet_dir}...")
        images, labels = load_imagenet_images(
            args.imagenet_dir, 
            num_images=16, 
            image_size=args.image_size,
            seed=args.seed
        )
    
    # ImageNetディレクトリがない場合、GitHubからダウンロード
    if images is None:
        print("Trying to download ImageNet samples from GitHub...")
        images, labels = download_imagenet_samples(16, args.image_size)
    
    if images is None or len(images) == 0:
        print("ERROR: Could not download any images. Please check your internet connection.")
        return
    
    print(f"\nSuccessfully loaded {len(images)} images!")
    
    # グリッド画像の作成と保存
    print(f"Creating 4x4 grid...")
    
    # 方法1: matplotlibで作成（ラベル付きオプション）
    fig = create_grid_figure(
        images, 
        labels=labels,
        title=args.title,
        show_labels=args.show_labels,
        figsize=(12, 12)
    )
    
    # matplotlibで保存
    output_matplotlib = args.output
    fig.savefig(output_matplotlib, dpi=args.dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Grid saved to {output_matplotlib}")
    plt.close()
    
    # 方法2: torchvisionで作成（シンプル版、ラベルなし）
    output_simple = args.output.replace('.png', '_simple.png')
    create_grid_with_torchvision(images, output_simple, padding=4)
    
    print("\nDone!")
    print(f"Output files:")
    print(f"  - {output_matplotlib} (with title)")
    print(f"  - {output_simple} (simple grid)")


if __name__ == '__main__':
    main()
