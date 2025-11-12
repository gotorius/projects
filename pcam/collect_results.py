"""
PCam - 敵対的防御結果集約スクリプト
FGSM/PGD/AutoAttack攻撃に対するDDPM/JPEG防御の精度をまとめて表示・保存
"""
import os
import pandas as pd
from pathlib import Path

# ========== 結果ディレクトリのパターン ==========
PROJECT_ROOT = '/mnt/data1/gotou/projects/pcam'

# 想定される結果ディレクトリ構造
RESULTS_DIRS = {
    # DDPM防御
    'DDPM+FGSM': [
        f'{PROJECT_ROOT}/ddpm/fgsm/results_*',
        f'{PROJECT_ROOT}/ddpm/fgsm/purify_examples',
    ],
    'DDPM+PGD': [
        f'{PROJECT_ROOT}/ddpm/pgd/results_*',
        f'{PROJECT_ROOT}/ddpm/pgd/purify_examples',
    ],
    'DDPM+AutoAttack': [
        f'{PROJECT_ROOT}/ddpm/autoattack/results_*',
        f'{PROJECT_ROOT}/ddpm/autoattack/purify_examples',
    ],
    # JPEG防御
    'JPEG+FGSM': [
        f'{PROJECT_ROOT}/jpeg/fgsm/results_*',
    ],
    'JPEG+PGD': [
        f'{PROJECT_ROOT}/jpeg/pgd/results_*',
    ],
    'JPEG+AutoAttack': [
        f'{PROJECT_ROOT}/jpeg/autoattack/results_*',
    ],
}

def find_summary_file(search_paths):
    """summary_statistics.txt を検索"""
    for pattern in search_paths:
        base_dir = Path(pattern.split('*')[0]).parent
        if not base_dir.exists():
            continue
        for summary_path in base_dir.rglob('summary_statistics.txt'):
            return summary_path
    return None

def parse_summary_file(summary_path):
    """summary_statistics.txt から精度情報を抽出"""
    if not summary_path or not summary_path.exists():
        return None
    
    with open(summary_path, 'r') as f:
        content = f.read()
    
    result = {
        'total': 0,
        'clean_acc': 0.0,
        'adv_acc': 0.0,
        'purified_acc': 0.0,
        'defense_improvement': 0.0,
    }
    
    # パース
    for line in content.split('\n'):
        line = line.strip()
        if 'Total images' in line and ':' in line:
            try:
                result['total'] = int(line.split(':')[1].strip())
            except:
                pass
        elif 'Clean Acc:' in line:
            try:
                result['clean_acc'] = float(line.split(':')[1].strip())
            except:
                pass
        elif 'Adversarial Acc:' in line or 'Adversarial Acc' in line:
            try:
                result['adv_acc'] = float(line.split(':')[1].strip())
            except:
                pass
        elif 'Purified Acc:' in line:
            try:
                result['purified_acc'] = float(line.split(':')[1].strip())
            except:
                pass
        elif 'Defense Improvement:' in line:
            try:
                val_str = line.split(':')[1].strip()
                result['defense_improvement'] = float(val_str.replace('+', ''))
            except:
                pass
    
    return result

def main():
    print("=" * 80)
    print("PCam - 敵対的防御結果サマリー (FGSM/PGD/AutoAttack)")
    print("=" * 80)
    print()
    
    # 結果を収集
    summary_data = []
    
    for method_name, search_paths in RESULTS_DIRS.items():
        summary_path = find_summary_file(search_paths)
        
        if summary_path:
            print(f"✓ {method_name:20s} -> {summary_path}")
            result = parse_summary_file(summary_path)
            if result:
                result['method'] = method_name
                result['summary_path'] = str(summary_path)
                summary_data.append(result)
        else:
            print(f"✗ {method_name:20s} -> 結果ファイルが見つかりません")
    
    print()
    print("=" * 80)
    
    if not summary_data:
        print("結果ファイルが1つも見つかりませんでした。")
        print("各評価スクリプトを実行してから再度実行してください。")
        print()
        print("実行例:")
        print("  python /mnt/data1/gotou/projects/pcam/ddpm/fgsm/ddpm_fgsm.py")
        print("  python /mnt/data1/gotou/projects/pcam/ddpm/pgd/ddpm_pgd.py")
        print("  python /mnt/data1/gotou/projects/pcam/ddpm/autoattack/ddpm_auto.py")
        print("  python /mnt/data1/gotou/projects/pcam/jpeg/fgsm/jpeg_fgsm.py")
        print("  python /mnt/data1/gotou/projects/pcam/jpeg/pgd/jpeg_pgd.py")
        print("  python /mnt/data1/gotou/projects/pcam/jpeg/autoattack/jpeg_auto.py")
        return
    
    # DataFrame作成
    df = pd.DataFrame(summary_data)
    df = df[['method', 'total', 'clean_acc', 'adv_acc', 'purified_acc', 'defense_improvement']]
    
    # 表示
    print("\n集約結果:")
    print("-" * 80)
    print(f"{'Method':<20s} {'Total':>6s} {'Clean':>7s} {'Adv':>7s} {'Purified':>9s} {'Improve':>8s}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        print(f"{row['method']:<20s} "
              f"{int(row['total']):6d} "
              f"{row['clean_acc']:7.4f} "
              f"{row['adv_acc']:7.4f} "
              f"{row['purified_acc']:9.4f} "
              f"{row['defense_improvement']:+8.4f}")
    
    print("-" * 80)
    
    # 攻撃タイプ別の平均を計算
    print("\n攻撃タイプ別の防御効果:")
    print("-" * 80)
    
    for attack in ['FGSM', 'PGD', 'AutoAttack']:
        attack_df = df[df['method'].str.contains(attack)]
        if len(attack_df) > 0:
            avg_improvement = attack_df['defense_improvement'].mean()
            avg_purified = attack_df['purified_acc'].mean()
            avg_adv = attack_df['adv_acc'].mean()
            print(f"{attack:15s}: "
                  f"Avg Adv={avg_adv:.4f}, "
                  f"Avg Purified={avg_purified:.4f}, "
                  f"Avg Improvement={avg_improvement:+.4f}")
    
    # 防御手法別の平均
    print("\n防御手法別の効果:")
    print("-" * 80)
    
    for defense in ['DDPM', 'JPEG']:
        defense_df = df[df['method'].str.contains(defense)]
        if len(defense_df) > 0:
            avg_improvement = defense_df['defense_improvement'].mean()
            avg_purified = defense_df['purified_acc'].mean()
            avg_adv = defense_df['adv_acc'].mean()
            print(f"{defense:15s}: "
                  f"Avg Adv={avg_adv:.4f}, "
                  f"Avg Purified={avg_purified:.4f}, "
                  f"Avg Improvement={avg_improvement:+.4f}")
    
    print()
    
    # CSVとして保存
    output_csv = os.path.join(PROJECT_ROOT, 'pcam_defense_summary.csv')
    df.to_csv(output_csv, index=False)
    print(f"結果をCSVに保存しました: {output_csv}")
    
    # テキストサマリーも保存
    output_txt = os.path.join(PROJECT_ROOT, 'pcam_defense_summary.txt')
    with open(output_txt, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PCam - 敵対的防御結果サマリー (FGSM/PGD/AutoAttack)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("集約結果:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Method':<20s} {'Total':>6s} {'Clean':>7s} {'Adv':>7s} {'Purified':>9s} {'Improve':>8s}\n")
        f.write("-" * 80 + "\n")
        
        for _, row in df.iterrows():
            f.write(f"{row['method']:<20s} "
                    f"{int(row['total']):6d} "
                    f"{row['clean_acc']:7.4f} "
                    f"{row['adv_acc']:7.4f} "
                    f"{row['purified_acc']:9.4f} "
                    f"{row['defense_improvement']:+8.4f}\n")
        
        f.write("-" * 80 + "\n\n")
        
        # 攻撃タイプ別
        f.write("攻撃タイプ別の防御効果:\n")
        f.write("-" * 80 + "\n")
        for attack in ['FGSM', 'PGD', 'AutoAttack']:
            attack_df = df[df['method'].str.contains(attack)]
            if len(attack_df) > 0:
                avg_improvement = attack_df['defense_improvement'].mean()
                avg_purified = attack_df['purified_acc'].mean()
                avg_adv = attack_df['adv_acc'].mean()
                f.write(f"{attack:15s}: "
                        f"Avg Adv={avg_adv:.4f}, "
                        f"Avg Purified={avg_purified:.4f}, "
                        f"Avg Improvement={avg_improvement:+.4f}\n")
        
        # 防御手法別
        f.write("\n防御手法別の効果:\n")
        f.write("-" * 80 + "\n")
        for defense in ['DDPM', 'JPEG']:
            defense_df = df[df['method'].str.contains(defense)]
            if len(defense_df) > 0:
                avg_improvement = defense_df['defense_improvement'].mean()
                avg_purified = defense_df['purified_acc'].mean()
                avg_adv = defense_df['adv_acc'].mean()
                f.write(f"{defense:15s}: "
                        f"Avg Adv={avg_adv:.4f}, "
                        f"Avg Purified={avg_purified:.4f}, "
                        f"Avg Improvement={avg_improvement:+.4f}\n")
    
    print(f"サマリーをテキストに保存しました: {output_txt}")
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
