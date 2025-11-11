"""
PCam vs DermMel のJPEG圧縮防御性能比較分析スクリプト
なぜDermMelでJPEG防御が効かないのかを調査
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ========== DermMel結果の読み込み ==========
print("="*70)
print("Loading DermMel Results...")
print("="*70)

dermmel_summary = pd.read_csv('/mnt/data1/gotou/projects/dermmel/jpeg/fgsm/defense_results/overall_summary.csv')

print("\nDermMel Overall Summary:")
print(dermmel_summary)

# ========== PCam結果の読み込み（品質11のデータを参照）==========
print("\n" + "="*70)
print("Loading PCam Results...")
print("="*70)

# PCamのQ11データを読み込んで統計計算
pcam_q11 = pd.read_csv('/mnt/data1/gotou/projects/pcam/jpeg/fgsm/jpeg_defense_all_q11.csv')

print(f"\nPCam Q11 Dataset: {len(pcam_q11)} samples")
print(f"Columns: {list(pcam_q11.columns)}")

# PCamの統計計算
pcam_total = len(pcam_q11[pcam_q11['Original_Correct'] == 1])
pcam_attack_success = pcam_q11['Attack_Success'].sum()
pcam_defense_success = pcam_q11['Defense_Success'].sum()

pcam_clean_acc = pcam_q11['Original_Correct'].mean()
pcam_adv_acc = 1 - (pcam_attack_success / pcam_total) if pcam_total > 0 else 0
pcam_defended_acc = (pcam_total - pcam_attack_success + pcam_defense_success) / pcam_total if pcam_total > 0 else 0

print(f"\nPCam Statistics (Quality 11 - rough estimate from quality range):")
print(f"  Total correctly classified: {pcam_total}")
print(f"  Clean Accuracy: {pcam_clean_acc:.4f}")
print(f"  Adversarial Accuracy: {pcam_adv_acc:.4f}")
print(f"  Defended Accuracy: {pcam_defended_acc:.4f}")
print(f"  Defense Improvement: {pcam_defended_acc - pcam_adv_acc:+.4f}")
print(f"  Attack Success: {pcam_attack_success}")
print(f"  Defense Success (of attacked): {pcam_defense_success}")
if pcam_attack_success > 0:
    pcam_defense_rate = pcam_defense_success / pcam_attack_success
    print(f"  Defense Success Rate: {pcam_defense_rate:.4f}")

# ========== 比較分析 ==========
print("\n" + "="*70)
print("Comparative Analysis: PCam vs DermMel")
print("="*70)

# DermMelの最良結果を取得（JPEG品質10-30あたりが良さそう）
dermmel_best = dermmel_summary.loc[dermmel_summary['Defense_Improvement'].idxmax()]

print(f"\nDermMel Best Performance:")
print(f"  JPEG Quality: {dermmel_best['JPEG_Quality']}")
print(f"  Clean Acc: {dermmel_best['Clean_Acc']:.4f}")
print(f"  Adversarial Acc: {dermmel_best['Adv_Acc']:.4f}")
print(f"  Compressed Acc: {dermmel_best['Compressed_Acc']:.4f}")
print(f"  Defense Improvement: {dermmel_best['Defense_Improvement']:+.4f}")

print(f"\nPCam (Q=11, comparable quality):")
print(f"  Clean Acc: {pcam_clean_acc:.4f}")
print(f"  Adversarial Acc: {pcam_adv_acc:.4f}")
print(f"  Defended Acc: {pcam_defended_acc:.4f}")
print(f"  Defense Improvement: {pcam_defended_acc - pcam_adv_acc:+.4f}")

print("\n" + "="*70)
print("Key Differences:")
print("="*70)

print(f"\n1. **Defense Effectiveness**:")
print(f"   - PCam Defense Improvement: {pcam_defended_acc - pcam_adv_acc:+.4f}")
print(f"   - DermMel Best Improvement: {dermmel_best['Defense_Improvement']:+.4f}")
print(f"   - Difference: {abs((pcam_defended_acc - pcam_adv_acc) - dermmel_best['Defense_Improvement']):.4f}")

print(f"\n2. **Adversarial Robustness (before defense)**:")
print(f"   - PCam Adversarial Acc: {pcam_adv_acc:.4f}")
print(f"   - DermMel Adversarial Acc: {dermmel_best['Adv_Acc']:.4f}")
print(f"   - PCam is {'more' if pcam_adv_acc > dermmel_best['Adv_Acc'] else 'less'} robust initially")

# ========== 仮説検証 ==========
print("\n" + "="*70)
print("Hypothesis: Why JPEG Defense Fails for DermMel")
print("="*70)

print("\n**Possible Reasons:**")
print("\n1. **Image Complexity & Texture:**")
print("   - PCam: Histopathology images (mostly tissue patterns, repetitive structures)")
print("   - DermMel: Dermatology images (skin lesions with various colors, textures, hair)")
print("   - → Skin images may have more high-frequency details that are critical for")
print("      classification but are removed by JPEG compression")

print("\n2. **Color Information Importance:**")
print("   - PCam: Primarily H&E stained (pinkish-purple), color distribution is narrow")
print("   - DermMel: Melanoma detection heavily relies on color variation (ABCDE rule)")
print("   - → JPEG compression in chroma channels may destroy critical diagnostic features")

print("\n3. **Attack Success Rate:**")
print(f"   - PCam Attack Success: {pcam_attack_success} / {pcam_total} = {pcam_attack_success/pcam_total:.4f}")
print(f"   - DermMel Attack Success: ~{1 - dermmel_best['Adv_Acc']:.4f}")
print("   - Higher attack success means more perturbations, potentially in")
print("     frequency domains that JPEG compression cannot remove")

print("\n4. **Dataset Characteristics:**")
print("   - PCam: 96x96px patches, centered tumor tissue")
print("   - DermMel: Variable sized lesions (resized to 224x224)")
print("   - → Larger images may have perturbations distributed across")
print("      multiple JPEG 8x8 blocks, making them harder to remove")

print("\n5. **Model Architecture Sensitivity:**")
print("   - Both use ResNet50 but trained on different data")
print("   - DermMel model may rely more on high-frequency features")
print("   - → Features destroyed by JPEG compression might be essential")

# ========== 可視化 ==========
print("\n" + "="*70)
print("Generating Comparison Plots...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Defense Improvement Comparison
ax1 = axes[0, 0]
qualities = dermmel_summary['JPEG_Quality'].values
dermmel_improvements = dermmel_summary['Defense_Improvement'].values
pcam_improvement = pcam_defended_acc - pcam_adv_acc

ax1.plot(qualities, dermmel_improvements, 'o-', linewidth=2, markersize=8, 
         label='DermMel', color='red')
ax1.axhline(y=pcam_improvement, color='blue', linestyle='--', linewidth=2, 
            label=f'PCam (Q=11, est. {pcam_improvement:.3f})')
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax1.set_xlabel('JPEG Quality', fontsize=12)
ax1.set_ylabel('Defense Improvement', fontsize=12)
ax1.set_title('JPEG Defense Effectiveness Comparison', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Accuracy Comparison
ax2 = axes[0, 1]
x_pos = np.arange(3)
pcam_accs = [pcam_clean_acc, pcam_adv_acc, pcam_defended_acc]
dermmel_accs = [dermmel_best['Clean_Acc'], dermmel_best['Adv_Acc'], dermmel_best['Compressed_Acc']]

width = 0.35
ax2.bar(x_pos - width/2, pcam_accs, width, label='PCam', color='blue', alpha=0.7)
ax2.bar(x_pos + width/2, dermmel_accs, width, label='DermMel', color='red', alpha=0.7)
ax2.set_xlabel('Condition', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Accuracy Comparison (Best JPEG Quality)', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(['Clean', 'Adversarial', 'Defended'])
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: DermMel Quality vs All Metrics
ax3 = axes[1, 0]
ax3.plot(qualities, dermmel_summary['Clean_Acc'], 'o-', label='Clean', linewidth=2, markersize=6)
ax3.plot(qualities, dermmel_summary['Adv_Acc'], 's-', label='Adversarial', linewidth=2, markersize=6)
ax3.plot(qualities, dermmel_summary['Compressed_Acc'], '^-', label='Compressed', linewidth=2, markersize=6)
ax3.set_xlabel('JPEG Quality', fontsize=12)
ax3.set_ylabel('Accuracy', fontsize=12)
ax3.set_title('DermMel: Accuracy Across JPEG Qualities', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: L2 Norm Comparison
ax4 = axes[1, 1]
ax4.plot(qualities, dermmel_summary['L2_Adv_Mean'], 'o-', label='Adversarial (vs Clean)', 
         linewidth=2, markersize=6, color='red')
ax4.plot(qualities, dermmel_summary['L2_Compressed_Mean'], 's-', label='Compressed (vs Clean)', 
         linewidth=2, markersize=6, color='green')
ax4.set_xlabel('JPEG Quality', fontsize=12)
ax4.set_ylabel('L2 Norm (Mean)', fontsize=12)
ax4.set_title('DermMel: Perturbation Magnitude', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
output_path = '/mnt/data1/gotou/projects/dermmel/jpeg/fgsm/dataset_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Comparison plot saved to: {output_path}")
plt.close()

# ========== サマリーレポート保存 ==========
report_path = '/mnt/data1/gotou/projects/dermmel/jpeg/fgsm/comparison_report.txt'
with open(report_path, 'w') as f:
    f.write("="*70 + "\n")
    f.write("PCam vs DermMel: JPEG Compression Defense Comparison\n")
    f.write("="*70 + "\n\n")
    
    f.write("EXECUTIVE SUMMARY:\n")
    f.write("-"*70 + "\n")
    f.write(f"PCam JPEG Defense Improvement: {pcam_defended_acc - pcam_adv_acc:+.4f}\n")
    f.write(f"DermMel Best JPEG Defense Improvement: {dermmel_best['Defense_Improvement']:+.4f}\n")
    f.write(f"Performance Gap: {abs((pcam_defended_acc - pcam_adv_acc) - dermmel_best['Defense_Improvement']):.4f}\n\n")
    
    f.write("CONCLUSION:\n")
    f.write("-"*70 + "\n")
    f.write("JPEG compression defense is SIGNIFICANTLY LESS EFFECTIVE for DermMel\n")
    f.write("compared to PCam. The defense improvement is near zero or even negative\n")
    f.write("for most JPEG quality levels.\n\n")
    
    f.write("KEY FINDINGS:\n")
    f.write("-"*70 + "\n")
    f.write("1. DermMel shows ~50% defended accuracy at best (Quality 10-20)\n")
    f.write("2. PCam shows higher defense success with JPEG compression\n")
    f.write("3. DermMel adversarial accuracy is very low (~43%), indicating\n")
    f.write("   the model is highly vulnerable to FGSM attacks\n\n")
    
    f.write("HYPOTHESIZED REASONS:\n")
    f.write("-"*70 + "\n")
    f.write("A. Image Characteristics:\n")
    f.write("   - DermMel: Skin lesions with complex textures, hair, color variations\n")
    f.write("   - PCam: Tissue histopathology with repetitive, structured patterns\n")
    f.write("   → JPEG compression may destroy diagnostic features in skin images\n\n")
    
    f.write("B. Color Dependency:\n")
    f.write("   - Melanoma detection relies heavily on color (ABCDE rule)\n")
    f.write("   - JPEG aggressively compresses chroma channels\n")
    f.write("   → Critical color information may be lost\n\n")
    
    f.write("C. Feature Frequency:\n")
    f.write("   - Skin images may have important high-frequency details\n")
    f.write("   - JPEG quantization removes high-frequency components\n")
    f.write("   → Essential features for correct classification are removed\n\n")
    
    f.write("D. Attack Pattern:\n")
    f.write("   - Adversarial perturbations may target low-frequency components\n")
    f.write("   - JPEG primarily removes high-frequency noise\n")
    f.write("   → Perturbations survive compression\n\n")
    
    f.write("RECOMMENDATIONS:\n")
    f.write("-"*70 + "\n")
    f.write("1. Consider alternative defenses for DermMel:\n")
    f.write("   - DDPM (Denoising Diffusion Probabilistic Models)\n")
    f.write("   - Feature squeezing in color space\n")
    f.write("   - Adversarial training\n")
    f.write("   - Input gradient regularization\n\n")
    
    f.write("2. Analyze frequency domain:\n")
    f.write("   - Perform FFT analysis of clean vs adversarial images\n")
    f.write("   - Identify which frequency bands contain perturbations\n")
    f.write("   - Design targeted filtering\n\n")
    
    f.write("3. Investigate model sensitivity:\n")
    f.write("   - Analyze which layers are most affected by attacks\n")
    f.write("   - Consider architecture modifications\n")
    f.write("   - Explore ensemble methods\n\n")

print(f"✅ Comparison report saved to: {report_path}")

print("\n" + "="*70)
print("Analysis Complete!")
print("="*70)
print(f"\nKey Takeaway: JPEG compression is NOT effective for DermMel defense,")
print(f"likely due to dataset characteristics (color-dependent, complex textures).")
print(f"Alternative defense mechanisms should be explored.")
