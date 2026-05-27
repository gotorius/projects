# Adversarial Defense via Diffusion Models on Medical Images

This project investigates **diffusion model-based adversarial purification** for medical image classification. We evaluate multiple defense methods across three medical imaging datasets, using ResNet50 and ViT-B/16 classifiers, against FGSM, PGD-10, and AutoAttack.

---

## Datasets

| Dataset | Task | Classes | Image Size |
|---------|------|---------|------------|
| **ChestXray** | Pneumonia detection | Normal / Pneumonia | 224x224 |
| **DermMel** | Melanoma classification | Melanoma / Not Melanoma | 224x224 |
| **PCam** | Cancer metastasis detection | Positive / Negative | 96x96 |

---

## Attacks

| Method | Type | Description |
|--------|------|-------------|
| **FGSM** | Single-step | Fast Gradient Sign Method (Goodfellow et al., 2015) |
| **PGD-10** | Iterative | Projected Gradient Descent, 10 iterations (Madry et al., 2018) |
| **AutoAttack** | Ensemble | Parameter-free ensemble attack (Croce & Hein, 2020) |

All attacks use L-inf perturbation with epsilon = 8/255.

---

## Defense Methods

| Method | Description |
|--------|-------------|
| **DDPM** | Partial diffusion + reverse denoising (DiffPure) |
| **VAE** | Encode to latent space, then decode |
| **GAN** | Generator-based image purification |
| **JPEG** | JPEG compression to remove high-frequency perturbations |
| **ImageNet** | Diffusion model pre-trained on ImageNet |

---

## Project Structure

```
projects/
├── resnet/               # ResNet50 experiments
│   ├── chestxray/
│   ├── dermmel/
│   └── pcam/
├── vit/                  # ViT-B/16 experiments
│   ├── chestxray/
│   ├── dermmel/
│   ├── pcam/
│   └── classifiers/
└── config/
```

Each dataset folder contains subdirectories per defense method (`ddpm/`, `vae/`, `gan/`, `jpeg/`, `imagenet/`), each with attack-specific evaluation scripts (`fgsm/`, `pgd/`, `autoattack/`).

---

## Results

### ResNet50 — ChestXray (500 samples, eps=8/255)

| Attack | No Defense | DDPM | JPEG (q=11) |
|--------|-----------|------|-------------|
| FGSM | 64.40% | **91.60%** | 64.40% |
| PGD-10 | 0.00% | **90.00%** | 67.40% |
| AutoAttack | 0.00% | **88.60%** | 67.00% |

DDPM params: `start_t=80`, `T_purify=50`

### ResNet50 — DermMel (3,434 samples, eps=8/255)

| Attack | No Defense | DDPM | JPEG (q=11) |
|--------|-----------|------|-------------|
| FGSM | 43.42% | **53.47%** | 49.53% |
| PGD-10 | 0.00% | **64.91%** | 49.56% |
| AutoAttack | 0.00% | **61.97%** | 49.56% |

### ViT-B/16 — ChestXray (500 samples, eps=8/255)

| Attack | No Defense | DDPM | VAE | GAN | ImageNet | JPEG |
|--------|-----------|------|-----|-----|----------|------|
| FGSM | 40.00% | 85.00% | **91.40%** | 88.80% | 56.00% | 51.40% |
| PGD-10 | 0.00% | 88.00% | **94.20%** | 88.80% | 71.00% | - |
| AutoAttack | 0.00% | 88.40% | 93.60% | 90.00% | **92.40%** | - |

DDPM: `start_t=280, T_purify=300` | VAE: `latent_dim=256` | GAN: `rec_iters=150`

### ViT-B/16 — PCam (500 samples, eps=8/255)

| Attack | No Defense | VAE | GAN | ImageNet | JPEG |
|--------|-----------|-----|-----|----------|------|
| FGSM | 40.60% | **76.60%** | 56.00% | 42.80% | 44.40% |
| PGD-10 | 0.80% | **77.60%** | - | - | - |
| AutoAttack | 0.00% | **78.00%** | 58.40% | - | - |

### ViT-B/16 — DermMel (500 samples, eps=8/255)

| Attack | No Defense | DDPM |
|--------|-----------|------|
| FGSM | 82.60% | 78.40% |
| PGD-10 | 16.20% | **77.20%** |
| AutoAttack | 0.00% | **76.80%** |

---

## Key Findings

- Generative model-based defenses (VAE, GAN, DDPM) consistently outperform JPEG compression.
- VAE achieves the highest average defense accuracy on ChestXray (ViT).
- DDPM is effective across all datasets, especially against strong attacks (PGD, AutoAttack).
- Defense effectiveness varies by dataset characteristics and classifier architecture.

---

## References

1. Ho et al. (2020). Denoising Diffusion Probabilistic Models. *NeurIPS*.
2. Nie et al. (2022). Diffusion Models for Adversarial Purification. *ICML*.
3. Goodfellow et al. (2015). Explaining and Harnessing Adversarial Examples. *ICLR*.
4. Madry et al. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks. *ICLR*.
5. Croce & Hein (2020). Reliable Evaluation of Adversarial Robustness. *ICML*.
