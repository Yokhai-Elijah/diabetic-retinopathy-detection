## 🗃️ Dataset Information

| | Training Dataset | External Validation Dataset |
|---|---|---|
| **Name** | Retinal Fundus Images (Nithish, 2024) | Diabetic Retinopathy 224×224 (2019 Data) (Rath, 2020) |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/kssanjaynithish03/retinal-fundus-images) | [Kaggle](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data) |
| **Size** | 4,482 images | 3,662 images |
| **Classes** | Normal (2,874), Diabetes (1,608) | Normal (1,805), Diabetic (1,857) — merged from 5 original severity classes (No_DR / Mild / Moderate / Severe / Proliferative_DR) into binary labels |
| **Format** | JPG fundus photographs, resized to 512×512 for training | JPG fundus photographs, originally 224×224, upsampled to 512×512 for inference |
| **Role** | Model development, 5-fold stratified cross-validation | Held out entirely — never used in training, tuning, or threshold selection |

Neither dataset is redistributed in this repository. Download both from the Kaggle links above and
use `organize_binary.py` to convert them into the binary folder structure expected by the training
and evaluation scripts. Both datasets are third-party Kaggle datasets — check each dataset's Kaggle
page for its specific usage license before redistribution.

---

## 🧪 Methodology

1. **Preprocessing** — fundus region cropping (largest-contour detection) to remove black borders, resize to 512×512, ImageNet normalization.
2. **Training** — EfficientNet-B2 (ImageNet-pretrained) fine-tuned end-to-end via 5-fold stratified cross-validation, using AdamW, weighted cross-entropy (1.15× multiplier on the diabetic class to offset class imbalance), and Albumentations-based augmentation (flips, rotation, color/contrast jitter, blur, noise, coarse dropout, CLAHE).
3. **Model selection** — for each fold, the checkpoint with the best validation AUC is kept; the fold with the strongest internal validation AUC (fold 2) is selected as the primary model.
4. **Statistical validation** — 95% confidence intervals via bootstrap resampling (1,000 iterations) for every reported metric; Expected Calibration Error (ECE) to assess probability calibration.
5. **Robustness check** — test-time augmentation (6 geometric views, softmax-averaged) evaluated for improvement across all metrics.
6. **External validation** — full evaluation on an independent, separately-sourced dataset that was never touched during development, to test true generalization.
7. **Interpretability** — Grad-CAM heatmaps generated on the external validation set to verify the model attends to clinically relevant retinal features (microaneurysms, hemorrhages, exudates).

Full detail on each step is in the accompanying manuscript (submitted to PeerJ Computer Science,
ID CS-2026:07:143944).

---

## 📜 License & Contributing

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full
terms. In short: you're free to use, modify, and redistribute this code, including for commercial
purposes, as long as the original copyright notice is retained.

**Contributions are welcome.** If you'd like to report a bug, suggest an improvement, or extend
this work:
1. Open a [GitHub Issue](../../issues) describing the change or problem.
2. For code changes, fork the repository, make your changes on a new branch, and open a Pull
   Request referencing the issue.
3. For questions about the research itself (methodology, results, the accompanying manuscript),
   feel free to reach out directly — see the manuscript for contact details.
