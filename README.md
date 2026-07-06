# Diabetic Retinopathy Detection using EfficientNet-B2

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/RickSorkin/diabetic-retinopathy-detection)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Automated detection of diabetic retinopathy from retinal fundus photographs using **EfficientNet-B2** with transfer learning, 5-fold cross-validation, and Grad-CAM interpretability.

---

## 🔬 Live Demo

Try the model instantly — no setup required:

👉 **[huggingface.co/spaces/RickSorkin/diabetic-retinopathy-detection](https://huggingface.co/spaces/RickSorkin/diabetic-retinopathy-detection)**

Upload a retinal fundus image and get:
- Binary prediction (Normal / Diabetic Retinopathy)
- Confidence score
- Grad-CAM heatmap showing which retinal regions influenced the decision

---

## 📊 Key Results

| Metric | Without TTA | With TTA |
|--------|------------|----------|
| ROC-AUC | 98.11% | 98.46% |
| Accuracy | 95.93% | 96.12% |
| Sensitivity | 94.13% | 94.40% |
| Specificity | 97.78% | 97.89% |
| F1 Score | 95.94% | 96.08% |
| MCC | 0.9193 | 0.9231 |

*Evaluated on 3,662 independent external validation images (Rath, 2020). All folds mean accuracy: 94.08% ± 0.98%.*

---

## 🗂️ Repository Structure

```
diabetic-retinopathy-detection/
│
├── app.py                        # Hugging Face Spaces Gradio demo
├── requirements.txt              # Python dependencies
│
├── organize_binary.py            # Convert 5-class DR dataset → binary (No_DR / DR)
├── compare_models.py             # Evaluate and rank all .pth models on a dataset
├── statistical_analysis.py       # Bootstrap CIs, ROC-AUC, PR-AUC, calibration curves
├── test_time_augmentation.py     # TTA evaluation with MCC bootstrap CI
├── calculate_mcc.py              # Standalone MCC + bootstrap CI calculator
├── gradcam_paper_figures.py      # Generate publication-quality Grad-CAM figures
└── README.md
```

> **Note:** The trained model weights (`fold_2_best.pth` and all fold models) are not included due to GitHub file size limits. The model is loaded automatically in the Hugging Face demo. To train from scratch, follow the Training section below.

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Yokhai-Elijah/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare your dataset

**Training dataset:** [Retinal Fundus Images (Nithish, 2024)](https://www.kaggle.com/datasets/kssanjaynithish03/retinal-fundus-images)
```
Normal and diabetes/
├── Normal (N)/
└── Diabetes (D)/
```

**External validation dataset:** [Diabetic Retinopathy 224×224 (Rath, 2020)](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data)

Convert to binary format:
```bash
python organize_binary.py
```
This creates `binary_dataset/No_DR/` and `binary_dataset/DR/`.

---

## 🔧 Scripts

### Evaluate all models
Compare every `.pth` file in your directory against a dataset:
```bash
python compare_models.py --image_dir binary_dataset
```

### Statistical analysis (AUC, CI, calibration)
```bash
python statistical_analysis.py \
  --model_path training_output_kfold/fold_2_best.pth \
  --data_dir binary_dataset \
  --output_dir statistical_results
```

### Test-Time Augmentation + MCC
```bash
python test_time_augmentation.py \
  --model_path training_output_kfold/fold_2_best.pth \
  --data_dir binary_dataset \
  --output_dir tta_results
```

### MCC with bootstrap CI
```bash
python calculate_mcc.py \
  --model_path training_output_kfold/fold_2_best.pth \
  --data_dir binary_dataset \
  --threshold 0.5409
```

### Grad-CAM publication figures
```bash
python gradcam_paper_figures.py \
  --model_path training_output_kfold/fold_2_best.pth \
  --image_dir binary_dataset \
  --output_dir gradcam_paper \
  --threshold 0.5409
```

---

## 🏋️ Model Details

| Parameter | Value |
|-----------|-------|
| Architecture | EfficientNet-B2 (timm) |
| Training images | 4,482 fundus images |
| Cross-validation | 5-fold stratified |
| Optimizer | AdamW (lr=5×10⁻⁵, wd=1×10⁻⁴) |
| Loss function | Weighted cross-entropy (1.15× diabetic) |
| Batch size | 6 |
| Max epochs | 30 (early stopping, patience=7) |
| Image size | 512×512 |
| Mixed precision | Yes (torch.cuda.amp) |
| Classification threshold | 0.5409 (Youden's Index) |

---

## 📦 Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
albumentations
opencv-python
Pillow
scikit-learn
matplotlib
seaborn
gradio>=4.0.0
numpy
pandas
tqdm
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📄 Citation

If you use this work, please cite:

```
@article{elijah2024diabetic,
  title={Automated Diabetic Retinopathy Detection Using EfficientNet-B2 
         with Transfer Learning: A Comprehensive Statistical Validation Study},
  author={Elijah, Yokhai},
  year={2024}
}
```

---

## 📚 References

- Nithish, K.S.S. (2024). *Retinal Fundus Images* [Dataset]. Kaggle.
- Rath, S. (2020). *Diabetic Retinopathy 224×224 (2019 Data)* [Dataset]. Kaggle.
- Tan, M. & Le, Q. (2019). EfficientNet: Rethinking model scaling for CNNs. ICML.
- Selvaraju, R.R. et al. (2017). Grad-CAM. ICCV.

---

## ⚕️ Disclaimer

This project is for **research purposes only** and is not a medical device. It should not be used as a substitute for professional ophthalmological examination. Do not upload real patient images containing identifiable health information.
