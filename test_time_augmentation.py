"""
test_time_augmentation.py
--------------------------
Applies test-time augmentation (TTA) to improve model predictions
by averaging predictions over multiple augmented versions of each image.

Augmentations:
- Original
- Horizontal flip
- Vertical flip  
- Rotate 90°, 180°, 270°

Usage:
    python test_time_augmentation.py --model_path training_output_kfold/fold_2_best.pth \
                                      --data_dir "Normal and diabetes" \
                                      --output_dir "tta_results"
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    f1_score, roc_curve, matthews_corrcoef
)
import timm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Configuration ────────────────────────────────────────────────────────────

IMG_SIZE = 512
BATCH_SIZE = 16  # Smaller batch for TTA (more augmentations)
NUM_WORKERS = 4
RANDOM_SEED = 42
TEST_RATIO = 0.15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset ──────────────────────────────────────────────────────────────────

class FundusDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def load_dataset(data_dir):
    """Load dataset paths and labels."""
    paths, labels = [], []
    class_map = {"No_DR": 0, "DR": 1, "Normal (N)": 0, "Diabetes (D)": 1}
    found = False
    for folder, label in class_map.items():
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        found = True
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(folder_path, fname))
                labels.append(label)
    if not found:
        raise FileNotFoundError(
            f"No recognised folders in '{data_dir}'. "
            f"Expected: No_DR/DR or 'Normal (N)'/'Diabetes (D)'"
        )
    return paths, labels


def get_base_transforms():
    """Base transforms without augmentation."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]),
    ])

# ── Model Loading ────────────────────────────────────────────────────────────

def load_model(ckpt_path):
    """Load the trained model."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    
    # Determine architecture and num_classes
    classifier_key = None
    for key in state.keys():
        if "classifier" in key and "weight" in key:
            classifier_key = key
            break
    
    num_classes = state[classifier_key].shape[0] if classifier_key else 2
    
    # Detect architecture
    conv_head_key = "conv_head.weight"
    if conv_head_key in state:
        head_channels = state[conv_head_key].shape[0]
        if head_channels == 1280:
            arch = "efficientnet_b0"
        elif head_channels == 1408:
            arch = "efficientnet_b2"
        elif head_channels == 1536:
            arch = "efficientnet_b3"
        else:
            arch = "efficientnet_b2"
    else:
        arch = "efficientnet_b2"
    
    print(f"Loading: {arch} with num_classes={num_classes}")
    
    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval(), num_classes, arch

# ── Test-Time Augmentation ──────────────────────────────────────────────────

def apply_tta_augmentations(img_tensor):
    """
    Apply TTA augmentations to a single image tensor.
    Returns list of augmented versions.
    """
    augmentations = [
        img_tensor,                                      # Original
        torch.flip(img_tensor, dims=[2]),               # Horizontal flip
        torch.flip(img_tensor, dims=[1]),               # Vertical flip
        torch.rot90(img_tensor, k=1, dims=[1, 2]),      # Rotate 90°
        torch.rot90(img_tensor, k=2, dims=[1, 2]),      # Rotate 180°
        torch.rot90(img_tensor, k=3, dims=[1, 2]),      # Rotate 270°
    ]
    return augmentations


@torch.no_grad()
def predict_with_tta(model, loader, num_classes=1, use_tta=True):
    """
    Generate predictions with optional test-time augmentation.
    
    Args:
        model: Trained model
        loader: Data loader
        num_classes: 1 (BCE) or 2 (CrossEntropy)
        use_tta: If True, apply TTA and average predictions
    
    Returns:
        probs: Array of predicted probabilities
        targets: Array of true labels
    """
    all_probs = []
    all_targets = []
    
    for imgs, labels in loader:
        batch_probs = []
        
        for img in imgs:
            if use_tta:
                # Apply augmentations
                aug_imgs = apply_tta_augmentations(img)
                aug_batch = torch.stack(aug_imgs).to(DEVICE)
                
                # Get predictions for all augmentations
                logits = model(aug_batch)
                
                if num_classes == 2:
                    probs = torch.softmax(logits, dim=1)[:, 1]
                else:
                    probs = torch.sigmoid(logits.squeeze(1))
                
                # Average predictions
                avg_prob = probs.mean().item()
            else:
                # No TTA - single prediction
                img_batch = img.unsqueeze(0).to(DEVICE)
                logits = model(img_batch)
                
                if num_classes == 2:
                    avg_prob = torch.softmax(logits, dim=1)[0, 1].item()
                else:
                    avg_prob = torch.sigmoid(logits[0, 0]).item()
            
            batch_probs.append(avg_prob)
        
        all_probs.extend(batch_probs)
        all_targets.extend(labels.numpy().tolist())
    
    return np.array(all_probs), np.array(all_targets)

# ── Metrics ──────────────────────────────────────────────────────────────────

def bootstrap_mcc(y_true, y_pred, n=1000, seed=42):
    """Bootstrap 95% CI for MCC."""
    np.random.seed(seed)
    scores = []
    n_samples = len(y_true)
    for _ in range(n):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(matthews_corrcoef(yt, yp))
    scores = np.array(scores)
    return float(np.mean(scores)), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def compute_metrics(y_true, y_probs, threshold=0.5):
    """Compute classification metrics including MCC with bootstrap CI."""
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    f1          = f1_score(y_true, y_pred, zero_division=0)
    auc         = roc_auc_score(y_true, y_probs)
    ap          = average_precision_score(y_true, y_probs)
    mcc         = matthews_corrcoef(y_true, y_pred)

    print("  Computing MCC bootstrap CI (1000 iterations)...")
    mcc_mean, mcc_lo, mcc_hi = bootstrap_mcc(y_true, y_pred)

    return {
        "threshold":   threshold,
        "accuracy":    round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision":   round(precision, 4),
        "f1_score":    round(f1, 4),
        "roc_auc":     round(auc, 4),
        "pr_auc":      round(ap, 4),
        "mcc":         round(mcc, 4),
        "mcc_ci":      f"[{mcc_lo:.4f}, {mcc_hi:.4f}]",
        "confusion_matrix": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}
    }


def find_optimal_threshold(y_true, y_probs):
    """Find optimal threshold using Youden's index."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    youden_idx = np.argmax(tpr - fpr)
    return float(thresholds[youden_idx])

# ── Visualization ────────────────────────────────────────────────────────────

def plot_comparison(results_no_tta, results_tta, output_path):
    """Plot comparison between regular and TTA predictions."""
    metrics = ["accuracy", "sensitivity", "specificity", "roc_auc", "pr_auc"]
    no_tta_vals = [results_no_tta[m] for m in metrics]
    tta_vals = [results_tta[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, no_tta_vals, width, label='Without TTA', alpha=0.8)
    bars2 = ax.bar(x + width/2, tta_vals, width, label='With TTA', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Comparison: TTA vs No TTA', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim([0.7, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved → {output_path}")

# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Device: {DEVICE}")
    print(f"Loading model from: {args.model_path}\n")
    
    # Load model
    model, num_classes, arch = load_model(args.model_path)
    
    # Load dataset
    paths, labels = load_dataset(args.data_dir)
    # Use ALL images as external validation
    # (models were trained on a separate dataset)
    test_paths = paths
    test_labels = labels
    transform = get_base_transforms()
    test_dataset = FundusDataset(test_paths, test_labels, transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)
    
    print(f"External validation set: {len(test_labels)} images\n")
    
    # ── Evaluate WITHOUT TTA ────────────────────────────────────────────────
    print("="*70)
    print("EVALUATION WITHOUT TEST-TIME AUGMENTATION")
    print("="*70)
    
    y_probs_no_tta, y_true = predict_with_tta(model, test_loader, num_classes, use_tta=False)
    threshold_no_tta = find_optimal_threshold(y_true, y_probs_no_tta)
    results_no_tta = compute_metrics(y_true, y_probs_no_tta, threshold_no_tta)
    
    print(f"Optimal threshold: {threshold_no_tta:.4f}")
    print(f"Accuracy:    {results_no_tta['accuracy']:.4f}")
    print(f"Sensitivity: {results_no_tta['sensitivity']:.4f}")
    print(f"Specificity: {results_no_tta['specificity']:.4f}")
    print(f"ROC-AUC:     {results_no_tta['roc_auc']:.4f}")
    print(f"PR-AUC:      {results_no_tta['pr_auc']:.4f}")
    print(f"MCC:         {results_no_tta['mcc']:.4f}  (95% CI: {results_no_tta['mcc_ci']})")
    
    # ── Evaluate WITH TTA ───────────────────────────────────────────────────
    print("\n" + "="*70)
    print("EVALUATION WITH TEST-TIME AUGMENTATION (6 augmentations)")
    print("="*70)
    print("Applying TTA (this may take a few minutes)...\n")
    
    y_probs_tta, _ = predict_with_tta(model, test_loader, num_classes, use_tta=True)
    threshold_tta = find_optimal_threshold(y_true, y_probs_tta)
    results_tta = compute_metrics(y_true, y_probs_tta, threshold_tta)
    
    print(f"Optimal threshold: {threshold_tta:.4f}")
    print(f"Accuracy:    {results_tta['accuracy']:.4f}")
    print(f"Sensitivity: {results_tta['sensitivity']:.4f}")
    print(f"Specificity: {results_tta['specificity']:.4f}")
    print(f"ROC-AUC:     {results_tta['roc_auc']:.4f}")
    print(f"PR-AUC:      {results_tta['pr_auc']:.4f}")
    print(f"MCC:         {results_tta['mcc']:.4f}  (95% CI: {results_tta['mcc_ci']})")
    
    # ── Compare Results ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("IMPROVEMENT WITH TTA")
    print("="*70)
    
    improvements = {}
    for metric in ["accuracy", "sensitivity", "specificity", "roc_auc", "pr_auc", "mcc"]:
        diff = results_tta[metric] - results_no_tta[metric]
        pct_change = (diff / results_no_tta[metric]) * 100 if results_no_tta[metric] > 0 else 0
        improvements[metric] = {
            "absolute": round(diff, 4),
            "percent": round(pct_change, 2)
        }
        sign = "+" if diff >= 0 else ""
        print(f"{metric:15s}: {sign}{diff:+.4f} ({sign}{pct_change:+.2f}%)")
    
    # Save results
    output_data = {
        "model_info": {
            "architecture": arch,
            "num_classes": num_classes,
            "checkpoint": args.model_path
        },
        "without_tta": results_no_tta,
        "with_tta": results_tta,
        "improvements": improvements
    }
    
    with open(os.path.join(args.output_dir, "tta_results.json"), "w") as f:
        json.dump(output_data, f, indent=2)
    
    # Plot comparison
    plot_comparison(results_no_tta, results_tta,
                   os.path.join(args.output_dir, "tta_comparison.png"))
    
    print(f"\n{'='*70}")
    print("TTA ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test-time augmentation evaluation")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", default="tta_results", help="Output directory")
    args = parser.parse_args()
    
    main(args)