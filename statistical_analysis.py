"""
statistical_analysis.py
-----------------------
Adds statistical rigor to model evaluation:
1. Bootstrap confidence intervals (95%)
2. Precision-Recall AUC
3. Calibration curves
4. Enhanced metrics reporting

Usage:
    python statistical_analysis.py --model_path training_output_kfold/fold_2_best.pth \
                                    --data_dir "Normal and diabetes" \
                                    --output_dir "statistical_results"
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, f1_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve
import timm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import stats

# ── Configuration ────────────────────────────────────────────────────────────

IMG_SIZE = 512
BATCH_SIZE = 32
NUM_WORKERS = 4
RANDOM_SEED = 42
TEST_RATIO = 0.15
VAL_RATIO = 0.15
N_BOOTSTRAP = 1000  # Bootstrap iterations for CI

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
    """Return (paths, labels) for entire dataset."""
    paths, labels = [], []
    # Supports both original naming and binary dataset naming
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
            f"No recognised class folders found in '{data_dir}'.\n"
            f"Expected: No_DR/DR  or  'Normal (N)'/'Diabetes (D)'"
        )
    return paths, labels


def get_transforms():
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

# ── Prediction ───────────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(model, loader, num_classes=1):
    probs, targets = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        if num_classes == 2:
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        else:
            prob = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
        probs.extend(prob.tolist())
        targets.extend(labels.numpy().tolist())
    return np.array(probs), np.array(targets)

# ── Bootstrap Confidence Intervals ───────────────────────────────────────────

def bootstrap_ci(y_true, y_pred, y_probs, metric_fn, n_iterations=1000, ci=95):
    """Compute bootstrap confidence intervals for a metric."""
    np.random.seed(RANDOM_SEED)
    scores = []
    n_samples = len(y_true)
    
    for i in range(n_iterations):
        # Resample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        y_probs_boot = y_probs[indices]
        
        # Skip if only one class in bootstrap sample
        if len(np.unique(y_true_boot)) < 2:
            continue
        
        try:
            score = metric_fn(y_true_boot, y_pred_boot, y_probs_boot)
            scores.append(score)
        except:
            continue
    
    scores = np.array(scores)
    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    mean = np.mean(scores)
    
    return mean, lower, upper


def compute_metrics_with_ci(y_true, y_probs, threshold=0.5, n_bootstrap=1000):
    """Compute all metrics with confidence intervals."""
    y_pred = (y_probs >= threshold).astype(int)
    
    # Define metric functions
    def acc_fn(yt, yp, ypr): return np.mean(yt == yp)
    def sens_fn(yt, yp, ypr): return recall_score(yt, yp, zero_division=0)
    def spec_fn(yt, yp, ypr): 
        tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    def prec_fn(yt, yp, ypr): return precision_score(yt, yp, zero_division=0)
    def f1_fn(yt, yp, ypr): return f1_score(yt, yp, zero_division=0)
    def auc_fn(yt, yp, ypr): return roc_auc_score(yt, ypr)
    def prauc_fn(yt, yp, ypr): return average_precision_score(yt, ypr)
    
    results = {}
    
    # Point estimates
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    results["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    results["threshold"] = threshold
    
    # Compute metrics with CI
    metrics = {
        "accuracy": acc_fn,
        "sensitivity": sens_fn,
        "specificity": spec_fn,
        "precision": prec_fn,
        "f1_score": f1_fn,
        "roc_auc": auc_fn,
        "pr_auc": prauc_fn
    }
    
    print("\nComputing bootstrap confidence intervals...")
    for name, fn in metrics.items():
        mean, lower, upper = bootstrap_ci(y_true, y_pred, y_probs, fn, n_bootstrap)
        results[name] = {
            "value": round(mean, 4),
            "ci_lower": round(lower, 4),
            "ci_upper": round(upper, 4),
            "ci_95": f"[{lower:.4f}, {upper:.4f}]"
        }
        print(f"  {name:15s}: {mean:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")
    
    return results

# ── Calibration Analysis ─────────────────────────────────────────────────────

def plot_calibration_curve(y_true, y_probs, output_path, n_bins=10):
    """Plot calibration curve and compute ECE."""
    # Compute calibration
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins, strategy='uniform')
    
    # Expected Calibration Error (ECE)
    ece = np.mean(np.abs(prob_true - prob_pred))
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    ax.plot(prob_pred, prob_true, 'o-', label=f'Model (ECE={ece:.4f})', 
            linewidth=2, markersize=8)
    
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('True Probability', fontsize=12)
    ax.set_title('Calibration Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\nCalibration curve saved → {output_path}")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    
    return ece, prob_true, prob_pred

# ── ROC and PR Curves ────────────────────────────────────────────────────────

def plot_roc_and_pr_curves(y_true, y_probs, output_dir):
    """Plot ROC and Precision-Recall curves."""
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
    plt.close()
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})', linewidth=2)
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=150)
    plt.close()
    
    print(f"ROC curve saved → {output_dir}/roc_curve.png")
    print(f"PR curve saved → {output_dir}/pr_curve.png")

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
    transform = get_transforms()
    test_dataset = FundusDataset(test_paths, test_labels, transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)
    
    print(f"External validation set: {len(test_labels)} images")
    print(f"  No_DR:    {test_labels.count(0)}")
    print(f"  DR:       {test_labels.count(1)}\n")
    
    # Get predictions
    print("Generating predictions...")
    y_probs, y_true = get_predictions(model, test_loader, num_classes)
    
    # Find optimal threshold (Youden's index)
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    youden_idx = np.argmax(tpr - fpr)
    optimal_threshold = float(thresholds[youden_idx])
    
    print(f"Optimal threshold (Youden): {optimal_threshold:.4f}\n")
    
    # Compute metrics with CI
    print("="*70)
    print("STATISTICAL ANALYSIS WITH 95% CONFIDENCE INTERVALS")
    print("="*70)
    
    results = compute_metrics_with_ci(
        y_true, y_probs, 
        threshold=optimal_threshold,
        n_bootstrap=N_BOOTSTRAP
    )
    
    # Calibration analysis
    print("\n" + "="*70)
    print("CALIBRATION ANALYSIS")
    print("="*70)
    
    ece, prob_true, prob_pred = plot_calibration_curve(
        y_true, y_probs,
        os.path.join(args.output_dir, "calibration_curve.png")
    )
    
    results["calibration"] = {
        "ece": round(ece, 4),
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist()
    }
    
    # ROC and PR curves
    print("\n" + "="*70)
    print("GENERATING CURVES")
    print("="*70)
    
    plot_roc_and_pr_curves(y_true, y_probs, args.output_dir)
    
    # Save results
    results["model_info"] = {
        "architecture": arch,
        "num_classes": num_classes,
        "checkpoint": args.model_path
    }
    
    with open(os.path.join(args.output_dir, "statistical_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {args.output_dir}/")
    print("\nKey Findings:")
    print(f"  ROC-AUC:  {results['roc_auc']['value']:.4f} {results['roc_auc']['ci_95']}")
    print(f"  PR-AUC:   {results['pr_auc']['value']:.4f} {results['pr_auc']['ci_95']}")
    print(f"  Accuracy: {results['accuracy']['value']:.4f} {results['accuracy']['ci_95']}")
    print(f"  ECE:      {ece:.4f} (lower is better)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical analysis with CI and calibration")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", default="statistical_results", help="Output directory")
    args = parser.parse_args()
    
    main(args)