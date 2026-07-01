"""
calculate_mcc.py
----------------
Calculates Matthews Correlation Coefficient (MCC) with 95% bootstrap
confidence intervals for the best model, without overwriting any
existing results.

Usage:
    python calculate_mcc.py --model_path training_output_kfold/fold_2_best.pth
                             --data_dir binary_dataset

Output:
    Prints MCC + CI to console
    Appends mcc result to statistical_results/mcc_result.json
"""

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import matthews_corrcoef
import timm

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512
SEED     = 42
N_BOOT   = 1000
LABEL_MAP = {"No_DR": 0, "DR": 1, "Normal (N)": 0, "Diabetes (D)": 1}

# ── Dataset ───────────────────────────────────────────────────────────────────

class FundusDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]

def load_dataset(data_dir):
    paths, labels = [], []
    found = False
    for folder, label in LABEL_MAP.items():
        fp = os.path.join(data_dir, folder)
        if not os.path.isdir(fp):
            continue
        found = True
        for fname in os.listdir(fp):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(fp, fname))
                labels.append(label)
    if not found:
        raise FileNotFoundError(
            f"No recognised folders in '{data_dir}'. "
            f"Expected: No_DR/DR or 'Normal (N)'/'Diabetes (D)'"
        )
    return paths, labels

# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(ckpt_path):
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    num_classes = 2
    for k, v in state.items():
        if "classifier" in k and "weight" in k:
            num_classes = v.shape[0]; break

    ch  = state.get("conv_head.weight", torch.zeros(1280)).shape[0]
    arch = {1280: "efficientnet_b0", 1408: "efficientnet_b2",
            1536: "efficientnet_b3"}.get(ch, "efficientnet_b2")

    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval(), num_classes

# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_probs(model, loader, num_classes):
    probs, targets = [], []
    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE)
        logits = model(imgs)
        prob   = (torch.softmax(logits, 1)[:, 1] if num_classes == 2
                  else torch.sigmoid(logits.squeeze(1)))
        probs.extend(prob.cpu().tolist())
        targets.extend(labels.tolist())
    return np.array(probs), np.array(targets)

# ── Bootstrap MCC ─────────────────────────────────────────────────────────────

def bootstrap_mcc(y_true, y_pred, n=1000, seed=42):
    np.random.seed(seed)
    scores = []
    n_samples = len(y_true)
    for _ in range(n):
        idx  = np.random.choice(n_samples, n_samples, replace=True)
        yt   = y_true[idx]
        yp   = y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(matthews_corrcoef(yt, yp))
    scores = np.array(scores)
    return float(np.mean(scores)), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--threshold",  type=float, default=0.5409,
                        help="Use the Youden threshold from statistical_analysis.py")
    parser.add_argument("--output_dir", default="statistical_results")
    args = parser.parse_args()

    print(f"Device : {DEVICE}")
    print(f"Loading : {args.model_path}")

    model, num_classes = load_model(args.model_path)

    paths, labels = load_dataset(args.data_dir)
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(FundusDataset(paths, labels, transform),
                        batch_size=32, shuffle=False, num_workers=0)

    print(f"Images  : {len(paths)} ({labels.count(0) if isinstance(labels, list) else (labels==0).sum()} normal, "
          f"{labels.count(1) if isinstance(labels, list) else (labels==1).sum()} diabetic)")
    print("Running inference...")

    y_probs, y_true = get_probs(model, loader, num_classes)
    y_pred = (y_probs >= args.threshold).astype(int)

    # Point estimate
    mcc_point = matthews_corrcoef(y_true, y_pred)

    # Bootstrap CI
    print(f"Running {N_BOOT} bootstrap iterations...")
    mcc_mean, mcc_lo, mcc_hi = bootstrap_mcc(y_true, y_pred, n=N_BOOT)

    print(f"\n{'='*50}")
    print(f"  MCC (point estimate) : {mcc_point:.4f}")
    print(f"  MCC (bootstrap mean) : {mcc_mean:.4f}")
    print(f"  95% CI               : [{mcc_lo:.4f}, {mcc_hi:.4f}]")
    print(f"{'='*50}")
    print(f"\nInterpretation: {mcc_point:.4f} is", end=" ")
    if mcc_point >= 0.9:
        print("excellent (≥0.90)")
    elif mcc_point >= 0.7:
        print("strong (0.70–0.90)")
    elif mcc_point >= 0.5:
        print("moderate (0.50–0.70)")
    else:
        print("weak (<0.50)")

    # Save without overwriting existing results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "mcc_result.json")
    result = {
        "model": args.model_path,
        "dataset": args.data_dir,
        "threshold": args.threshold,
        "n_images": len(paths),
        "mcc_point_estimate": round(mcc_point, 4),
        "mcc_bootstrap_mean": round(mcc_mean, 4),
        "ci_lower": round(mcc_lo, 4),
        "ci_upper": round(mcc_hi, 4),
        "ci_95": f"[{mcc_lo:.4f}, {mcc_hi:.4f}]"
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to: {out_path}")
    print("\nAdd this to your report:")
    print(f"  MCC: {mcc_point:.4f} (95% CI: [{mcc_lo:.4f}, {mcc_hi:.4f}])")

if __name__ == "__main__":
    main()