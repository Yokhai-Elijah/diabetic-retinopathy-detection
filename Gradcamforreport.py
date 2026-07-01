"""
gradcam_paper_figures.py
------------------------
Runs Grad-CAM on fold_2_best.pth and saves only the best representative
examples for publication figures:
  - 2 best true positives  (high confidence, clear heatmap)
  - 2 best true negatives  (high confidence, minimal activation)
  - 1 best false positive  (healthy flagged as DR)
  - 1 best false negative  (DR missed by model)

Much faster than running on all 3,662 images.

Usage:
    python gradcam_paper_figures.py --model_path training_output_kfold/fold_2_best.pth
                                     --image_dir binary_dataset
                                     --output_dir gradcam_paper
                                     --threshold 0.5409
"""

import os
import json
import argparse
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
import timm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512
LABEL_MAP = {"No_DR": 0, "DR": 1, "normal": 0, "diabetic": 1}

# ── Grad-CAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, m, i, o): self.activations = o.detach()
    def _save_gradient(self, m, gi, go): self.gradients   = go[0].detach()

    def generate(self, tensor, class_idx=1):
        self.model.eval()
        out = self.model(tensor)
        self.model.zero_grad()
        if out.shape[1] == 2:
            one_hot = torch.zeros_like(out)
            one_hot[0, class_idx] = 1
            out.backward(gradient=one_hot, retain_graph=True)
        else:
            out.backward(retain_graph=True)
        weights = self.gradients.mean(dim=[0, 2, 3])
        acts    = self.activations[0].clone()
        for i, w in enumerate(weights): acts[i] *= w
        cam = acts.mean(0).cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0: cam /= cam.max()
        return cam

# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(path):
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    num_classes = 2
    for k, v in state.items():
        if "classifier" in k and "weight" in k:
            num_classes = v.shape[0]; break
    ch   = state.get("conv_head.weight", torch.zeros(1408)).shape[0]
    arch = {1280:"efficientnet_b0", 1408:"efficientnet_b2",
            1536:"efficientnet_b3"}.get(ch, "efficientnet_b2")
    print(f"Architecture: {arch} | classes: {num_classes}")
    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval(), num_classes

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def overlay(img_np, heatmap, alpha=0.45):
    h = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    colored = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return np.clip((1 - alpha) * img_np + alpha * colored, 0, 255).astype(np.uint8)

# ── Collection phase ──────────────────────────────────────────────────────────

def collect_candidates(model, num_classes, image_dir, threshold, gcam, transform):
    """Run inference on all images, collect metadata, no figure saving yet."""
    buckets = {"TP": [], "TN": [], "FP": [], "FN": []}

    for folder, true_label in LABEL_MAP.items():
        fpath = os.path.join(image_dir, folder)
        if not os.path.isdir(fpath):
            continue
        files = [f for f in os.listdir(fpath)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        print(f"  Scanning {folder} ({len(files)} images)...")
        for fname in files:
            img_path = os.path.join(fpath, fname)
            try:
                img_pil = Image.open(img_path).convert("RGB")
                tensor  = transform(img_pil).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    out  = model(tensor)
                    prob = (torch.softmax(out, 1)[0, 1].item()
                            if num_classes == 2
                            else torch.sigmoid(out[0, 0]).item())

                pred = 1 if prob >= threshold else 0
                cat  = ("TP" if true_label == 1 and pred == 1 else
                        "TN" if true_label == 0 and pred == 0 else
                        "FP" if true_label == 0 and pred == 1 else "FN")

                # Confidence = distance from 0.5 (higher = more certain)
                confidence = abs(prob - 0.5)

                buckets[cat].append({
                    "path":       img_path,
                    "fname":      fname,
                    "folder":     folder,
                    "prob":       prob,
                    "true_label": true_label,
                    "pred":       pred,
                    "confidence": confidence,
                    "cat":        cat,
                })
            except Exception as e:
                print(f"    Warning: {fname}: {e}")

    return buckets

# ── Figure generation ─────────────────────────────────────────────────────────

def make_figure(entry, gcam, transform, num_classes, output_path, fig_label):
    """Generate and save a publication-quality 3-panel Grad-CAM figure."""
    img_pil = Image.open(entry["path"]).convert("RGB")
    img_np  = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE)))
    tensor  = transform(img_pil).unsqueeze(0).to(DEVICE)

    heatmap  = gcam.generate(tensor, class_idx=1)
    overlaid = overlay(img_np, heatmap)

    label_str = {0: "Normal (No DR)", 1: "Diabetic Retinopathy"}
    true_str  = label_str[entry["true_label"]]
    pred_str  = label_str[entry["pred"]]
    correct   = "✓ Correct" if entry["true_label"] == entry["pred"] else "✗ Incorrect"
    prob_pct  = entry["prob"] * 100

    fig = plt.figure(figsize=(15, 5.5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.05)

    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img_np)
    ax1.set_title("Original Image", fontsize=13, fontweight="bold", pad=8)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    ax2.set_title("Grad-CAM Heatmap", fontsize=13, fontweight="bold", pad=8)
    ax2.axis("off")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(overlaid)
    ax3.set_title("Overlay", fontsize=13, fontweight="bold", pad=8)
    ax3.axis("off")

    cat_colors = {"TP": "#1D7A54", "TN": "#2E5496", "FP": "#B85C00", "FN": "#8B0000"}
    color = cat_colors.get(entry["cat"], "black")

    fig.suptitle(
        f"[{fig_label}]  True: {true_str}  |  Predicted: {pred_str} "
        f"({prob_pct:.1f}%)  |  {correct}",
        fontsize=12, fontweight="bold", color=color, y=1.01
    )

    plt.savefig(output_path, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)

# ── Main ──────────────────────────────────────────────────────────────────────

TARGETS = {
    "TP": {"n": 2, "sort_key": lambda x: -x["confidence"],  "label_prefix": "TP"},
    "TN": {"n": 2, "sort_key": lambda x: -x["confidence"],  "label_prefix": "TN"},
    "FP": {"n": 1, "sort_key": lambda x:  x["confidence"],  "label_prefix": "FP"},
    "FN": {"n": 1, "sort_key": lambda x:  x["confidence"],  "label_prefix": "FN"},
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--image_dir",  required=True)
    parser.add_argument("--output_dir", default="gradcam_paper")
    parser.add_argument("--threshold",  type=float, default=0.5409)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device    : {DEVICE}")
    print(f"Threshold : {args.threshold}")
    print(f"Output    : {args.output_dir}/\n")

    model, num_classes = load_model(args.model_path)
    target_layer = model.conv_head
    gcam      = GradCAM(model, target_layer)
    transform = get_transform()

    # ── Phase 1: collect all predictions ──────────────────────────────────
    print("Phase 1: Scanning all images...")
    buckets = collect_candidates(model, num_classes, args.image_dir,
                                 args.threshold, gcam, transform)

    for cat, items in buckets.items():
        print(f"  {cat}: {len(items)} images")

    # ── Phase 2: select best examples & generate figures ──────────────────
    print("\nPhase 2: Generating publication figures...")
    summary = {}
    fig_counter = 1

    for cat, cfg in TARGETS.items():
        pool = sorted(buckets[cat], key=cfg["sort_key"])
        selected = pool[:cfg["n"]]

        if not selected:
            print(f"  WARNING: no {cat} images found")
            continue

        for i, entry in enumerate(selected, 1):
            label    = f"{cfg['label_prefix']}{i}"
            out_name = f"{label}_{entry['fname'].rsplit('.', 1)[0]}.jpg"
            out_path = os.path.join(args.output_dir, out_name)

            make_figure(entry, gcam, transform, num_classes, out_path, label)
            summary[label] = {
                "category":   cat,
                "file":       entry["fname"],
                "folder":     entry["folder"],
                "prob":       round(entry["prob"], 4),
                "confidence": round(entry["confidence"], 4),
                "true_label": entry["true_label"],
                "pred":       entry["pred"],
                "saved_as":   out_name,
            }
            print(f"  ✓ {label}: {entry['fname']} "
                  f"(prob={entry['prob']:.3f}, conf={entry['confidence']:.3f})")
        fig_counter += cfg["n"]

    # ── Phase 3: combined figure for paper ────────────────────────────────
    print("\nPhase 3: Building combined 2×3 panel figure for paper...")
    ordered_keys = [k for k in ["TP1","TP2","TN1","TN2","FP1","FN1"] if k in summary]
    n_panels = len(ordered_keys)

    fig, axes = plt.subplots(n_panels, 3, figsize=(15, 5 * n_panels))
    if n_panels == 1: axes = [axes]

    cat_colors = {"TP":"#1D7A54","TN":"#2E5496","FP":"#B85C00","FN":"#8B0000"}
    label_str  = {0:"Normal (No DR)", 1:"Diabetic Retinopathy"}

    for row, key in enumerate(ordered_keys):
        info    = summary[key]
        img_pil = Image.open(
            os.path.join(args.image_dir, info["folder"], info["file"])
        ).convert("RGB")
        img_np  = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE)))
        tensor  = transform(img_pil).unsqueeze(0).to(DEVICE)
        heatmap = gcam.generate(tensor, class_idx=1)
        overlaid = overlay(img_np, heatmap)

        axes[row][0].imshow(img_np);     axes[row][0].axis("off")
        axes[row][1].imshow(heatmap, cmap="jet"); axes[row][1].axis("off")
        axes[row][2].imshow(overlaid);   axes[row][2].axis("off")

        color = cat_colors.get(info["category"], "black")
        title = (f"[{key}] True: {label_str[info['true_label']]}  |  "
                 f"Pred: {label_str[info['pred']]} ({info['prob']*100:.1f}%)")
        axes[row][0].set_title(title, fontsize=10, color=color,
                               fontweight="bold", loc="left", pad=6)

    axes[0][0].set_title("Original",   fontsize=11, pad=4)
    axes[0][1].set_title("Grad-CAM",   fontsize=11, pad=4)
    axes[0][2].set_title("Overlay",    fontsize=11, pad=4)

    plt.suptitle(
        "Figure 3: Grad-CAM Visualizations — Representative Cases\n"
        "Green=TP, Blue=TN, Orange=FP, Red=FN",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    combined_path = os.path.join(args.output_dir, "Figure3_GradCAM_combined.png")
    plt.savefig(combined_path, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  ✓ Combined figure saved → {combined_path}")

    # Save summary JSON
    with open(os.path.join(args.output_dir, "figure_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*55}")
    print("DONE — figures saved to:", args.output_dir)
    print(f"{'='*55}")
    print("\nFiles for your report:")
    for key, info in summary.items():
        print(f"  {key}: {info['saved_as']}")
    print(f"  Combined: Figure3_GradCAM_combined.png  ← use this in report")

if __name__ == "__main__":
    main()