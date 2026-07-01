# requirements: pip install torch torchvision timm albumentations opencv-python pandas scikit-learn tqdm pytorch-grad-cam
import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import timm
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
import os
# Change to your project directory
os.chdir(r"")
#This script tests multiple AI's and creates gradcam results
# ---- Utility: simple fundus crop ----
def crop_circle_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    cropped = img[y:y+h, x:x+w]
    return cropped

# ---- Dataset ----
class FundusDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None, size=512):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = transforms
        self.size = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'])
        img = cv2.imread(img_path)[:,:,::-1]
        img = crop_circle_image(img)
        img = cv2.resize(img, (self.size, self.size))
        if self.transforms:
            img = self.transforms(image=img)['image']
        label = torch.tensor(row['label']).long()
        return img, label, img_path

# ---- Transforms ----
valid_transforms = A.Compose([
    A.Resize(512,512),
    A.Normalize(),
    ToTensorV2(),
])

# ---- Load metadata ----
meta = pd.read_csv(r"your meta.csv")
train_df, valid_df = train_test_split(meta, test_size=0.2, stratify=meta['label'], random_state=42)

# ---- Setup ----
data_dir = r"your dataset"
valid_ds = FundusDataset(valid_df, img_dir=data_dir, transforms=valid_transforms)
valid_loader = DataLoader(valid_ds, batch_size=16, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Evaluation Function ----
def evaluate_model(model_path, loader, device):
    """Evaluate a model and return comprehensive metrics"""
    # Load model
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for imgs, labels, _ in tqdm(loader, desc=f"Evaluating {os.path.basename(model_path)}", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = (all_preds == all_labels).mean()
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate detailed metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for diabetic
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall for normal
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision for diabetic
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Precision for normal
    
    # F-scores
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
    f2 = 5 * (ppv * sensitivity) / (4 * ppv + sensitivity) if (4 * ppv + sensitivity) > 0 else 0
    
    # MCC
    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_num / mcc_den if mcc_den > 0 else 0
    
    return {
        'model': model,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'f1_score': f1,
        'f2_score': f2,
        'mcc': mcc,
        'confusion_matrix': cm,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
    }

# ---- GradCAM Visualization ----
def generate_gradcam_comparison(models_dict, sample_image_path, output_path="gradcam_comparison.png"):
    """Generate GradCAM visualizations for all models on the same image"""
    
    # Load and preprocess the image
    img_bgr = cv2.imread(sample_image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = crop_circle_image(img_rgb)
    img_rgb = cv2.resize(img_rgb, (512, 512))
    
    # Normalize for display
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # Prepare input tensor
    transform = A.Compose([
        A.Normalize(),
        ToTensorV2(),
    ])
    input_tensor = transform(image=img_rgb)['image'].unsqueeze(0).to(device)
    
    # Create figure
    num_models = len(models_dict)
    fig, axes = plt.subplots(1, num_models + 1, figsize=(5 * (num_models + 1), 5))
    
    # Show original image
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Generate GradCAM for each model
    for idx, (model_name, result) in enumerate(models_dict.items()):
        model = result['model']
        
        # Get target layer (last convolutional layer of EfficientNet)
        target_layers = [model.conv_head]
        
        # Create GradCAM
        cam = GradCAM(model=model, target_layers=target_layers)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
        
        # Generate CAM (use predicted class as target)
        targets = [ClassifierOutputTarget(pred_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Overlay CAM on image
        cam_image = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
        
        # Display
        axes[idx + 1].imshow(cam_image)
        class_name = "Diabetic" if pred_class == 1 else "Normal"
        title = f"{os.path.basename(model_name).replace('.pth', '')}\n{class_name} ({confidence*100:.1f}%)"
        axes[idx + 1].set_title(title, fontsize=10, fontweight='bold')
        axes[idx + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ GradCAM comparison saved to: {output_path}")
    plt.close()

# ---- Compare Models ----
print("\n" + "="*100)
print(" "*35 + "MODEL COMPARISON TOOL")
print("="*100 + "\n")

# Automatically find all .pth files in the directory
print("🔍 Scanning for model files (.pth)...\n")
all_files = os.listdir('.')
model_files = [f for f in all_files if f.endswith('.pth')]

# Filter to only existing models (redundant but safe)
existing_models = [m for m in model_files if os.path.exists(m)]

if not existing_models:
    print("❌ No model files found! Please check your directory.")
    print(f"   Looking in: {os.getcwd()}")
    exit()

print(f"Found {len(existing_models)} model(s) to compare:\n")
for m in existing_models:
    print(f"  📁 {m}")
print()

# Evaluate all models
results = {}
for model_path in existing_models:
    print(f"\n{'='*100}")
    print(f"Evaluating: {model_path}")
    print('='*100)
    results[model_path] = evaluate_model(model_path, valid_loader, device)

# ---- Print Comparison Table ----
print("\n\n" + "="*100)
print(" "*35 + "COMPARISON RESULTS")
print("="*100 + "\n")

# Create comparison DataFrame
comparison_data = []
for model_name, metrics in results.items():
    comparison_data.append({
        'Model': os.path.basename(model_name).replace('.pth', ''),
        'Accuracy': f"{metrics['accuracy']:.4f}",
        'Sensitivity': f"{metrics['sensitivity']:.4f}",
        'Specificity': f"{metrics['specificity']:.4f}",
        'PPV': f"{metrics['ppv']:.4f}",
        'NPV': f"{metrics['npv']:.4f}",
        'F1-Score': f"{metrics['f1_score']:.4f}",
        'F2-Score': f"{metrics['f2_score']:.4f}",
        'MCC': f"{metrics['mcc']:.4f}",
    })

df_comparison = pd.DataFrame(comparison_data)
print(df_comparison.to_string(index=False))

# ---- Detailed Metrics for Each Model ----
print("\n\n" + "="*100)
print(" "*35 + "DETAILED BREAKDOWN")
print("="*100)

for model_name, metrics in results.items():
    print(f"\n\n📊 {model_name}")
    print("-" * 100)
    print(f"\n✅ Overall Performance:")
    print(f"   Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   F1-Score:     {metrics['f1_score']:.4f}")
    print(f"   MCC:          {metrics['mcc']:.4f}")
    
    print(f"\n🎯 Medical Metrics:")
    print(f"   Sensitivity:  {metrics['sensitivity']:.4f} ({metrics['sensitivity']*100:.2f}%) - Diabetic detection rate")
    print(f"   Specificity:  {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%) - Normal detection rate")
    print(f"   PPV:          {metrics['ppv']:.4f} ({metrics['ppv']*100:.2f}%) - When predicts diabetic, accuracy")
    print(f"   NPV:          {metrics['npv']:.4f} ({metrics['npv']*100:.2f}%) - When predicts normal, accuracy")
    
    print(f"\n📋 Confusion Matrix:")
    print(f"                  Predicted Normal    Predicted Diabetic")
    print(f"   Actual Normal:      {metrics['true_negatives']:>6}              {metrics['false_positives']:>6}")
    print(f"   Actual Diabetic:    {metrics['false_negatives']:>6}              {metrics['true_positives']:>6}")
    
    print(f"\n🔢 Raw Counts:")
    print(f"   True Positives:   {metrics['true_positives']} (Correctly identified diabetic)")
    print(f"   True Negatives:   {metrics['true_negatives']} (Correctly identified normal)")
    print(f"   False Positives:  {metrics['false_positives']} (Normal flagged as diabetic)")
    print(f"   False Negatives:  {metrics['false_negatives']} (Diabetic missed)")

# ---- Find Best Model ----
print("\n\n" + "="*100)
print(" "*35 + "RECOMMENDATIONS")
print("="*100 + "\n")

best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
best_sensitivity = max(results.items(), key=lambda x: x[1]['sensitivity'])
best_f1 = max(results.items(), key=lambda x: x[1]['f1_score'])
best_balanced = max(results.items(), key=lambda x: (x[1]['sensitivity'] + x[1]['specificity']) / 2)

print(f"🏆 Best Overall Accuracy:     {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.4f})")
print(f"🎯 Best Sensitivity:          {best_sensitivity[0]} ({best_sensitivity[1]['sensitivity']:.4f})")
print(f"⚖️  Best F1-Score:             {best_f1[0]} ({best_f1[1]['f1_score']:.4f})")
print(f"🔄 Best Balanced (Sens+Spec): {best_balanced[0]} (avg: {(best_balanced[1]['sensitivity'] + best_balanced[1]['specificity']) / 2:.4f})")

print("\n💡 For medical deployment, prioritize:")
print("   1. High Sensitivity (>90%) - Don't miss diabetic cases")
print("   2. Acceptable Specificity (>85%) - Minimize false alarms")
print("   3. High PPV - When flagged diabetic, be confident")

# ---- Generate GradCAM Comparison ----
print("\n\n" + "="*100)
print(" "*35 + "GRADCAM VISUALIZATION")
print("="*100 + "\n")

# Find a sample diabetic image for visualization
diabetic_samples = valid_df[valid_df['label'] == 1]
if len(diabetic_samples) > 0:
    sample_img = diabetic_samples.iloc[0]['image']
    sample_path = os.path.join(data_dir, sample_img)
    
    print(f"Generating GradCAM comparison for sample image: {sample_img}")
    print("This shows what each model focuses on when making predictions...\n")
    
    generate_gradcam_comparison(results, sample_path, "gradcam_comparison_diabetic.png")
    
    # Also generate for a normal sample
    normal_samples = valid_df[valid_df['label'] == 0]
    if len(normal_samples) > 0:
        sample_img = normal_samples.iloc[0]['image']
        sample_path = os.path.join(data_dir, sample_img)
        
        print(f"Generating GradCAM comparison for normal image: {sample_img}")
        generate_gradcam_comparison(results, sample_path, "gradcam_comparison_normal.png")
else:
    print("⚠️ No diabetic samples found in validation set for GradCAM visualization")

print("\n" + "="*100)
print("✅ Comparison Complete!")
print("   📊 Check the comparison tables above")
print("   🔍 Check gradcam_comparison_diabetic.png for visual comparison")
print("   🔍 Check gradcam_comparison_normal.png for visual comparison")
print("="*100 + "\n")
