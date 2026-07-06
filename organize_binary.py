import os
import shutil

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SOURCE_DIR  = "colored_images"          # folder containing the 5 subfolders
OUTPUT_DIR  = "binary_dataset"          # where the two new folders will be created
NORMAL_FOLDER   = "No_DR"              # kept as-is
DIABETIC_FOLDER = "DR"                 # everything else merged here
DR_CLASSES = ["Mild", "Moderate", "Severe", "Proliferate_DR"]
# ───────────────────────────────────────────────────────────────────────────────

def organize_binary():
    normal_out   = os.path.join(OUTPUT_DIR, NORMAL_FOLDER)
    diabetic_out = os.path.join(OUTPUT_DIR, DIABETIC_FOLDER)
    os.makedirs(normal_out,   exist_ok=True)
    os.makedirs(diabetic_out, exist_ok=True)

    # ── Copy No_DR images ──────────────────────────────────────────────────────
    normal_src = os.path.join(SOURCE_DIR, NORMAL_FOLDER)
    if not os.path.isdir(normal_src):
        print(f"ERROR: Could not find '{normal_src}'. Check SOURCE_DIR and folder name.")
        return

    copied_normal = 0
    for fname in os.listdir(normal_src):
        src  = os.path.join(normal_src, fname)
        dest = os.path.join(normal_out, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dest)
            copied_normal += 1

    print(f"No_DR  → copied {copied_normal} images")

    # ── Copy DR images (Mild + Moderate + Severe + Proliferate_DR) ────────────
    copied_dr = 0
    for cls in DR_CLASSES:
        cls_src = os.path.join(SOURCE_DIR, cls)
        if not os.path.isdir(cls_src):
            print(f"  WARNING: folder '{cls_src}' not found — skipping")
            continue

        count = 0
        for fname in os.listdir(cls_src):
            src = os.path.join(cls_src, fname)
            if not os.path.isfile(src):
                continue

            # Avoid name collisions by prefixing with the class name
            dest_fname = f"{cls}_{fname}"
            dest = os.path.join(diabetic_out, dest_fname)
            shutil.copy2(src, dest)
            count += 1

        print(f"  {cls:<18} → copied {count} images")
        copied_dr += count

    print(f"DR     → copied {copied_dr} images total")
    print(f"\nDone! Binary dataset saved to '{OUTPUT_DIR}/'")
    print(f"  {OUTPUT_DIR}/No_DR/  ({copied_normal} images)")
    print(f"  {OUTPUT_DIR}/DR/     ({copied_dr} images)")

if __name__ == "__main__":
    organize_binary()
