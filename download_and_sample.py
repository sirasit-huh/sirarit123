import os
import zipfile
import io
import subprocess
import random
from pathlib import Path
from PIL import Image

RAW_DIR = Path("data_raw")
ZIP_PATH = RAW_DIR / "a-large-scale-fish-dataset.zip"
DATA_DIR = Path("data/fish_dataset")
SAMPLES_DIR = Path("sample_test_images")
SAMPLE_BASE_PER_CLASS = 150  # 150 base * 4 versions = 600 per class * 9 = 5,400 balanced images

def download_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 100_000:
        print(f"Found existing archive at {ZIP_PATH}")
        return ZIP_PATH
    
    print("Downloading 'crowww/a-large-scale-fish-dataset' from Kaggle...")
    cmd = ["kaggle", "datasets", "download", "-d", "crowww/a-large-scale-fish-dataset", "-p", str(RAW_DIR)]
    subprocess.run(cmd, check=True)
    print("Download completed successfully!")
    return ZIP_PATH

def sample_and_augment(zip_path):
    # Clean existing data directory to prevent stale files
    if DATA_DIR.exists():
        import shutil
        shutil.rmtree(DATA_DIR)
        
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading zip archive: {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        all_files = set(z.namelist())
        
        # Discover species classes
        class_files = {}
        for fname in all_files:
            lower = fname.lower()
            if not (lower.endswith('.png') or lower.endswith('.jpg') or lower.endswith('.jpeg')):
                continue
            if " gt" in lower or "_gt" in lower or "gt/" in lower:
                continue
            
            parts = Path(fname).parts
            if len(parts) >= 4:
                species = parts[-2]
                if "gt" in species.lower():
                    continue
                if species not in class_files:
                    class_files[species] = []
                class_files[species].append(fname)
                
        species_list = sorted(list(class_files.keys()))
        print(f"Discovered {len(species_list)} species: {species_list}")
        
        random.seed(42)
        total_created = 0
        
        # Realistic background palettes to combat domain shift
        tint_palettes = [
            (240, 240, 242),  # Clean light gray
            (235, 225, 210),  # Light wooden cutting board
            (210, 225, 235),  # Ice / water tint
            (225, 225, 225),  # Stainless steel counter
            (245, 238, 230),  # White-beige prep tray
        ]

        for species in species_list:
            out_class_dir = DATA_DIR / species
            out_class_dir.mkdir(parents=True, exist_ok=True)
            
            files = sorted(class_files[species])
            sampled = random.sample(files, min(len(files), SAMPLE_BASE_PER_CLASS))
            print(f"Processing {len(sampled)} base images for '{species}'...")
            
            for i, fpath in enumerate(sampled):
                fname_only = Path(fpath).name
                gt_path = fpath.replace(f"/{species}/{fname_only}", f"/{species} GT/{fname_only}")
                
                img_data = z.read(fpath)
                orig_img = Image.open(io.BytesIO(img_data)).convert("RGB")
                
                # 1. Original Image (Blue Tray)
                out_orig_name = f"{species.replace(' ', '_')}_{i:04d}_orig.png"
                orig_img.save(out_class_dir / out_orig_name)
                total_created += 1
                
                # Check for GT mask
                has_gt = gt_path in all_files
                if has_gt:
                    mask_data = z.read(gt_path)
                    mask_img = Image.open(io.BytesIO(mask_data)).convert("L")
                    
                    # 2. White Background Image (Clean Canvas)
                    white_canvas = Image.new("RGB", orig_img.size, (255, 255, 255))
                    white_canvas.paste(orig_img, mask=mask_img)
                    out_white_name = f"{species.replace(' ', '_')}_{i:04d}_white.png"
                    white_canvas.save(out_class_dir / out_white_name)
                    total_created += 1
                    
                    # 3. Synthetic Realistic Tinted Background Image
                    tint_color = random.choice(tint_palettes)
                    tint_canvas = Image.new("RGB", orig_img.size, tint_color)
                    tint_canvas.paste(orig_img, mask=mask_img)
                    out_tint_name = f"{species.replace(' ', '_')}_{i:04d}_tint.png"
                    tint_canvas.save(out_class_dir / out_tint_name)
                    total_created += 1
                    
                    # 4. Flipped White Canvas (Orientation variation)
                    flipped_white = white_canvas.transpose(Image.FLIP_LEFT_RIGHT)
                    out_flip_name = f"{species.replace(' ', '_')}_{i:04d}_flip.png"
                    flipped_white.save(out_class_dir / out_flip_name)
                    total_created += 1
                else:
                    # Fallback if GT missing: horizontal flip
                    flipped = orig_img.transpose(Image.FLIP_LEFT_RIGHT)
                    out_flip_name = f"{species.replace(' ', '_')}_{i:04d}_flip.png"
                    flipped.save(out_class_dir / out_flip_name)
                    total_created += 1
                    
                # Save demo samples
                if i == 0:
                    orig_img.save(SAMPLES_DIR / f"sample_{species.replace(' ', '_')}_0000.png")
                    if has_gt:
                        white_canvas.save(SAMPLES_DIR / f"sample_{species.replace(' ', '_')}_white.png")

    print(f"\n[SUCCESS] Successfully prepared multi-background dataset with {total_created} images in {DATA_DIR}!")

if __name__ == "__main__":
    archive = download_dataset()
    sample_and_augment(archive)
