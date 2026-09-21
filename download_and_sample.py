import os
import zipfile
import subprocess
import shutil
from pathlib import Path
import random

RAW_DIR = Path("data_raw")
DATA_DIR = Path("data/fish_dataset")
SAMPLES_DIR = Path("sample_test_images")
SAMPLE_PER_CLASS = 250  # 250 images * 9 classes = 2,250 images (optimal balance of speed and high accuracy)

def download_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "a-large-scale-fish-dataset.zip"
    
    if zip_path.exists() and zip_path.stat().st_size > 100_000:
        print(f"Found existing archive at {zip_path}")
        return zip_path
    
    print("Downloading 'crowww/a-large-scale-fish-dataset' from Kaggle...")
    cmd = ["kaggle", "datasets", "download", "-d", "crowww/a-large-scale-fish-dataset", "-p", str(RAW_DIR)]
    subprocess.run(cmd, check=True)
    print("Download completed successfully!")
    return zip_path

def sample_and_extract(zip_path):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Opening {zip_path} and scanning images...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        all_files = z.namelist()
        
        # Identify image files (exclude GT/masks and non-image files)
        # Structure in dataset is typically:
        # Fish_Dataset/Fish_Dataset/<Species>/<Species>/00001.png
        # Fish_Dataset/Fish_Dataset/<Species>/<Species> GT/00001.png
        class_files = {}
        for fname in all_files:
            lower = fname.lower()
            if not (lower.endswith('.png') or lower.endswith('.jpg') or lower.endswith('.jpeg')):
                continue
            if " gt" in lower or "_gt" in lower or "gt/" in lower:
                # Exclude Ground Truth masks
                continue
            
            parts = Path(fname).parts
            # Look for folder structure
            # Example: ('Fish_Dataset', 'Fish_Dataset', 'Sea Bass', 'Sea Bass', '00001.png')
            if len(parts) >= 4:
                # The species name is typically parts[-2] or parts[-3]
                species = parts[-2]
                if "gt" in species.lower():
                    continue
                if species not in class_files:
                    class_files[species] = []
                class_files[species].append(fname)
        
        print(f"Detected {len(class_files)} classes: {list(class_files.keys())}")
        
        # Ensure repeatable sampling
        random.seed(42)
        
        for species, files in class_files.items():
            out_class_dir = DATA_DIR / species
            out_class_dir.mkdir(parents=True, exist_ok=True)
            
            sampled = random.sample(files, min(len(files), SAMPLE_PER_CLASS))
            print(f"Extracting {len(sampled)} images for class: '{species}'...")
            
            for i, fpath in enumerate(sampled):
                img_data = z.read(fpath)
                out_name = f"{species.replace(' ', '_')}_{i:04d}{Path(fpath).suffix}"
                with open(out_class_dir / out_name, "wb") as f_out:
                    f_out.write(img_data)
                
                # Also save the first image of each species to sample_test_images for Streamlit quick demo
                if i == 0:
                    with open(SAMPLES_DIR / f"sample_{out_name}", "wb") as f_sample:
                        f_sample.write(img_data)

    print(f"\nSuccessfully prepared dataset in {DATA_DIR}!")
    print(f"Sample test images available in {SAMPLES_DIR}!")

if __name__ == "__main__":
    archive = download_dataset()
    sample_and_extract(archive)
