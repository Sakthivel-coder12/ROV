# scripts/prepare_and_split.py
import os
import random
import shutil
import time
from pathlib import Path
from sklearn.model_selection import train_test_split

# ---------- CONFIG ----------
RAW_ROOT = r"N://ROV//archive (1)//hagrid-classification-512p-127k"  # change to your Kaggle folder
OUT_ROOT = r"N://ROV//data"
RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
COPY_FILES = True   # if True, copies files; if False, will move files (use copy safer)
VERBOSE = True      # Set to True to see detailed progress
# ----------------------------

# mapping from original folder name -> desired class name
map_to_command = {
    "like": "Forward",
    "fist": "Reverse",
    "palm": "Stop",
    "one": "Left",
    "peace": "Right"
}

# Target class names (keeps desired order)
TARGET_CLASSES = ["Forward", "Reverse", "Stop", "Left", "Right", "Invalid"]

def ensure_dirs(base):
    """Create directory structure for the dataset splits"""
    for split in ("train", "val", "test"):
        for cls in TARGET_CLASSES:
            Path(base, split, cls).mkdir(parents=True, exist_ok=True)

def collect_images_per_class(raw_root):
    """Collect all image files and organize by class with progress tracking"""
    print(f"[INFO] Scanning source directory: {raw_root}")
    
    if not os.path.exists(raw_root):
        print(f"[ERROR] Source directory does not exist: {raw_root}")
        return {}
    
    entries = {}
    total_files = 0
    start_time = time.time()
    
    # First, count total directories to scan
    dirs_to_scan = [d for d in os.listdir(raw_root) if os.path.isdir(os.path.join(raw_root, d))]
    print(f"[INFO] Found {len(dirs_to_scan)} directories to scan")
    
    for i, src in enumerate(dirs_to_scan, 1):
        src_path = os.path.join(raw_root, src)
        
        # map folder name to class
        mapped = map_to_command.get(src.lower(), "Invalid")
        entries.setdefault(mapped, [])
        
        # Count files in this directory
        image_files = [f for f in os.listdir(src_path) 
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        # Add all image files in src_path
        for fname in image_files:
            entries[mapped].append(os.path.join(src_path, fname))
            total_files += 1
        
        if VERBOSE and i % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[PROGRESS] Scanned {i}/{len(dirs_to_scan)} directories, "
                  f"found {total_files} images so far... ({elapsed:.1f}s)")
    
    # Print summary
    elapsed = time.time() - start_time
    print(f"\n[SUMMARY] Found {total_files} total images in {len(dirs_to_scan)} directories")
    for cls in sorted(entries.keys()):
        print(f"  {cls}: {len(entries[cls])} images")
    print(f"Scanning completed in {elapsed:.2f} seconds")
    
    return entries

def copy_split(entries, out_root):
    """Split and copy/move files with detailed progress tracking"""
    random.seed(RANDOM_SEED)
    start_time = time.time()
    total_copied = 0
    total_to_copy = sum(len(files) for files in entries.values())
    
    print(f"\n[INFO] Starting to process {total_to_copy} images...")
    print("=" * 60)
    
    for cls_idx, (cls, files) in enumerate(entries.items(), 1):
        n = len(files)
        if n == 0:
            print(f"[WARNING] Class '{cls}' has no images, skipping")
            continue
            
        print(f"\nProcessing class '{cls}' ({cls_idx}/{len(entries)}): {n} images")
        
        # Split files
        train_files, temp = train_test_split(
            files, 
            train_size=TRAIN_RATIO, 
            random_state=RANDOM_SEED, 
            shuffle=True
        )
        val_files, test_files = train_test_split(
            temp, 
            test_size=TEST_RATIO/(TEST_RATIO+VAL_RATIO), 
            random_state=RANDOM_SEED, 
            shuffle=True
        )
        
        splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }
        
        class_copied = 0
        
        # Process each split
        for split_name, file_list in splits.items():
            dest_dir = os.path.join(out_root, split_name, cls)
            split_count = len(file_list)
            
            print(f"  {split_name}: {split_count} images", end="")
            if VERBOSE:
                print("")
            
            for file_idx, src_file in enumerate(file_list, 1):
                dst = os.path.join(dest_dir, os.path.basename(src_file))
                
                # Handle filename collisions
                if os.path.exists(dst):
                    base_name, ext = os.path.splitext(os.path.basename(src_file))
                    dst = os.path.join(dest_dir, f"{base_name}_{random.randint(0, 99999)}{ext}")
                
                # Copy or move the file
                if COPY_FILES:
                    shutil.copy2(src_file, dst)
                else:
                    shutil.move(src_file, dst)
                
                class_copied += 1
                total_copied += 1
                
                # Show progress for large classes
                if VERBOSE and file_idx % 100 == 0:
                    elapsed = time.time() - start_time
                    overall_progress = (total_copied / total_to_copy) * 100
                    print(f"    Processed {file_idx}/{split_count} images "
                          f"({overall_progress:.1f}% overall, {elapsed:.1f}s)")
            
            if not VERBOSE:
                print(" ✓")
        
        print(f"  Completed: {class_copied} images copied")
    
    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"[DONE] Successfully processed {total_copied}/{total_to_copy} images")
    print(f"Operation {'(COPY)' if COPY_FILES else '(MOVE)'} completed in {elapsed:.2f} seconds")
    
    # Verify the output
    verify_output(out_root)

def verify_output(out_root):
    """Verify that files were correctly distributed"""
    print("\n[VERIFICATION] Checking output structure...")
    
    total_files = 0
    class_counts = {}
    
    for split in ("train", "val", "test"):
        split_path = os.path.join(out_root, split)
        if not os.path.exists(split_path):
            print(f"[WARNING] Split directory not found: {split_path}")
            continue
            
        for cls in TARGET_CLASSES:
            cls_path = os.path.join(split_path, cls)
            if os.path.exists(cls_path):
                files = [f for f in os.listdir(cls_path) 
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                count = len(files)
                total_files += count
                class_counts[f"{split}/{cls}"] = count
                
                if VERBOSE and count > 0:
                    print(f"  {split}/{cls}: {count} images")
    
    print(f"\n[SUMMARY] Total images in output: {total_files}")
    
    # Show distribution by split
    for split in ("train", "val", "test"):
        split_total = sum(v for k, v in class_counts.items() if k.startswith(f"{split}/"))
        print(f"  {split}: {split_total} images")
    
    return total_files

def check_source_directory():
    """Check if source directory exists and has content"""
    print("[PRE-CHECK] Verifying source directory...")
    
    if not os.path.exists(RAW_ROOT):
        print(f"✗ Source directory does not exist: {RAW_ROOT}")
        print("Please check the RAW_ROOT path in the script.")
        return False
    
    dirs = [d for d in os.listdir(RAW_ROOT) if os.path.isdir(os.path.join(RAW_ROOT, d))]
    if not dirs:
        print(f"✗ No directories found in: {RAW_ROOT}")
        print("The directory might be empty or contain only files.")
        return False
    
    print(f"✓ Source directory exists and contains {len(dirs)} subdirectories")
    
    # Check for expected folders
    expected_folders = list(map_to_command.keys())
    found_expected = [d for d in dirs if d.lower() in expected_folders]
    print(f"  Found {len(found_expected)} out of {len(expected_folders)} expected gesture folders")
    
    if len(found_expected) < len(expected_folders):
        missing = set(expected_folders) - set([d.lower() for d in dirs])
        print(f"  Missing folders: {missing}")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("DATASET PREPARATION SCRIPT")
    print(f"Source: {RAW_ROOT}")
    print(f"Destination: {OUT_ROOT}")
    print(f"Mode: {'COPY' if COPY_FILES else 'MOVE'}")
    print("=" * 60)
    
    # Pre-check
    if not check_source_directory():
        print("\n[ERROR] Source directory check failed. Exiting.")
        exit(1)
    
    # Ask for confirmation
    response = input("\nProceed with dataset preparation? (y/n): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        exit(0)
    
    print("\nStarting dataset preparation...")
    
    # Create output directories
    print("[STEP 1] Creating directory structure...")
    ensure_dirs(OUT_ROOT)
    print("✓ Directory structure created")
    
    # Collect images
    print("\n[STEP 2] Collecting images from source...")
    entries = collect_images_per_class(RAW_ROOT)
    
    if not entries:
        print("[ERROR] No images found. Exiting.")
        exit(1)
    
    # Process and split
    print("\n[STEP 3] Splitting and copying files...")
    copy_split(entries, OUT_ROOT)
    
    print("\n" + "=" * 60)
    print(f"[COMPLETE] Dataset prepared successfully in: {OUT_ROOT}")
    print("=" * 60)