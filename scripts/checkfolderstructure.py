# check_dataset.py
import os
import sys

def check_dataset_structure(data_path):
    print(f"Checking dataset structure at: {data_path}")
    print("=" * 60)
    
    if not os.path.exists(data_path):
        print(f"❌ ERROR: Directory does not exist: {data_path}")
        return False
    
    required_splits = ['train', 'val']
    optional_splits = ['test']
    found_classes = set()
    
    for split in required_splits:
        split_path = os.path.join(data_path, split)
        if not os.path.exists(split_path):
            print(f"❌ ERROR: Missing required split: {split}")
            return False
        
        print(f"\n✅ Found split: {split}")
        classes = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        
        if not classes:
            print(f"❌ ERROR: No class folders found in {split}")
            return False
        
        if split == 'train':
            found_classes.update(classes)
        
        # Count images in each class
        for cls in sorted(classes):
            cls_path = os.path.join(split_path, cls)
            images = [f for f in os.listdir(cls_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"   {cls}: {len(images)} images")
    
    print("\n" + "=" * 60)
    print("✅ Dataset structure is correct!")
    print(f"Total classes: {len(found_classes)}")
    print(f"Classes: {sorted(found_classes)}")
    print("\nNow you can run:")
    print(f'yolo classify train data="{data_path}" model=yolo11n-cls.pt epochs=30 imgsz=224 batch=64 project=runs/classify name=hagrid_run device=0 workers=4')
    return True

if __name__ == "__main__":
    # Try common paths
    possible_paths = [
        r"N:\ROV\data",  # Your data path
        "data",           # Relative path
        os.path.join(os.getcwd(), "data")  # Absolute from current dir
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            if check_dataset_structure(path):
                sys.exit(0)
    
    print("❌ Could not find data directory. Checked:")
    for path in possible_paths:
        print(f"  - {path}")
    
    # Ask user for path
    user_path = input("\nEnter the full path to your data directory: ").strip()
    if user_path:
        check_dataset_structure(user_path)


# import os

# base = r"N://ROV//data//train"

# for cls in os.listdir(base):
#     path = os.path.join(base, cls)
#     print(cls, "=>", len(os.listdir(path)))
