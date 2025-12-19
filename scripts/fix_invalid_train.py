import os
import random

folder = r"../data/train/Invalid"
max_images = 6000

files = os.listdir(folder)

print("Before:", len(files))

if len(files) > max_images:
    delete_files = random.sample(files, len(files) - max_images)

    for f in delete_files:
        os.remove(os.path.join(folder, f))

print("After:", len(os.listdir(folder)))
print("✅ Invalid class balanced")
