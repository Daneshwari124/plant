import os, shutil, random
from pathlib import Path

# Source: folder where you unzipped the Kaggle dataset
SRC = Path("plantvillage")   # adjust if your folder has a different name
DST = Path("dataset")        # output folder with train/test
TEST_RATIO = 0.15            # 15% of images go to test set

for cls in SRC.iterdir():
    if not cls.is_dir():
        continue
    images = list(cls.glob("*.*"))
    random.shuffle(images)
    cut = int(len(images) * (1 - TEST_RATIO))

    # make folders
    train_dir = DST / "train" / cls.name
    test_dir = DST / "test" / cls.name
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # copy images
    for p in images[:cut]:
        shutil.copy(p, train_dir / p.name)
    for p in images[cut:]:
        shutil.copy(p, test_dir / p.name)

print("✅ Dataset split into train/ and test/ inside 'dataset/' folder")
