import os
import shutil


# Train

labels = ["Negative", "Positive"]
train_dir = "/workspace/DBT_US_Soroka/semi-supervised_data/labeled_data-384/Train"
new_dir = "/workspace/DBT_US_Soroka/semi-supervised_data/mt-git-data/train"

for l in labels:
    path = os.path.join(train_dir, l)
    files = os.listdir(path)
    for f in files:
        shutil.copy(os.path.join(path, f), os.path.join(new_dir, f))
print("all train data transferred")

# Val
val_dar = "/workspace/DBT_US_Soroka/semi-supervised_data/labeled_data-384/Val"
new_dir = "/workspace/DBT_US_Soroka/semi-supervised_data/mt-git-data/validation"
for l in labels:
    path = os.path.join(val_dar, l)
    files = os.listdir(path)
    for f in files:
        shutil.copy(os.path.join(path, f), os.path.join(new_dir, l, f))
print("all validation data transferred")
