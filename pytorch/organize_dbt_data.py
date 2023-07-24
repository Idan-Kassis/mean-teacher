import os
import shutil

labeled = True
unlabeled = True

if labeled:
    # Train - labeled
    labels = ["Negative", "Positive"]
    train_dir = "/workspace/DBT_US_Soroka/semi-supervised_data/labeled_data-384/Train"
    new_dir = "/workspace/DBT_US_Soroka/semi-supervised_data/mt-git-data/train"

    for l in labels:
        path = os.path.join(train_dir, l)
        files = os.listdir(path)
        for f in files:
            shutil.copy(os.path.join(path, f), os.path.join(new_dir, l, f))
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


if unlabeled:
    path = "/workspace/DBT_US_Soroka/semi-supervised_data/unlabeled_data-384"
    new_dir = "/workspace/DBT_US_Soroka/semi-supervised_data/mt-git-data/train"
    files = os.listdir(path)
    pos_labeled = os.listdir("/workspace/DBT_US_Soroka/semi-supervised_data/labeled_data-384/Train/Positive")
    neg_labeled = os.listdir("/workspace/DBT_US_Soroka/semi-supervised_data/labeled_data-384/Train/Negative")
    for f in files:
        if any(f[:8] in item for item in pos_labeled):
            l = "Positive"
        elif any(f[:8] in item for item in neg_labeled):
            l = "Negative"
        shutil.copy(os.path.join(path, f), os.path.join(new_dir, l, f))


