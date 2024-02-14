import pandas as pd
import random

#labels_p = [0.1, 0.25, 0.5, 0.75]
labels_p = [0.01]

def get_unique(f_list):
    uniqe_filename = []
    for name in f_list:
        if name[:8] not in uniqe_filename:
            uniqe_filename.append(name[:8])
    return uniqe_filename

def get_images_names(all, subjects):
    images = []
    for name in subjects:
        res = all.str.contains(pat=name)
        relevant_files = all[res.values]
        for f in relevant_files:
            images.append(f)
    return images


df = pd.read_csv("/workspace/DBT_US_Soroka/semi-supervised_data/mt-git-data/labels.txt", sep=" ")
all_pos = df[df.iloc[:, 1] == 'Positive'].iloc[:, 0]
all_neg = df[df.iloc[:, 1] == 'Negative'].iloc[:, 0]


unique_pos = get_unique(all_pos)
unique_neg = get_unique(all_neg)
num_pos = len(unique_pos)
num_neg = len(unique_neg)

for p in labels_p:
    random.shuffle(unique_pos)
    random.shuffle(unique_neg)

    current_pos = unique_pos[:int(round(p * num_pos))]
    current_neg = unique_neg[:int(round(p * num_neg))]

    # return from subjects to image names and write to txt
    images_pos = get_images_names(all_pos, current_pos)
    images_neg = get_images_names(all_neg, current_neg)

    images = images_neg + images_pos
    labels = ['Negative']* len(images_neg) + ['Positive']* len(images_pos)

    print('---------------------------------- '+str(int(p*100))+'% ----------------------------------')
    print(str(len(images_neg)) + ' Negative Images')
    print(str(len(images_pos)) + ' Positive Images')
    data = list(zip(images, labels))
    df = pd.DataFrame(data)
    df.columns = [None] * df.shape[1]
    df.to_csv('/workspace/DBT_US_Soroka/semi-supervised_data/mt-git-data/labels-'+str(int(p*100))+'.txt', sep=' ', index=False, header=False)
