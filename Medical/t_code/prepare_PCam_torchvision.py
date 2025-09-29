import pandas as pd
import random
import os
import shutil

# load Ground Truth
df = pd.read_csv("./MedImages/histopathologic-cancer-detection/train_labels.csv")

# split train / test
random.seed(123)
trainset = random.sample(list(range(len(df))), int(0.7 * len(df)))

source_dirpath = './MedImages/histopathologic-cancer-detection/train' # source directory

for index, row in df.iterrows():
    # get file name
    filename = row['id'] + '.tif'
    # get label
    label = ['normal', 'tumor'][row['label']]
    # train or test
    datasettype = ['test', 'train'][index in trainset]

    target_dirpath = './MedImages/PCam_ImageFolder/' + datasettype + '/' + label
    print(os.path.join(source_dirpath, filename), "to", os.path.join(target_dirpath, filename))
    
    # file copy
    shutil.copy2(os.path.join(source_dirpath, filename), os.path.join(target_dirpath, filename))
