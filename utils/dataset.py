import glob
import os
import re

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.utils import unique_name, merge_and_split, get_class_imbalance_weights

'''
Feature       | Labels
----------------------
Interstate       |  2
Urban            |  1
Residential      |  0
'''


def get_df(path, zone):
    data = list()
    path = os.path.join(path, '**', '*.jpg')

    for filepath in glob.iglob(path, recursive=True):
        temp = list()
        temp.append(filepath)  # path
        name = unique_name(filepath)

        temp.append(name)
        temp.append(zone)  # Residential / Urban / Interstate
        data.append(temp)
    df = pd.DataFrame(data, columns=['path', 'name', 'label'])
    return df


def get_transform(resize_shape, jitter=0.10, split="val"):
    if split == "train":
        transform = transforms.Compose([transforms.Resize((resize_shape[0], resize_shape[1])),
                                        transforms.ColorJitter(saturation=jitter, hue=jitter,
                                                               contrast=jitter, brightness=jitter), 
                                        transforms.RandomRotation(degrees=(-60,60)),
                                        # transforms.RandomAdjustSharpness(sharpness_factor=2),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
                                        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
                                        ])
        return transform
    elif split == "val" or split == "test":
        transform = transforms.Compose([transforms.Resize((resize_shape[0], resize_shape[1])),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
                                        ])
        return transform


class DatasetBaseline(Dataset):

    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image = Image.open(self.df.iloc[item, 0])
        w, h = image.size
        image = image.crop((0, 30, w, h))  # cropping watermark
        image = self.transform(image)
        label = self.df.iloc[item, 2]
        data = {"image": image, "label": label}
        return data


def dataset_baseline(path_res, path_urban, path_interstate, out_dir="output", resize_shape=(240, 360),
                     jitter=0.05, dataset=DatasetBaseline, save=True):
    """****************** Residential ******************"""
    res_df = get_df(path_res, zone=0)

    """****************** Urban ******************"""
    urban_df = get_df(path_urban, zone=1) 
    
    """****************** Interstate ******************"""
    interstate_df = get_df(path_interstate, zone=2)


    all_data_df, train_df, test_df, val_df = merge_and_split(res_df=res_df, urban_df=urban_df,
                                                             interstate_df=interstate_df,out_dir=out_dir, save=save)
    weights_int = get_class_imbalance_weights(train_df)

    # sanity check
    # train_test_df = test_df["name"].isin(train_df["name"])
    # train_test_df.to_csv("train_test_df.csv")
    # train_val_df = val_df["name"].isin(train_df["name"])
    # train_val_df.to_csv("train_val_df.csv")

    """Creating dataset"""
    train_set = dataset(df=train_df, transform=get_transform(resize_shape, jitter, split="train"))
    test_set = dataset(df=test_df, transform=get_transform(resize_shape, split="test"))
    val_set = dataset(df=val_df, transform=get_transform(resize_shape, split="val"))

    return train_set, test_set, val_set, weights_int


if __name__ == "__main__":
    pass
