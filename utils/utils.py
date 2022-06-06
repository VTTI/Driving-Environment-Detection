import os
import random
import re
from ast import literal_eval

import pandas as pd
import torch
import yaml


def get_class_imbalance_weights(train_df):
    """Class imbalance mitigation"""
    count_classes = train_df.groupby('label').size().to_list()  # managing class imbalance
    weights_classes = torch.FloatTensor(
        [max(count_classes) / count_classes[0], max(count_classes) / count_classes[1],max(count_classes)/count_classes[2]])
    return weights_classes


def unique_name(path):
    name = re.findall("_([0-9a-z]+)_", path.split(os.sep)[-1])[0]  # unique vehicle number
    return name


def split(len_dataset, p_train=0.80, p_test=0.10, p_val=0.10):
    len_train = int(len_dataset * p_train)
    len_test = int(len_dataset * p_test)
    len_val = int(len_dataset * p_val)

    if len_dataset == len_train + len_test + len_val:
        return len_train, len_test, len_val
    else:
        difference = len_dataset - (len_train + len_test + len_val)
        return len_train, len_test + difference, len_val


def merge_and_split(res_df, urban_df,interstate_df, out_dir="output", save=True):
    """Merging both datasets"""
    all_data_df = pd.concat([res_df,urban_df,interstate_df], ignore_index=True, sort=False)
    unique_values = list(all_data_df['name'].unique())
    random.Random(42).shuffle(unique_values)
    train_split, test_split, val_split = split(len(unique_values))

    train_df = all_data_df.loc[all_data_df['name'].isin(unique_values[0:train_split])]
    test_df = all_data_df.loc[all_data_df['name'].isin(unique_values[train_split:train_split + test_split])]
    val_df = all_data_df.loc[all_data_df['name'].isin(unique_values[train_split + test_split:])]

    if save:
        path = os.path.join(out_dir, 'csv')
        os.makedirs(path, exist_ok=True)
        all_data_df.to_csv(os.path.join(path, 'all_data.csv'))
        train_df.to_csv(os.path.join(path, 'train.csv'))
        test_df.to_csv(os.path.join(path, 'test.csv'))
        val_df.to_csv(os.path.join(path, 'val.csv'))
        print("Saving csv files...")

    return all_data_df, train_df, test_df, val_df


def get_configs(path):
    with open(path) as out:
        configs = yaml.load(out, Loader=yaml.FullLoader)

    for key, value in configs.items():
        print(key, ": ", value)

    res_dir = configs["RES_DIR"]
    urban_dir = configs["URBAN_DIR"]
    interstate_dir = configs["INTERSTATE_DIR"]
    output_dir = configs["OUTPUT_DIR"]
    model_name = configs["MODEL"]
    backbone = configs["BACKBONE"]
    epochs = configs["EPOCHS"]
    lr = configs["LR"]
    custom = configs["CUSTOM_IMAGES_PATH"]
    resize_shape = literal_eval(configs["RESIZE_SHAPE"])
    optimizer = configs["OPTIMIZER"]
    batch_size = configs["BATCH_SIZE"]
    log_step = configs["LOG_STEP"]

    # create output directory
    os.makedirs(output_dir, exist_ok=True)

    return [res_dir, urban_dir,interstate_dir, output_dir,
            model_name, backbone, epochs, lr,
            resize_shape, optimizer, batch_size, log_step,
            custom]
