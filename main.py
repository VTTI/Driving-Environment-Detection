import argparse
import os
import random

import numpy as np
import torch

from utils.baseline import RunBaseline
from utils.utils import get_configs


def set_seed(seed, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(device=device)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-f", help="path to config file", default="./configs/config_baseline.yaml")
    args.add_argument("--mode", "-m", help="mode to run the baseline model on",
                      choices=["train", "test", "test_single"], default="train")
    args.add_argument("--comment", "-c", help="comment for training", default="_")
    args.add_argument("--weight", "-w", help="path to model weights", default=None)
    args.add_argument("--device", "-d", help="set device number if you have multiple GPUs", default=0, type=int)
    args = args.parse_args()
    return args.config, args.comment, args.mode, args.weight, args.device


def load_model(path, comment, mode):
    configs = get_configs(path)
    ## TODO : Update config outputs Data split into three directpris one for each class
    residential = configs[0]
    urban = configs[1]
    interstate = configs[2]
    output_dir = configs[3]
    model_name = configs[4]
    backbone = configs[5]
    epochs = configs[6]
    lr = configs[7]
    resize_shape = configs[8]
    optimizer = configs[9]
    batch_size = configs[10]
    log_step = configs[11]
    custom_image_path = configs[12]

    baseline = RunBaseline(comment=comment,
                           res_dir=residential,
                           urban_dir=urban,
                           interstate_dir = interstate,
                           model_name=model_name,
                           optimizer=optimizer,
                           num_epochs=epochs,
                           batch_size=batch_size,
                           log_step=log_step,
                           out_dir=output_dir,
                           lr=lr,
                           resize_shape=resize_shape,
                           mode=mode,
                           custom_image_path=custom_image_path)

    return baseline


def main():
    config, comment, mode, weight, device = parse_args()
    set_seed(42, device)
    net = load_model(config, comment, mode)
    if mode == "train":
        net.train()
        print("Testing")
        net.test()
    elif mode == "test":
        # make sure that the weights are present in the output folder
        print("Testing")
        net.test(weight=weight)
    elif mode == "test_single":
        net.test_on_single_images(weight=weight)


if __name__ == "__main__":
    main()
