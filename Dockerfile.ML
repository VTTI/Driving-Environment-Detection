### Dockerfile to build docker image for driving environment project ###
# Tensorflow image as a base
# This image contains both tf and torch dependencies
# We also mark this image as "ml-base-tf-torch" so we could use it by name
FROM tensorflow/tensorflow:latest-gpu AS ml-base-tf-torch

## These lines are to account for Nvidia is rotating public keys ##
## Last checked on 06/05/2022
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 21
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install Python and its tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools
RUN pip3 -q install pip --upgrade
# Install all basic packages
RUN pip3 install \
    # Jupyter itself
    jupyter \
    # Numpy and Pandas are required a-priori
    numpy pandas \
    # Pytorch and torchvision 
    torch torchvision\
    # Upgraded version of Tensorboard with more features
    tensorboardX

ENV CUDA_VISIBLE_DEVICES 0,1,2,3
RUN export CUDA_VISIBLE_DEVICES=0,1,2,3
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get install vim -y
RUN apt-get install p7zip

# Install additional packages
RUN pip3 install opencv-python Shapely einops matplotlib scipy tqdm scikit_learn tensorboard pillow jupyterlab PyYAML seaborn segmentation-models-pytorch pytorch-gradcam

WORKDIR /opt/app
RUN mkdir  /.cache
RUN chmod -R 777 /.cache
RUN mkdir /.local
RUN chmod -R 777 /.local
## -------------
## Add alias to ensure calling python calls python3
## -------------
RUN echo 'alias python="python3"' >> ~/.bashrc
RUN echo 'alias jupyter="jupyter lab --port=8888 --no-browser --ip=0.0.0.0 --allow-root"' >> ~/.bashrc
ENTRYPOINT /bin/bash  && cd /opt/app
