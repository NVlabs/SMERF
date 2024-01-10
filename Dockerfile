# a docker container from https://ngc.nvidia.com/catalog/containers/ with a pytorch installed
FROM nvcr.io/nvidia/pytorch:21.03-py3  

ENV CUDA_HOME "/usr/local/cuda"
ENV FORCE_CUDA "1"

# specifying which python version
ARG PYTHON_VERSION=3.8.15

# installing other nice functionalities
RUN apt-get update
RUN apt-get install libgl1 -y

# installing torch packages
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install opencv-python==4.5.5.64

# install MMCV-series
# RUN pip install -U openmim
# RUN mim install "mmcv-full==1.5.2"
# RUN mim install "mmdet==2.26.0"
# RUN pip install "mmsegmentation==0.29.1"

RUN pip install mmcv-full==1.5.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
RUN pip install mmdet==2.26.0
RUN pip install mmsegmentation==0.29.1
RUN pip install timm

# copy all the files to the container
COPY . /workspace/OpenLane-V2

# install mmdetection3d 1.0.0rc6
WORKDIR /workspace/OpenLane-V2
RUN pip install -r requirements.txt
RUN pip install -v -e .

WORKDIR /workspace/OpenLane-V2/mmdetection3d
RUN pip install -v -e .

# install pip requirements
WORKDIR /workspace/OpenLane-V2
# RUN pip install jupyter-core==4.7.1 jupyter-client==6.1.12 jupyterlab==2.2.9 traitlets==5.0.5
# RUN pip install --upgrade jupyter_core jupyter_client
RUN pip install jupyter_core jupyter_client
RUN pip install --upgrade "protobuf<=3.20.1"
RUN pip install wandb

# install packages for OpenStreetMap
RUN pip install osmnx==1.5.1
RUN pip install av2==0.2.1
RUN pip install geocube==0.3.3
RUN pip install sparse==0.14.0

ENV SHELL /bin/bash
