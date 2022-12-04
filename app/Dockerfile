FROM nvidia/cuda:11.6.0-base-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get -y install vim htop libgl1-mesa-glx git libglib2.0-0 libsm6 libxrender1 libxext6
RUN apt -y install python3.9 python3-pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN mkdir /app
RUN mkdir -p /app/bak
COPY . /app
WORKDIR /app

RUN git clone https://github.com/open-mmlab/mmpose.git 
WORKDIR /app/mmpose
RUN pip3 install -e .
RUN pip3 install mmcv
WORKDIR /app
RUN cp custom/coco_msc_tray.py mmpose/configs/_base_/datasets/coco_msc_tray.py
RUN cp custom/msc_tray_bottom_up_coco.py mmpose/mmpose/datasets/datasets/bottom_up/msc_tray_bottom_up_coco.py
RUN cp custom/__init__.py mmpose/mmpose/datasets/datasets/bottom_up/__init__.py

EXPOSE  8000
