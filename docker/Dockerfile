FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt update && \
    apt install --no-install-recommends -y python3.9 python3-pip python3-setuptools python3-distutils  libpython3.9-dev && \
    apt install ffmpeg libsm6 libxext6 wget curl -y && \
    apt clean && rm -rf /var/lib/apt/lists/*
RUN ln -svf /usr/bin/python3.9 /usr/bin/python

COPY ./docker/requirements.txt /requirements.txt
RUN python -m pip install --upgrade pip
RUN python -m pip install -r /requirements.txt

RUN mkdir /home/gerumo
RUN mkdir /home/gerumo/datasets

COPY . /home/gerumo/gerumo2

WORKDIR /home/gerumo/gerumo2

EXPOSE 8888
