# Build an image that can do training and inference with Geneformer in SageMaker

FROM ubuntu:24.04

MAINTAINER Lan Guo <languo@amazon.com>

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update

RUN apt-get install -y build-essential python3.12 python3-pip python3.12-venv

# install cuda and nvidia driver
RUN add-apt-repository ppa:graphics-drivers/ppa
RUN apt-get update
RUN apt install -y nvidia-cuda-toolkit
RUN apt install -y nvidia-driver-560

# make virtual env
RUN python3 -m venv geneformer
# RUN source ~/env/geneformer/bin/activate
# ENV PATH="~/env/geneformer/bin:$PATH"
ENV VIRTUAL_ENV /geneformer                     # activating environment
ENV PATH /geneformer/bin:$PATH                 # activating environment

# Down load and install geneformer and dependencies
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
RUN apt-get install -y git-lfs
RUN git lfs install
RUN git clone https://huggingface.co/ctheodoris/Geneformer && cd Geneformer && pip install . && pip install accelerate
RUN pip install --disable-pip-version-check -q -U 'boto3==1.35.16' 'sagemaker==2.231.0' 'mlflow==2.13.2' 'sagemaker-mlflow==0.1.0' 'setuptools' 'igraph' 'leidenalg' 'sagemaker-training'