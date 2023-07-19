FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN pip install --upgrade pip
RUN pip install --no-cache-dir matplotlib wandb