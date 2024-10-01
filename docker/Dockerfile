FROM zombbie/cuda11.1-cudnn8-ubuntu20.04:v1.0
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install -y python3.8 python3-pip
RUN apt-get -y install git

RUN pip --default-timeout=2000 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN pip install cupy-cuda111

RUN apt-get update
RUN apt-get -y install qt5-default python3-pyqt5
RUN apt-get -y install net-tools

RUN mkdir -p /home/data
RUN mkdir -p /home/ws
ADD ../requirements.txt /home/requirements.txt
ADD ../.tmux.conf /home/.tmux.conf
RUN apt-get -y install tmux

WORKDIR /home/ws/TESNet
RUN pip install -r ../../requirements.txt

RUN apt-get install htop
RUN apt-get install -y xclip
RUN apt-get install -y zip

ARG UID
ARG GID

# Update the package list, install sudo, create a non-root user, and grant password-less sudo permissions
RUN apt update && \
    apt install -y sudo && \
    addgroup --gid $GID nonroot && \
    adduser --uid $UID --gid $GID --disabled-password --gecos "" nonroot && \
    echo 'nonroot ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers

# Set the non-root user as the default user
USER nonroot
RUN echo "alias tn='. run_tmux.sh'" >> ~/.bashrc
