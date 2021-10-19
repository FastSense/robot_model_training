FROM nvidia/cuda:11.2.0-base-ubuntu20.04


# >>>>>>>>>>>> Utils 
RUN apt-get update && \
    apt-get install -y \
        curl \
        ca-certificates \
        sudo \
        git \
        bzip2 \
        libx11-6 \
        tmux \
        ranger \
        wget \
        neovim \
        nano \
        htop \
        iputils-ping && \
        rm -rf /var/lib/apt/lists/*
# <<<<<<<<<<<< Utils


# >>>>>>>>>>>> Python packages
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

RUN apt-get update && \
    apt-get install -y \
       python3-tk \
       python3-pip && \
       rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip && \
    pip3 install \
       torch==1.9.0

RUN pip3 install --upgrade pip && \
    pip3 install \
        tensorboard \
        pandas \
        scipy \
        jupyterlab \
        sklearn \
        opencv-python \
        tqdm \
        numpy \
        pandas \
        numba \
        matplotlib \
        wandb \
        seaborn \
        tqdm \
        pyyaml \
        nnio \
        onnxruntime \
        wandb \
        setuptools
# <<<<<<<<<<<<< Python packages


# >>>>>>>>>>>> Create User
ENV ROSUSER="user"

RUN adduser --home /home/${ROSUSER} --shell /bin/zsh --gecos '' --disabled-password ${ROSUSER} && \
    usermod -a -G root,sudo,video ${ROSUSER} && \
    echo "${ROSUSER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${ROSUSER}
user $ROSUSER

# All users can use /home/user as their home directory
ENV HOME=/home/user
ENV PATH="/home/user/.local/bin:${PATH}"
RUN chmod 777 /home/user
# <<<<<<<<<<<< Create User


# Scripts
RUN mkdir -p /home/user/scripts
WORKDIR /home/user/scripts

COPY scripts/utils-setup.sh .
RUN ./utils-setup.sh        

WORKDIR /home/user
RUN rm -rf scripts


EXPOSE 8888

# Set the default command
#ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]
