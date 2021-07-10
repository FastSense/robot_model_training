FROM nvidia/cuda:10.2-base-ubuntu18.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    wget
# && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Install Python3.8
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.8 \
    python3.8-dev \
    python3.8-distutils

# Utils 
RUN apt-get install -y \
        tmux \
        ranger \
        wget \
        neovim && \
        rm -rf /var/lib/apt/lists/*    

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/zsh user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
ENV PATH="/home/user/.local/bin:${PATH}"
RUN chmod 777 /home/user

# Install Pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py

# Python packages
RUN pip install matplotlib \
    pandas \
    scipy \
    jupyterlab \
    sklearn \
    opencv-python \
    tqdm

# Torch
RUN pip install torch==1.6.0



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
