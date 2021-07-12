#!/bin/bash

image_name=pytorch-image
container_name=pytorch-container

docker run -it -d --privileged --net=host \
      --name $container_name \
      --runtime=nvidia \
      --gpus=all \
      -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
      -v ${PWD}/../:/home/user/ws:rw \
      -e DISPLAY=$DISPLAY \
      -e NVIDIA_VISIBLE_DEVICES="all" \
      -e NVIDIA_DRIVER_CAPABILITIES="all" \
      -e QT_X11_NO_MITSHM=1 $image_name zsh 
