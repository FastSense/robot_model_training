#!/bin/bash

image_name=pytorch-image

docker build . -t $image_name
