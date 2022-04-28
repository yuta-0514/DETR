~#!/bin/bash

CONTAINER_NAME=detr
IMAGES=yuta0514/detr
TAGS=1.9
PORT=8888

docker run --rm -it --gpus all --ipc host -v $PWD:$PWD -v ~/dataset:/mnt -p ${PORT}:${PORT} --name ${CONTAINER_NAME} ${IMAGES}:${TAGS}

#run "umask 000" after this script
#docker makes file or dir by root, therefore make permission 777

