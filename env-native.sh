#!/usr/bin/env bash
SRC_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_NAME="${PWD##*/}"
docker run -it -p 8888:8888 -p 6006:6006 --entrypoint /bin/bash -v $SRC_PATH:/$SRC_NAME gcr.io/tensorflow/tensorflow:0.11.0rc0
