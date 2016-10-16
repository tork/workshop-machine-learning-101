#!/usr/bin/env bash
SRC_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_NAME="${PWD##*/}"
docker run -itp 8888:8888 --entrypoint /bin/bash -v $SRC_PATH:/$SRC_NAME gcr.io/tensorflow/tensorflow:0.11.0rc0
