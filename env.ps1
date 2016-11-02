$SRC_NAME=(Split-Path $PSScriptRoot -Leaf)
docker run -it -p 8888:8888 -p 6006:6006 --entrypoint /bin/bash -v $PSScriptRoot:/$SRC_NAME gcr.io/tensorflow/tensorflow:0.11.0rc0
