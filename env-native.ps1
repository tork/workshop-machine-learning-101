$CONTAINER="ml101-env"
$EXISTING=docker inspect $CONTAINER | ConvertFrom-Json
if ($EXISTING.Length -eq 0) {
    $SRC_NAME=(Split-Path $PSScriptRoot -Leaf)
    $SHARE="${PSScriptRoot}:/$SRC_NAME" -replace '\\','/'
    $IDX=$SHARE.IndexOf(":")
    $SHARE="/" + $SHARE.Substring(0, $IDX).ToLower() + $SHARE.Substring($IDX + 1)
    docker run -it -p 8888:8888 -p 6006:6006 --entrypoint /bin/bash -v $SHARE --name $CONTAINER gcr.io/tensorflow/tensorflow:0.11.0rc0
} else {
    docker start -i $CONTAINER
}