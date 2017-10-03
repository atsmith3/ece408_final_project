#! /bin/bash

# Cause the script to fail on errors
set -eou pipefail -x

SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")
echo $SCRIPTPATH

# Where the sources for the new layer are
SRC_DIR="$SCRIPTPATH/ece408_src"
SRCS=`find $SRC_DIR -type f`

# Where MXNet source lives. MXNET_SRC_ROOT is defined in the rai execution environment, if you are developing locally you can define it yourself, or enter it below
if [ -z ${MXNET_SRC_ROOT+x} ]; then
    MXNET_SRC_ROOT="$HOME/repos/incubator-mxnet"
fi

# Copy our files to the custom operator directory
for src in $SRCS; do
    cp -v "$src" "$MXNET_SRC_ROOT/src/operator/custom/."
done

# Build MXNet
nice -n20 \
    make -j`nproc` \
        -C "$MXNET_SRC_ROOT" \
        USE_CUDA=1 \
        USE_CUDA_PATH=/usr/local/cuda \
        USE_OPENCV=1 \
        USE_CUDNN=1 \
        USE_BLAS=openblas

# Install python bindings
pip install --user -e "$MXNET_SRC_ROOT/python"