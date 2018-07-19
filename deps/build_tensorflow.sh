#!/bin/bash

#git clone https://github.com/tensorflow/tensorflow.git

#cd tensorflow

#git checkout v1.8.0

export TF_NEED_CUDA=1
export TF_CUDA_VERSION=9.0
export CUDA_TOOLKIT_PATH=/usr/local/cuda
export TF_CUDNN_VERSION=7.0.5
export CUDNN_INSTALL_PATH=/usr/lib/aarch64-linux-gnu/
export TF_CUDA_COMPUTE_CAPABILITIES=6.2

#bazel build --jobs 4 --config=opt --config=cuda tensorflow:libtensorflow_cc.so

sudo mkdir -p /opt/tensorflow/{lib,include}
#
sudo cp -r tensorflow /opt/tensorflow/include/
#
sudo find /opt/tensorflow/include/ -type f ! -name "*.h" -delete
#
sudo cp bazel-genfiles/tensorflow/core/framework/*.h /opt/tensorflow/include/tensorflow/core/framework/
#
sudo cp bazel-genfiles/tensorflow/core/lib/core/*.h /opt/tensorflow/include/tensorflow/core/lib/core/
#
sudo cp bazel-genfiles/tensorflow/core/protobuf/*.h /opt/tensorflow/include/tensorflow/core/protobuf/
#
sudo cp bazel-genfiles/tensorflow/core/util/*.h /opt/tensorflow/include/tensorflow/core/util/
#
sudo cp bazel-genfiles/tensorflow/cc/ops/*.h /opt/tensorflow/include/tensorflow/cc/ops/
# not sure if required
sudo cp -r third_party/ /opt/tensorflow/include/
#
#sudo apt install libeigen3-dev
#
sudo rm -r /opt/tensorflow/include/third_party/py
#
sudo cp bazel-bin/tensorflow/libtensorflow_cc.so /opt/tensorflow/lib/

