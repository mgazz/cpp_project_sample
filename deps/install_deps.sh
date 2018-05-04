#!/bin/bash -       
#description     :
#author          :Michele Gazzetti
#date            :04.05.18
#notes           :
#==============================================================================


####
## Tensorflow build
####
#--config=monolitic is  required if you want to load images using Opencv
git clone https://github.com/tensorflow/tensorflow.git && cd tensorflow && bazel build --jobs 4 --config=monolithic tensorflow:libtensorflow_cc.so
#
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
#
#sudo cp bazel-bin/tensorflow/libtensorflow_framework.so /opt/tensorflow/lib/
#
#sudo apt-get remove libprotobuf-dev
#
#sudo apt-get remove protobuf-compiler
#
cd -

####
### Install protobuf 3
####
mkdir protobuf
#
cd protobuf/
#
wget https://github.com/google/protobuf/releases/download/v3.5.1/protobuf-cpp-3.5.1.zip
#
#extract protobuf-cpp-3.5.1.zip 
unzip protobuf-cpp-3.5.1.zip 
#
cd protobuf-3.5.1/
#
sudo apt-get install autoconf automake libtool curl make g++ unzip
#
./autogen.sh 
#
./configure --prefix=/opt/protobuf --includedir=/opt/protobuf/include/
#
make -j8
#
sudo make install
#
cd ../../

####
## Install eigen3
####
git clone https://github.com/eigenteam/eigen-git-mirror
#
cd eigen-git-mirror/
#
git checkout 3.3.4
#
mkdir build
#
cd build/
#
cmake -DCMAKE_INSTALL_PREFIX=/opt/eigen3 ..
#
make -j8
#
sudo make install
#
cd ../../

