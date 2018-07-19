#!/bin/bash
# NVIDIA Jetson TX2
# TensorFlow Installation
# Install Bazel
# Version 0.10.0
# We use the release distribution so that we don't have to build protobuf
#

# From: https://github.com/jetsonhacks/installTensorFlowTX2/blob/master/scripts/installBazel.sh

sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update && sudo apt-get install  build-essential python zip

### Super hack to solve 404 error !!!!!
sudo apt-get install oracle-java8-installer

cd /var/lib/dpkg/info
sudo sed -i 's|JAVA_VERSION=8u171|JAVA_VERSION=8u181|' oracle-java8-installer.*
sudo sed -i 's|PARTNER_URL=http://download.oracle.com/otn-pub/java/jdk/8u171-b11/512cd62ec5174c3487ac17c61aaa89e8/|PARTNER_URL=http://download.oracle.com/otn-pub/java/jdk/8u181-b13/96a7b8442fe848ef90c96a2fad6ed6d1/|' oracle-java8-installer.*
sudo sed -i 's|SHA256SUM_TGZ="b6dd2837efaaec4109b36cfbb94a774db100029f98b0d78be68c27bec0275982"|SHA256SUM_TGZ="1845567095bfbfebd42ed0d09397939796d05456290fb20a83c476ba09f991d3"|' oracle-java8-installer.*
sudo sed -i 's|J_DIR=jdk1.8.0_171|J_DIR=jdk1.8.0_181|' oracle-java8-installer.*

sudo apt-get update

sudo apt-get install oracle-java8-installer

cd -
### End Super hack!!!!!



bazel_version="0.11.1"

./compile.sh

wget --no-check-certificate https://github.com/bazelbuild/bazel/releases/download/$bazel_version/bazel-$bazel_version-dist.zip

unzip bazel-$bazel_version-dist.zip -d bazel-$bazel_version-dist

cd bazel-$bazel_version-dist

./compile.sh 

sudo cp output/bazel /usr/local/bin
