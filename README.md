#### Install Dependencies
** To be checked... **
```
./deps/install_bazel_tx2.sh
./deps/build_tensorflow.sh

```

#### Compile project
```
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz

extract ssd_mobilenet_v1_coco_11_06_2017.tar.gz 

mkdir build

cd build

cmake ..

make -j$(nproc)

./bin/example ../ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb image_tensor:0 299 ../resources/data/test4.jpg
```

**Result will of 50 inference iterations will be logged in:** ```./latency_results_in_ms.csv ```
