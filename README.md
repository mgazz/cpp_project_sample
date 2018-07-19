```
#### Install Dependencies
```
./installBazel.sh

```

wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz

extract ssd_mobilenet_v1_coco_11_06_2017.tar.gz 

mkdir build

cd build

cmake ..

make -j$(nproc)

./bin/example
```
