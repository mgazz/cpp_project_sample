////////////////////////////////////////////////////////////////////////
//
// hello-world.cpp
//
// This is a simple, introductory OpenCV program. The program reads an
// image from a file, inverts it, and displays the result. 
//
////////////////////////////////////////////////////////////////////////
#include "tensorflow/core/public/session.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <iostream>
//#include "tensorflow/cc/ops/const_op.h"
//#include "tensorflow/cc/ops/image_ops.h"
//#include "tensorflow/cc/ops/standard_ops.h"
//#include "tensorflow/core/framework/graph.pb.h"
//#include "tensorflow/core/framework/tensor.h"
//#include "tensorflow/core/graph/default_device.h"
//#include "tensorflow/core/graph/graph_def_builder.h"
//#include "tensorflow/core/lib/core/errors.h"
//#include "tensorflow/core/lib/core/stringpiece.h"
//#include "tensorflow/core/lib/core/threadpool.h"
//#include "tensorflow/core/lib/io/path.h"
//#include "tensorflow/core/lib/strings/stringprintf.h"
//#include "tensorflow/core/platform/init_main.h"
//#include "tensorflow/core/platform/logging.h"
//#include "tensorflow/core/platform/types.h"
//#include "tensorflow/core/util/command_line_flags.h"


int main(int argc, char *argv[])
{

  //if(argc<2){
    //printf("Usage: main <image-file-name>\n");
    //exit(0);
  //}

  // load an image  
  std::string input = "/home/mgazz/workspace/mgazz/cpp-tensorflow-inference/resources/data/lena.jpg";
  std::cout << input.c_str() << std::endl;
  cv::Mat image=cv::imread(input.c_str(),1);
  std::string graph = "tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb";
  std::string output_layer = "InceptionV3/Predictions/Reshape_1";
  std::unique_ptr<tensorflow::Session> session;

  if( image.empty()){
    std::cout <<  "Could not open or find the image" << std::endl ;
    return -1;
  }
  cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE ); // Create a window for display.
  cv::imshow( "Display window", image ); 
  cv::waitKey(0);


  return 0;

}
