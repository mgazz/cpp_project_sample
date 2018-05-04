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
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"
#include <fstream>
#include <utility>
#include <vector>
#include <iostream>

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;


// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
Status ReadLabelsFile(const string& file_name, std::vector<string>* result,
                      size_t* found_label_count) {
    std::ifstream file(file_name);
    if (!file) {
        return tensorflow::errors::NotFound("Labels file ", file_name,
                                            " not found.");
    }
    result->clear();
    string line;
    while (std::getline(file, line)) {
        result->push_back(line);
    }
    *found_label_count = result->size();
    const int padding = 16;
    while (result->size() % padding) {
        result->emplace_back();
    }
    return Status::OK();
}

static Status ReadEntireFile(tensorflow::Env* env, const string& filename,
                             Tensor* output) {
    tensorflow::uint64 file_size = 0;
    TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

    string contents;
    contents.resize(file_size);

    std::unique_ptr<tensorflow::RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

    tensorflow::StringPiece data;
    TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
    if (data.size() != file_size) {
        return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                            "' expected ", file_size, " got ",
                                            data.size());
    }
    output->scalar<string>()() = data.ToString();
    return Status::OK();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors) {
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    string input_name = "file_reader";
    string output_name = "normalized";

    // read file_name into a tensor named input
    Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
    TF_RETURN_IF_ERROR(
            ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

    // use a placeholder to read input data
    auto file_reader =
            Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
            {"input", input},
    };

    // Now try to figure out what kind of file it is and decode it.
    const int wanted_channels = 3;
    tensorflow::Output image_reader;
    //if (tensorflow::StringPiece(file_name).ends_with(".png")) {
        //image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                                 //DecodePng::Channels(wanted_channels));
    //} else if (tensorflow::StringPiece(file_name).ends_with(".gif")) {
        //// gif decoder returns 4-D tensor, remove the first dim
        //image_reader =
                //Squeeze(root.WithOpName("squeeze_first_dim"),
                        //DecodeGif(root.WithOpName("gif_reader"), file_reader));
    //} else {
        //// Assume if it's neither a PNG nor a GIF then it must be a JPEG.
        //image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                                  //DecodeJpeg::Channels(wanted_channels));
    //}
    image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                              DecodeJpeg::Channels(wanted_channels));
    // Now cast the image data to float so we can do normal math on it.
    // auto float_caster =
    //     Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);

    auto uint8_caster =  Cast(root.WithOpName("uint8_caster"), image_reader, tensorflow::DT_UINT8);

    // The convention for image ops in TensorFlow is that all images are expected
    // to be in batches, so that they're four-dimensional arrays with indices of
    // [batch, height, width, channel]. Because we only have a single image, we
    // have to add a batch dimension of 1 to the start with ExpandDims().
    auto dims_expander = ExpandDims(root.WithOpName("dim"), uint8_caster, 0);

    // Bilinearly resize the image to fit the required dimensions.
    // auto resized = ResizeBilinear(
    //     root, dims_expander,
    //     Const(root.WithOpName("size"), {input_height, input_width}));


    // Subtract the mean and divide by the scale.
    // auto div =  Div(root.WithOpName(output_name), Sub(root, dims_expander, {input_mean}),
    //     {input_std});


    //cast to int
    //auto uint8_caster =  Cast(root.WithOpName("uint8_caster"), div, tensorflow::DT_UINT8);

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output tensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(
            tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {"dim"}, {}, out_tensors));
    return Status::OK();
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
            ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                            graph_file_name, "'");
    }
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return Status::OK();
}
// inspired from: https://github.com/jhjin/tensorflow-cpp
// inspired from: https://github.com/memo/ofxMSATensorFlow/issues/34
//
int main(int argc, char *argv[])
{

  //if(argc<2){
    //printf("Usage: main <image-file-name>\n");
    //exit(0);
  //}

  // load an image  
  std::string input = "/home/mgazz/workspace/mgazz/cpp-tensorflow-inference/resources/data/test4.jpg";
  std::cout << input.c_str() << std::endl;
  cv::Mat image=cv::imread(input.c_str(),1);

  if( image.empty()){
    std::cout <<  "Could not open or find the image" << std::endl ;
    return -1;
  }
  //cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE ); // Create a window for display.
  //cv::imshow( "Display window", image ); 
  //cv::waitKey(0);
  

  //#########################################################
  //string graph ="ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb";
      // creating a Tensor for storing the data
    string labels ="mscoco_label_map.pbtxt";
    int32 input_width = 299;
    int32 input_height = 299;
    int32 depth =3;
    float input_mean = 0;
    float input_std = 255;
    string input_layer = "image_tensor:0";
    std::vector<string> output_layer ={ "detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0" };

    bool self_test = false;
    string root_dir = "";

    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,input_height,input_width,depth}));
    auto input_tensor_mapped = input_tensor.tensor<float, 4>();
    cv::Mat img_stream;
    const float * source_data;
    // copying the data into the corresponding tensor
    //cv::namedWindow("camera");
    cv::Size s(input_height,input_width);
    cv::Mat resized_img;
    cv::Mat resized_i;
    std::unique_ptr<tensorflow::Session> session;
    string graph_path ="../ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb";
    std::vector<Tensor> resized_tensors;
    std::string image_path= input;
    Status read_tensor_status;
    std::vector<Tensor> outputs;
    int image_width;
    int image_height;

    // First we load and initialize the model.
    LoadGraph(graph_path, &session);
    ReadTensorFromImageFile(image_path, input_height, input_width, input_mean,
                            input_std, &resized_tensors);
    //============================================//
    const Tensor& resized_tensor = resized_tensors[0];
    // << ",data:" << resized_tensor.flat<tensorflow::uint8>();
    // Actually run the image through the model.

    session->Run({{input_layer, resized_tensor}},
                                     output_layer, {}, &outputs);
    image_width = resized_tensor.dims();
    image_height = 0;
    //int image_height = resized_tensor.shape()[1];
    //tensorflow::TTypes<float>::Flat iNum = outputs[0].flat<float>();
    tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
    tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
    tensorflow::TTypes<float>::Flat num_detections = outputs[3].flat<float>();
    auto boxes = outputs[0].flat_outer_dims<float,3>();
//        LOG(ERROR) << "num_detections:" << num_detections(0) << "," << outputs[0].shape().DebugString();
//
    cv::Size image_size = image.size();

    for(size_t i = 0; i < num_detections(0) && i < 20;++i)
    {
        if(scores(i) > 0.4)
        {
            std::cout<< i << ",score:" << scores(i) << ",class:" << classes(i)<< ",box:" << "," << boxes(0,i,0) << "," << boxes(0,i,1) << "," << boxes(0,i,2)<< "," << boxes(0,i,3);
            cv::Point p1 = cv::Point(boxes(0,i,1)*image_size.width,
                                     boxes(0,i,0)*image_size.height);
            cv::Point p2 = cv::Point(
                                     boxes(0,i,3)*image_size.width,
                                     boxes(0,i,2)*image_size.height);
            //cv::Rect r =cv::Rect(boxes(0,i,0)*image_size.width,
                      //boxes(0,i,1)*image_size.height,
                      //(boxes(0,i,2)-boxes(0,i,0))*image_size.width,
                      //(boxes(0,i,1)-boxes(0,i,3))*image_size.height);
            cv::rectangle(image,p1,p2,cv::Scalar(0,0,0),1,8);
            
        }
    }
    cv::imshow( "Display window", image ); 
    cv::waitKey(0);



    return 0;
  

  //#################################
  //string graph_path = "../data/tensorflow_inception_graph.pb";
  //tensorflow::port::InitMain(input.c_str(),&argc, &argv);

  //tensorflow::GraphDef graph_def;
  //if (!ReadBinaryProto(tensorflow::Env::Default(), graph_path, &graph_def).ok()) {
    //LOG(ERROR) << "Read proto";
    //return -1;
  //}

  //std::unique_ptr<tensorflow::Session> session;
  //tensorflow::SessionOptions sess_opt;
  //sess_opt.config.mutable_gpu_options()->set_allow_growth(true);
  //(&session)->reset(tensorflow::NewSession(sess_opt));
  //if (!session->Create(graph_def).ok()) {
    //LOG(ERROR) << "Create graph";
    //return -1;
  //}

  ////const int batch_size = argc - 1;
  //const int batch_size = 1;
  //if (batch_size != 1) {
    //LOG(ERROR) << "Batch mode for the pretrained inception-v3 is unsupported";
    //LOG(ERROR) << " - https://github.com/tensorflow/tensorflow/issues/554";
    //return -1;
  //}

  //int32 input_dim = 299;
  //int32 input_mean = 128;
  //int32 input_std = 128;
  //std::vector<Tensor> inputs;
  ////std::string image_path(argv[1]);
  //if (!ReadTensorFromImageFile(input, input_dim, input_dim, input_mean,
                               //input_std, &inputs).ok()) {
    //LOG(ERROR) << "Load image";
    //return -1;
  //}

  //std::vector<Tensor> outputs;
  //string input_layer = "Mul";
  //string output_layer = "softmax";
  //if (!session->Run({{input_layer, inputs[0]}},
                     //{output_layer}, {}, &outputs).ok()) {
    //LOG(ERROR) << "Running model failed";
    //return -1;
  //}

  //Eigen::Map<Eigen::VectorXf> pred(outputs[0].flat<float>().data(),
                                   //outputs[0].NumElements());
  //int maxIndex; float maxValue = pred.maxCoeff(&maxIndex);
  //LOG(INFO) << "P( " << maxIndex << " | image ) = " << maxValue;

  //return 0;

}
