
include(FindPackageHandleStandardArgs)
unset(TensorFlow_FOUND)

find_path(TensorFlow_INCLUDE_DIR
        NAMES
        tensorflow/core
        tensorflow/cc
        third_party
        HINTS
        /opt/tensorflow/include/)
        

find_library(TensorFlow_LIBRARY NAMES tensorflow_cc tensorflow_framework
        HINTS
        /usr/lib
        /usr/local/lib
        /opt/tensorflow/lib)

# set TensorFlow_FOUND
find_package_handle_standard_args(TensorFlow DEFAULT_MSG TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY)
message(STATUS "tensor lib: " ${TensorFlow_LIBRARY}) 
message(STATUS "tensor include: " ${TensorFlow_INCLUDE_DIR}) 

# set external variables for usage in CMakeLists.txt
if(TensorFlow_FOUND)
    set(TensorFlow_LIBRARIES ${TensorFlow_LIBRARY})
    set(TensorFlow_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIR})
endif()

# hide locals from GUI
mark_as_advanced(TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY)
