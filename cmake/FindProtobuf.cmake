include(FindPackageHandleStandardArgs)
unset(Protobuf_FOUND)

find_path(Protobuf_INCLUDE_DIR
        NAMES
        google/protobuf 
        HINTS
        /opt/protobuf/include/)
        

find_library(Protobuf_LIBRARY 
        NAMES 
        protobuf
        HINTS
        /usr/lib
        /usr/local/lib
        /opt/protobuf/lib)

# set TensorFlow_FOUND
find_package_handle_standard_args(Protobuf DEFAULT_MSG Protobuf_INCLUDE_DIR Protobuf_LIBRARY)

# set external variables for usage in CMakeLists.txt
#if(TensorFlow_FOUND)
    #set(TensorFlow_LIBRARIES ${_LIBRARY})
    #set(TensorFlow_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIR})
#endif()

# hide locals from GUI
mark_as_advanced(Protobuf_INCLUDE_DIR Protobuf_LIBRARY)
