cmake_minimum_required (VERSION 2.8.11)

set(PACKAGE_VERSION 1.3)
project("sample")

#=========== variables explanation ============
#project_SOURCES: *.cpp files written in ./src (apart from main)
#project_LIBS: depenencies (OpenCV, Caffe)
#project_INCLUDE_DIRS: include directories for the project

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11 " COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

#add compiler support 
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")

# set archive (lib) output directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#.so in /lib
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# exe in /bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#allow user to set custom directory
set(CPACK_set_DESTDIR "on")




##### ======= find dependencies =======

#if -DDEBUG=on compile in debug mode
if(DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
endif(DEBUG)


##### ======= find dependencies =======
#Note:  if you have built a dendency but it doesn't find it, 
#       set <dependency>_DIR, cmake will take care of finding the Config.cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include("FindEigen")
include("FindProtobuf")
include("FindTensorFlow")

# has to be before Caffe
find_package( OpenCV REQUIRED )

#find_package(Eigen3 REQUIRED)

#find_package(TensorFlow REQUIRED)

find_package(Protobuf REQUIRED)



##### ======= add all dependencies together in ${project_INCLUDE_DIRS}/${project_INCLUDE_DIRS} =======
if (TensorFlow_FOUND)
     MESSAGE(STATUS "tensorflow include: ${TensorFlow_INCLUDE_DIRS} ")
     MESSAGE(STATUS "tensorflow libs: ${TensorFlow_LIBRARY} ")
     set ( project_INCLUDE_DIRS ${project_INCLUDE_DIRS} ${TensorFlow_INCLUDE_DIRS} )
     set ( dependencies_LIBS ${dependencies_LIBS} ${TensorFlow_LIBRARY})
endif()
if (OpenCV_FOUND)
     MESSAGE(STATUS "opencv include: ${OpenCV_INCLUDE_DIRS} ")
     MESSAGE(STATUS "opencv libs: ${OpenCV_LIBS} ")
     #set ( project_INCLUDE_DIRS ${project_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} )
     set ( dependencies_LIBS ${dependencies_LIBS} ${OpenCV_LIBS})
endif()


if (Eigen_FOUND)
    MESSAGE(STATUS "eigen include: ${Eigen_INCLUDE_DIR} ")
    set ( project_INCLUDE_DIRS ${project_INCLUDE_DIRS} ${Eigen_INCLUDE_DIR} )
endif()

if (Protobuf_FOUND)
    MESSAGE(STATUS "Protobuf include: ${Protobuf_INCLUDE_DIR} ")
    MESSAGE(STATUS "Protobuf libs: ${Protobuf_LIBRARY} ")
    set ( project_INCLUDE_DIRS ${project_INCLUDE_DIRS} ${Protobuf_INCLUDE_DIR} )
    set ( dependencies_LIBS ${dependencies_LIBS} ${Protobuf_LIBRARY})
endif()



##### ======= add all dependencies together in ${project_INCLUDE_DIRS}/${project_INCLUDE_DIRS} =======

#add_subdirectory(src)


##### ======= create library libsample.so =======
#add_library( ${PROJECT_NAME} SHARED
    #${project_SOURCES}
    #)

#target_link_libraries(${PROJECT_NAME}
        #${dependencies_LIBS})

#target_include_directories(${PROJECT_NAME} PUBLIC
        #${project_INCLUDE_DIRS})

#add_subdirectory(python)





##### ======= add examples in (bin/) and link libensemble =======
#add executable example
add_executable( example ./src/main.cpp)

##link ensemble library to executable
target_link_libraries( example ${dependencies_LIBS})
target_include_directories(example PUBLIC
        ${project_INCLUDE_DIRS})
