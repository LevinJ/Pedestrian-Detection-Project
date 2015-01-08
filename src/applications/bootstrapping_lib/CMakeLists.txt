# This is a CMake build file, for more information consult:
# http://en.wikipedia.org/wiki/CMake
# and
# http://www.cmake.org/Wiki/CMake
# http://www.cmake.org/cmake/help/syntax.html
# http://www.cmake.org/Wiki/CMake_Useful_Variables
# http://www.cmake.org/cmake/help/cmake-2-8-docs.html

# to compile the local code you can use: cmake ./ && make -j2

cmake_minimum_required (VERSION 2.6)

set(doppia_root "../../..")

set(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "./" ${doppia_root} ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "/home/rodrigob/work/code/doppia_references/cuda/FindCUDA/CMake/cuda" ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "/users/visics/rbenenso/code/references/cuda/FindCUDA/CMake/cuda" ${CMAKE_MODULE_PATH})

include(FindPkgConfig)
project (BootstrappingLib)

# ----------------------------------------------------------------------
# Site specific configurations
include(${doppia_root}/common_settings.cmake)

pkg_check_modules(libpng REQUIRED libpng)
#pkg_check_modules(opencv REQUIRED opencv>=2.3)

set(opencv_LIBRARIES
    opencv_core opencv_imgproc opencv_highgui opencv_ml
    opencv_video opencv_features2d
    opencv_calib3d
    #opencv_objdetect opencv_contrib
    opencv_legacy opencv_flann
   ) # quick hack for opencv2.4 support

find_package(Boost REQUIRED  
   COMPONENTS 
   unit_test_framework
   program_options filesystem system thread
)

# ----------------------------------------------------------------------
# Setup CUDA
if(USE_GPU)
  find_package(CUDA 4.0 REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_CUT_INCLUDE_DIR})
endif(USE_GPU)

# ----------------------------------------------------------------------
set(local_INCLUDE_DIRS
    "${doppia_root}/libs"
    "${doppia_root}/libs/eigen3"
    "${doppia_root}/src"
    "${doppia_root}/src/applications/boosted_learning"
    #"/usr/include/eigen2"
    #"/usr/local/include/eigen2"
    "/users/visics/rbenenso/no_backup/usr/local/include"
    "${doppia_root}/src/objects_detection/integral_channels" # to avoid doppia specific paths in the library header
    #"/users/visics/mmathias/devel/chFeatDet/" # include mmathias code
    "/usr/local/cuda/include"
    )

link_directories(
  "/usr/local/lib"
  "/users/visics/rbenenso/no_backup/usr/local/lib"
  ${libpng_LIBRARY_DIRS}
  ${opencv_LIBRARY_DIRS}
  ${Boost_LIBRARY_DIRS}
)

include_directories(
 ${local_INCLUDE_DIRS}
 "${doppia_root}/libs/cudatemplates/include"
 "${doppia_root}/src/applications/bootstrapping_lib"
  ${opencv_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

if(USE_GPU)
cuda_include_directories("${doppia_root}/libs/")
endif(USE_GPU)

# ----------------------------------------------------------------------
# Collect source files

add_definitions(-DBOOTSTRAPPING_LIB)

set(doppia_src "${doppia_root}/src")

file(GLOB SrcCpp
  "./*.*pp"
  #"${doppia_src}/objects_detection/*.c*"
  "${doppia_src}/objects_detection/Abstract*.c*"
  "${doppia_src}/objects_detection/*Converter.c*"
  "${doppia_src}/objects_detection/Base*.c*"
  "${doppia_src}/objects_detection/*Factory.c*"
  "${doppia_src}/objects_detection/Detection*.c*"
  "${doppia_src}/objects_detection/*Model.c*"
  "${doppia_src}/objects_detection/*Stage.c*"
  "${doppia_src}/objects_detection/Integral*.c*"
  "${doppia_src}/objects_detection/VeryFastIntegral*.c*"
  "${doppia_src}/objects_detection/SearchRange*.c*"
  "${doppia_src}/objects_detection/MultiscalesIntegral*.c*"
  "${doppia_src}/objects_detection/integral_channels/Integral*.c*"
  "${doppia_src}/objects_detection/FastestPedestrian*.c*"
  "${doppia_src}/objects_detection/DetectorSearchRange.c*"
  "${doppia_src}/objects_detection/*.pb.c*"
  "${doppia_src}/objects_detection/Greedy*.c*"
  "${doppia_src}/objects_detection/cascade_stages/*.c*"
  "${doppia_src}/objects_detection/non_maximal_suppression/*.c*"
  "${doppia_src}/objects_detection/integral_channels/Abstract[^(Gpu)]*.cpp"



  "${doppia_src}/applications/Abstract*.cpp"
  "${doppia_src}/applications/BaseApplication.cpp"

  "${doppia_src}/video_input/*.cpp"
  "${doppia_src}/video_input/calibration/*.c*"
  "${doppia_src}/video_input/preprocessing/*.cpp"
  "${doppia_src}/image_processing/*.cpp"

)

file(GLOB HelpersCpp
  #"${doppia_src}/helpers/*.cpp"
  #"${doppia_src}/helpers/data/*.c*"
  "${doppia_src}/helpers/any_to_string.cpp"
  "${doppia_src}/helpers/Log.cpp"
 "${doppia_src}/helpers/ModuleLog.cpp"
  "${doppia_src}/helpers/loggers.cpp"
  "${doppia_src}/helpers/replace_environment_variables.cpp"
  #"${doppia_src}/helpers/AlignedImage.cpp"
  #"${doppia_src}/drawing/gil/*.cpp"
  "${doppia_src}/drawing/gil/draw_matrix.cpp"
  "${doppia_src}/drawing/gil/colors.cpp"
  "${doppia_src}/helpers/objects_detection/*.cpp"
)


file(GLOB SrcGpuCpp
  "${doppia_src}/objects_detection/AbstractGpu*.cpp"
  "${doppia_src}/objects_detection/Gpu*.cpp"
  "${doppia_src}/objects_detection/integral_channels/AbstractGpu*.cpp"
  "${doppia_src}/objects_detection/integral_channels/Gpu*.cpp"
  "${doppia_src}/helpers/gpu/*.cpp"

  #"${doppia_stereo}/SimpleTreesGpuStereo.cpp"
)

file(GLOB SrcCuda
  "${doppia_src}/objects_detection/integral_channels/gpu/*.c*"
  "${doppia_src}/objects_detection/gpu/*.c*"
  #"${doppia_src}/helpers/gpu/*.cu"

#  "${doppia_stereo}/*.cu.c*"
#  "${doppia_stereo}/*.cu"
#  "${doppia_stereo}/gpu/*.cu.c*"
#  "${doppia_stereo}/gpu/*.cu"
)

list(REMOVE_ITEM SrcCpp ${SrcCuda}) # just in case

if(USE_GPU)

# add GPU related source code to the executable list
list(APPEND SrcCpp ${SrcGpuCpp})

# add GPU related libraries
list(APPEND opencv_LIBRARIES opencv_gpu)

# ----------------------------------------------------------------------
# Compile CUDA stuff

cuda_include_directories(${local_CUDA_CUT_INCLUDE_DIRS})
cuda_include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_CUT_INCLUDE_DIR} ${local_CUDA_CUT_INCLUDE_DIR})
link_directories(${local_CUDA_CUT_LIBRARY_DIRS})

cuda_add_library(cuda_stuff_library ${SrcCuda})
target_link_libraries(cuda_stuff_library
   ${CUDA_LIBRARIES}
   #cutil
)
endif(USE_GPU)
# ----------------------------------------------------------------------
# Create the library
add_library( bootstrapping SHARED ${SrcCpp} ${HelpersCpp} )

target_link_libraries(bootstrapping
   ${Boost_LIBRARIES}
   pthread gomp
   protobuf
   ${opencv_LIBRARIES}
#   SDL Xext Xrandr X11
   ${libpng_LIBRARIES} jpeg
)


if(USE_GPU)
target_link_libraries(bootstrapping cuda_stuff_library ${local_CUDA_LIB})
endif(USE_GPU)

# ----------------------------------------------------------------------
# Create the test executable
add_executable(test_bootstrapping_lib "test_bootstrapping_lib.cpp")

target_link_libraries(test_bootstrapping_lib bootstrapping)

target_link_libraries(test_bootstrapping_lib
   ${Boost_LIBRARIES}
   pthread gomp
# SDL Xext Xrandr X11
   ${libpng_LIBRARIES} jpeg
)


# ----------------------------------------------------------------------
