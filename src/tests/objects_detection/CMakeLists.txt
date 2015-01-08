# This is a CMake build file, for more information consult:
# http://en.wikipedia.org/wiki/CMake
# and
# http://www.cmake.org/Wiki/CMake
# http://www.cmake.org/cmake/help/syntax.html
# http://www.cmake.org/Wiki/CMake_Useful_Variables
# http://www.cmake.org/cmake/help/cmake-2-8-docs.html

# to compile the local code you can use: cmake ./ && make -j2

cmake_minimum_required (VERSION 2.6)

set(doppia_root "../../../")

set(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "./" ${doppia_root} ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "/home/rodrigob/work/code/doppia_references/cuda/FindCUDA/CMake/cuda" ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "/users/visics/rbenenso/code/references/cuda/FindCUDA/CMake/cuda" ${CMAKE_MODULE_PATH})

# ----------------------------------------------------------------------
# Setup the project

include(FindPkgConfig)
project (TestObjectsDetection)

# ----------------------------------------------------------------------
# Site specific configurations
include(${doppia_root}/common_settings.cmake)

# ----------------------------------------------------------------------
# Setup required libraries

pkg_check_modules(opencv REQUIRED opencv>=2.3)
pkg_check_modules(libpng REQUIRED libpng)

set(opencv_LIBRARIES
    opencv_core opencv_imgproc opencv_highgui opencv_ml
    opencv_video opencv_features2d
    opencv_calib3d
    #opencv_objdetect opencv_contrib
    opencv_legacy opencv_flann
   ) # quick hack for opencv2.4 support

# ----------------------------------------------------------------------
# Setup CUDA

if(USE_GPU)
find_package(CUDA 4.0 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_CUT_INCLUDE_DIR})
endif(USE_GPU)

# ----------------------------------------------------------------------
set(local_LIBRARY_DIRS
  "/usr/local/lib"
  "/users/visics/rbenenso/no_backup/usr/local/lib"
  "/usr/lib64"
  "/usr/lib64/atlas"
  "/usr/lib/sse2/atlas"
  "/usr/lib/llvm-2.8/lib"
  ${local_CUDA_LIB_DIR}
)

set(local_INCLUDE_DIRS
  "/users/visics/rbenenso/no_backup/usr/local/include"
  "/usr/include/eigen2/"
   "/usr/local/include/eigen2"
   "/usr/local/cuda/include"
   ${CUDA_INCLUDE_DIRS}
)


link_directories(
  ${libpng_LIBRARY_DIRS}
  ${opencv_LIBRARY_DIRS}
  ${local_LIBRARY_DIRS}
)

include_directories(
  "${doppia_root}/libs"
  "${doppia_root}/src"
  ${libpng_INCLUDE_DIRS}
  ${OpenEXR_INCLUDE_DIRS}
  ${opencv_INCLUDE_DIRS}
  ${local_INCLUDE_DIRS}
  "${doppia_root}/libs/cudatemplates/include"
)


if(USE_GPU)
cuda_include_directories("${doppia_root}/libs/")
endif(USE_GPU)
# ----------------------------------------------------------------------
# Collect source files

set(doppia_src "${doppia_root}/src")
set(doppia_stereo "${doppia_root}/src/stereo_matching")

file(GLOB SrcCpp
  "./*.*pp"
  #"${doppia_src}/objects_detection/*.c*"
  "${doppia_src}/objects_detection/Abstract*.c*"
  "${doppia_src}/objects_detection/Base*.c*"
  "${doppia_src}/objects_detection/*Factory.c*"
  "${doppia_src}/objects_detection/Detection*.c*"
  "${doppia_src}/objects_detection/*Model*.c*"
  "${doppia_src}/objects_detection/*Stage.c*"
  "${doppia_src}/objects_detection/*Integral*.c*"
  "${doppia_src}/objects_detection/SearchRange*.c*"
  "${doppia_src}/objects_detection/integral_channels/Integral*.c*"
  "${doppia_src}/objects_detection/FastestPedestrian*.c*"
  "${doppia_src}/objects_detection/DetectorSearchRange.c*"
  "${doppia_src}/objects_detection/*.pb.c*"
  "${doppia_src}/objects_detection/Greedy*.c*"
  "${doppia_src}/objects_detection/non_maximal_suppression/*.c*"

  "${doppia_src}/objects_tracking/*.cpp"
  "${doppia_src}/objects_tracking/tracked_detections/*.cpp"

  "${doppia_src}/applications/Abstract*.cpp"
  "${doppia_src}/applications/Base*.cpp"
  "${doppia_src}/applications/*Gui.cpp"

  "${doppia_src}/applications/objects_detection/ObjectsDetection*.cpp"
  "${doppia_src}/applications/objects_detection/draw*.cpp"
  "${doppia_src}/applications/objects_tracking/*Application.cpp"

# -------------------------------------
# Copy-pasted from cmake file of ObjectsDetectionApplication

  "${doppia_src}/objects_detection/Abstract*.c*"
  "${doppia_src}/objects_detection/*Converter.c*"
  "${doppia_src}/objects_detection/Base*.c*"
  "${doppia_src}/objects_detection/*Factory.c*"
  "${doppia_src}/objects_detection/Greedy*.c*"
  "${doppia_src}/objects_detection/Detection*.c*"
  "${doppia_src}/objects_detection/*Model.c*"
  "${doppia_src}/objects_detection/*Stage.c*"
  "${doppia_src}/objects_detection/*Integral*.c*"
  "${doppia_src}/objects_detection/MultiscalesIntegral*.c*"
  "${doppia_src}/objects_detection/FastestPedestrian*.c*"
  "${doppia_src}/objects_detection/DetectorSearchRange.c*"
  "${doppia_src}/objects_detection/*.pb.c*"
  "${doppia_src}/objects_detection/cascade_stages/*.c*"
  "${doppia_src}/objects_detection/non_maximal_suppression/*.c*"
  "${doppia_src}/objects_detection/integral_channels/Integral*.cpp"
  "${doppia_src}/objects_detection/integral_channels/AbstractIntegral*.cpp"
  "${doppia_src}/objects_detection/integral_channels/AbstractChannels*.cpp"

  "${doppia_src}/applications/*.cpp"
  "${doppia_src}/applications/stixel_world/*Gui.cpp"
  "${doppia_src}/applications/stixel_world/draw*.cpp"

  #"${doppia_stereo}/*.cpp"
  "${doppia_stereo}/cost_volume/*CostVolume.cpp"
  "${doppia_stereo}/cost_volume/*CostVolumeEstimator*.cpp"
  "${doppia_stereo}/cost_volume/DisparityCostVolumeFromDepthMap.cpp"
  "${doppia_stereo}/cost_functions.cpp"
  "${doppia_stereo}/CensusCostFunction.cpp"
  "${doppia_stereo}/CensusTransform.cpp"
  "${doppia_stereo}/GradientTransform.cpp"
  "${doppia_stereo}/AbstractStereoMatcher.cpp"
  "${doppia_stereo}/AbstractStereoBlockMatcher.cpp"
  "${doppia_stereo}/SimpleBlockMatcher.cpp"
  "${doppia_stereo}/MutualInformationCostFunction.cpp"
  "${doppia_stereo}/ConstantSpaceBeliefPropagation.cpp"
  "${doppia_stereo}/qingxiong_yang/*.cpp"
  "${doppia_stereo}/SimpleTreesOptimizationStereo.cpp"
  "${doppia_stereo}/OpenCvStereo.cpp"

  "${doppia_stereo}/ground_plane/*.cpp"
  "${doppia_stereo}/stixels/*.cpp"
   #"${doppia_stereo}/stixels/*.cc"
  "${doppia_src}/video_input/*.cpp"
  "${doppia_src}/video_input/calibration/*.c*"
  "${doppia_src}/video_input/preprocessing/*.cpp"
  #"${doppia_src}/features_tracking/*.cpp"
  "${doppia_src}/image_processing/*.cpp"
  "${doppia_src}/drawing/gil/*.cpp"


  "${doppia_src}/helpers/data/*.c*"
  "${doppia_src}/helpers/any_to_string.cpp"
  "${doppia_src}/helpers/get_section_options.cpp"
  "${doppia_src}/helpers/Log.cpp"
  "${doppia_src}/helpers/loggers.cpp"
  "${doppia_src}/helpers/AlignedImage.cpp"
  "${doppia_src}/helpers/replace_*prefix.cpp"

# ---------------------------------------

  "${doppia_src}/video_input/ImagesFromDirectory.cpp"
  "${doppia_src}/video_input/*.cpp"
  "${doppia_src}/video_input/calibration/*.c*"
  "${doppia_src}/video_input/preprocessing/*.cpp"
  "${doppia_src}/image_processing/*.cpp"
)

file(GLOB HelpersCpp
  #"${doppia_src}/helpers/*.cpp"
  #"${doppia_src}/helpers/data/*.c*"
  "${doppia_src}/helpers/any_to_string.cpp"
  "${doppia_src}/helpers/ModuleLog.cpp"
  "${doppia_src}/helpers/Log.cpp"
  "${doppia_src}/helpers/loggers.cpp"
  "${doppia_src}/helpers/replace_environment_variables.cpp"
  #"${doppia_src}/helpers/AlignedImage.cpp"
  "${doppia_src}/drawing/gil/*.cpp"
)

file(GLOB SrcGpuCpp
  "${doppia_src}/objects_detection/Gpu*.cpp"
  "${doppia_src}/objects_detection/integral_channels/AbstractGpu*.cpp"
  "${doppia_src}/objects_detection/integral_channels/Gpu*.cpp"
  "${doppia_src}/helpers/gpu/*.cpp"
)

file(GLOB SrcCuda
  "${doppia_src}/objects_detection/integral_channels/gpu/*.cu"
  "${doppia_src}/objects_detection/integral_channels/gpu/*.cpp"
  "${doppia_src}/objects_detection/gpu/*.cu"
  "${doppia_src}/objects_detection/gpu/*.cpp"

  #"${doppia_stereo}/*.cu.c*"
  #"${doppia_stereo}/*.cu"
  #"${doppia_stereo}/gpu/*.cu.c*"
  #"${doppia_stereo}/gpu/*.cu"
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
  ${cutil_LIB}
)
endif(USE_GPU)
# ----------------------------------------------------------------------
add_definitions(-DBOOST_TEST_DYN_LINK -DTESTING)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x") # required for unrestricted unions

add_executable(test_objects_detection
  ${SrcCpp}
  ${HelpersCpp}
)

target_link_libraries(test_objects_detection
  boost_unit_test_framework-mt
  boost_program_options-mt boost_filesystem-mt boost_system-mt
  boost_thread-mt pthread gomp
  protobuf
  ${opencv_LIBRARIES} opencv_legacy # required when using opencv 2.4
  SDL X11 Xext #Xrandr
  ${libpng_LIBRARIES} jpeg
)

if(USE_GPU)
target_link_libraries(test_objects_detection cuda_stuff_library ${local_CUDA_LIB})
endif(USE_GPU)

# ----------------------------------------------------------------------
