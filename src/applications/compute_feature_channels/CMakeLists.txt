# This is a CMake build file, for more information consult:
# http://en.wikipedia.org/wiki/CMake
# and
# http://www.cmake.org/Wiki/CMake
# http://www.cmake.org/cmake/help/syntax.html
# http://www.cmake.org/Wiki/CMake_Useful_Variables
# http://www.cmake.org/cmake/help/cmake-2-8-docs.html

# to compile the local code you can use: cmake ./ && make -j2

# ----------------------------------------------------------------------
# Base CMake setup

cmake_minimum_required (VERSION 2.6)

# ----------------------------------------------------------------------
# Setup the project

include(FindPkgConfig)
project(ComputeFeatureChannels)


set(doppia_root "${CMAKE_CURRENT_LIST_DIR}/../../..")

# ----------------------------------------------------------------------
# Site specific configurations
include(${doppia_root}/common_settings.cmake)

# ----------------------------------------------------------------------
# Setup required libraries

pkg_check_modules(libpng REQUIRED libpng)
#pkg_check_modules(opencv REQUIRED opencv>=2.3)

find_package(Boost REQUIRED  
   COMPONENTS program_options filesystem system thread
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
  "${doppia_root}/libs/cudatemplates/include"
  "${doppia_root}/src"
)

link_directories(
  ${libpng_LIBRARY_DIRS}
  ${opencv_LIBRARY_DIRS}
  ${Boost_LIBRARY_DIRS}
)

include_directories(
  ${local_INCLUDE_DIRS}
  "/usr/local/cuda/include"
  ${libpng_INCLUDE_DIRS}
  ${opencv_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

if(USE_GPU)
cuda_include_directories("${doppia_root}/libs/")
endif(USE_GPU)

# ----------------------------------------------------------------------

set(doppia_src "${doppia_root}/src")
set(doppia_stereo "${doppia_src}/stereo_matching")

file(GLOB SrcCpp
  "./*.*pp"

  "${doppia_src}/applications/Abstract*.cpp"
  "${doppia_src}/applications/BaseApplication.cpp"
  "${doppia_src}/applications/EmptyGui.cpp"

  "${doppia_src}/video_input/ImagesFromDirectory.cpp"

  "${doppia_src}/objects_detection/integral_channels/Hog*.cpp"
  "${doppia_src}/objects_detection/integral_channels/Geodesic*.cpp"
  "${doppia_src}/objects_detection/integral_channels/ColorSimilarity*.cpp"
  "${doppia_src}/objects_detection/integral_channels/Concatenated*.cpp"
  "${doppia_src}/objects_detection/integral_channels/IntegralChannelsForPedestrians.cpp"
  "${doppia_src}/objects_detection/integral_channels/AbstractChannelsComputer.cpp"
  "${doppia_src}/objects_detection/integral_channels/AbstractIntegralChannelsComputer.cpp"
  "${doppia_src}/objects_detection/integral_channels/*ChannelsComputerFactory.cpp"



  "${doppia_src}/image_processing/GeodesicTransform.cpp"
  "${doppia_src}/image_processing/fast_rgb_to_luv.cpp"
  "${doppia_src}/image_processing/rgb_to_hsv.cpp"

  "${doppia_src}/drawing/gil/draw_matrix.cpp"
  "${doppia_src}/drawing/gil/colors.cpp"

  "${doppia_src}/helpers/any_to_string.cpp"
  "${doppia_src}/helpers/Log.cpp"
  "${doppia_src}/helpers/ModuleLog.cpp"
  "${doppia_src}/helpers/progress_display_*.cpp"

)

file(GLOB SrcGpuCpp
  "${doppia_src}/objects_detection/integral_channels/AbstractGpu*.cpp"
  "${doppia_src}/objects_detection/integral_channels/Gpu*.cpp"
  "${doppia_src}/helpers/gpu/*.cpp"
)

file(GLOB SrcCuda
  "${doppia_src}/objects_detection/integral_channels/gpu/*.c*"
  #"${doppia_src}/helpers/gpu/*.cu"
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
# create the executable

add_executable (compute_feature_channels ${SrcCpp} )

target_link_libraries (compute_feature_channels

  ${Boost_LIBRARIES}
  gomp pthread
  ${libpng_LIBRARIES} jpeg
  ${opencv_LIBRARIES}
  opencv_core opencv_imgproc
)

if(USE_GPU)
target_link_libraries(compute_feature_channels cuda_stuff_library ${local_CUDA_LIB})
endif(USE_GPU)


# ----------------------------------------------------------------------
