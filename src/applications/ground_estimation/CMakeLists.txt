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

cmake_minimum_required(VERSION 2.6)

set(doppia_root "${CMAKE_CURRENT_LIST_DIR}/../../..")

set(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "./" ${doppia_root} ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "/home/rodrigob/work/code/doppia_references/cuda/FindCUDA/CMake/cuda" ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "/users/visics/rbenenso/code/references/cuda/FindCUDA/CMake/cuda" ${CMAKE_MODULE_PATH})

# ----------------------------------------------------------------------
# Setup the project

include(FindPkgConfig)
project (GroundEstimation)

# ----------------------------------------------------------------------
# Site specific configurations
include(${doppia_root}/common_settings.cmake)

# ----------------------------------------------------------------------
# Setup required libraries

pkg_check_modules(libpng REQUIRED libpng)
#pkg_check_modules(libjpeg REQUIRED libjpeg)
pkg_check_modules(opencv REQUIRED opencv>=2.3)

set(opencv_LIBRARIES
    opencv_core opencv_imgproc opencv_highgui opencv_ml
    opencv_video opencv_features2d
    opencv_calib3d
    #opencv_objdetect opencv_contrib
    opencv_legacy opencv_flann
    opencv_gpu
   ) # quick hack for opencv2.4 support

find_package(Boost REQUIRED  
   COMPONENTS program_options filesystem system thread
)

# ----------------------------------------------------------------------
# Setup link and include directories

set(local_LIBRARY_DIRS
  "/usr/local/lib"
  "/users/visics/rbenenso/no_backup/usr/local/lib"
  "/usr/lib64"
  "/usr/lib64/atlas"
  "/usr/lib/sse2/atlas"
)
set(local_INCLUDE_DIRS
  "/users/visics/rbenenso/no_backup/usr/local/include"
  "/usr/include/eigen2/"
   "/usr/local/include/eigen2"
  "${doppia_root}/libs/cudatemplates/include"
)

link_directories(
  ${libpng_LIBRARY_DIRS}
  ${opencv_LIBRARY_DIRS}
  ${Boost_LIBRARY_DIRS}
  ${local_LIBRARY_DIRS}
)

include_directories(
  ${doppia_root}/libs
  ${doppia_root}/src
  ${libpng_INCLUDE_DIRS}
  ${opencv_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${local_INCLUDE_DIRS}
)

#cuda_include_directories(./libs )

# ----------------------------------------------------------------------
# Collect source files

set(doppia_src "${doppia_root}/src")
set(doppia_stereo "${doppia_root}/src/stereo_matching")

file(GLOB SrcCpp
  "./*.cpp"
  "${doppia_src}/*.cpp"
  "${doppia_src}/applications/*.cpp"

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
  #"${doppia_stereo}/stixels/*.cpp"
  #"${doppia_stereo}/stixels/*.cc"
  "${doppia_src}/video_input/*.cpp"
  "${doppia_src}/video_input/calibration/*.c*"
  "${doppia_src}/video_input/preprocessing/*.cpp"
  #"${doppia_src}/features_tracking/*.cpp"
  "${doppia_src}/image_processing/*.cpp"
  "${doppia_src}/drawing/gil/draw_the_ground_corridor.cpp"
  "${doppia_src}/drawing/gil/*line.cpp"
  "${doppia_src}/drawing/gil/colors.cpp"
  "${doppia_src}/drawing/gil/hsv_to_rgb.cpp"
  "${doppia_src}/drawing/gil/draw_matrix.cpp"
)

file(GLOB HelpersCpp
  #"${doppia_src}/helpers/*.cpp"
  "${doppia_src}/helpers/data/*.c*"
  "${doppia_src}/helpers/any_to_string.cpp"
  "${doppia_src}/helpers/get_section_options.cpp"
  "${doppia_src}/helpers/Log.cpp"
  "${doppia_src}/helpers/ModuleLog.cpp"
  "${doppia_src}/helpers/AlignedImage.cpp"
)

# ----------------------------------------------------------------------
# Create the executable

add_executable(ground_estimation
  ${SrcCpp}
  ${HelpersCpp}
)

target_link_libraries(ground_estimation

   ${cg_LIBRARIES}
   # linking with CgGL _after_ boost_program_options generates a segmentation fault ! boost_program_options 1.39 has a bug

   ${Boost_LIBRARIES}
   protobuf pthread
   SDL
   gomp Xext Xrandr X11
   ${libpng_LIBRARIES} jpeg
   ${opencv_LIBRARIES} opencv_legacy

   #csparse sparse spblas mv
   #lapack blas atlas

   #${google_perftools_LIBS}
   # faster malloc and non intrusive profiler
   # via http://google-perftools.googlecode.com
)

# ----------------------------------------------------------------------
