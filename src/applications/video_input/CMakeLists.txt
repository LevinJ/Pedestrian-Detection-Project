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

set(doppia_root "../../../")

set(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "./" ${doppia_root} ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "/home/rodrigob/work/code/doppia_references/cuda/FindCUDA/CMake/cuda" ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "/users/visics/rbenenso/code/references/cuda/FindCUDA/CMake/cuda" ${CMAKE_MODULE_PATH})


# ----------------------------------------------------------------------
# Setup the project

include(FindPkgConfig)
project (VideoInput)

# ----------------------------------------------------------------------
# Site specific configurations
include(${doppia_root}/common_settings.cmake)

# ----------------------------------------------------------------------
# Setup required libraries

pkg_check_modules(libpng REQUIRED libpng)
pkg_check_modules(opencv REQUIRED opencv>=2.1)
list(APPEND opencv_LIBRARIES opencv_legacy)

set(opencv_LIBRARIES
    opencv_core opencv_imgproc opencv_highgui opencv_ml
    opencv_video opencv_features2d
    opencv_calib3d
    #opencv_objdetect opencv_contrib
    opencv_legacy opencv_flann
   ) # quick hack for opencv2.4 support

find_package(Boost REQUIRED  
   COMPONENTS program_options filesystem system thread
)

# ----------------------------------------------------------------------
# Setup CUDA

set(CUDA_BUILD_CUBIN OFF)
find_package(CUDA 2.2 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_CUT_INCLUDE_DIR})

# ----------------------------------------------------------------------
# Setup link and include directories

set(local_LIBRARY_DIRS
  "/users/visics/rbenenso/no_backup/usr/local/lib"
  "/usr/lib64"
  "/usr/lib64/atlas"
  "/usr/lib/sse2/atlas"
)
set(local_INCLUDE_DIRS
  "/users/visics/rbenenso/no_backup/usr/local/include"
  "/usr/include/eigen2/"
   "/usr/local/include/eigen2"
)

link_directories(
  ${libpng_LIBRARY_DIRS}
  ${OpenEXR_LIBRARY_DIRS}
  ${opencv_LIBRARY_DIRS}
  ${Boost_LIBRARY_DIRS}
  ${local_LIBRARY_DIRS}
)

include_directories(
  ${doppia_root}/libs
  ${doppia_root}/src
  ${libpng_INCLUDE_DIRS}
  ${OpenEXR_INCLUDE_DIRS}
  ${opencv_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${local_INCLUDE_DIRS}
)
#cuda_include_directories(./libs )


# ----------------------------------------------------------------------
# Collect source files

set(doppia_src "${doppia_root}/src")

file(GLOB SrcCpp "./*.cpp"
                 "${doppia_src}/*.cpp"
                 "${doppia_src}/applications/*.cpp"
                 "${doppia_src}/video_input/*.cpp"
                 "${doppia_src}/video_input/calibration/*.c*"
                 "${doppia_src}/video_input/preprocessing/*.cpp"
)


file(GLOB HelpersCpp
  #"${doppia_src}/helpers/*.cpp"
  "${doppia_src}/helpers/any_to_string.cpp"
  "${doppia_src}/helpers/get_section_options.cpp"
  "${doppia_src}/helpers/Log.cpp"  
  "${doppia_src}/helpers/ModuleLog.cpp"
  "${doppia_src}/helpers/loggers.cpp"
)


# ----------------------------------------------------------------------
# Create the executable
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_EXR")

add_executable(video_input
  ${SrcCpp}
  ${HelpersCpp}

  ${ShaderObjectFiles}
)

target_link_libraries(video_input

   ${cg_LIBRARIES}
   # linking with CgGL _after_ boost_program_options generates a segmentation fault ! boost_program_options 1.39 has a bug

   ${Boost_LIBRARIES}
   protobuf pthread
   SDL
   gomp Xext Xrandr X11
   ${libpng_LIBRARIES}  jpeg
   ${opencv_LIBRARIES}

   #csparse sparse spblas mv
   #lapack blas atlas
)

# ----------------------------------------------------------------------
