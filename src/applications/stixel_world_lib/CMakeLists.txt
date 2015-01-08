# This is a CMake build file, for more information consult:
# http://en.wikipedia.org/wiki/CMake
# and
# http://www.cmake.org/Wiki/CMake
# http://www.cmake.org/cmake/help/syntax.html
# http://www.cmake.org/Wiki/CMake_Useful_Variables
# http://www.cmake.org/cmake/help/cmake-2-8-docs.html

# to compile the local code you can use: cmake ./ && make -j2

cmake_minimum_required (VERSION 2.6)

set(doppia_root "${CMAKE_CURRENT_LIST_DIR}/../../..")

set(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "./" ${doppia_root} ${CMAKE_MODULE_PATH})

include(FindPkgConfig)
project (StixelWorldLib)

# ----------------------------------------------------------------------
# Site specific configurations
include(${doppia_root}/common_settings.cmake)
set(USE_GPU OFF CACHE BOOL "Should the GPU be used ?" FORCE)

# ----------------------------------------------------------------------
# Setup required libraries

pkg_check_modules(libpng REQUIRED libpng)
pkg_check_modules(opencv REQUIRED opencv>=2.3)

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
# Setup link and include directories

set(local_INCLUDE_DIRS
    "${doppia_root}/libs"
    "${doppia_root}/src"
    "${doppia_root}/src/applications/boosted_learning"
    #"/usr/include/eigen2"
    #"/usr/local/include/eigen2"
    "/users/visics/rbenenso/no_backup/usr/local/include"
    )

set(local_LIBRARY_DIRS
  "/users/visics/rbenenso/no_backup/usr/local/lib"
  "/usr/lib64"
  "/usr/lib64/atlas"
  "/usr/lib/sse2/atlas"
  "/usr/lib/llvm-2.8/lib"
  ${local_CUDA_LIB_DIR}
)

link_directories(
  ${libpng_LIBRARY_DIRS}
  ${OpenEXR_LIBRARY_DIRS}
  ${opencv_LIBRARY_DIRS}
  ${Boost_LIBRARY_DIRS}
  ${local_LIBRARY_DIRS}
)

include_directories(
 ${local_INCLUDE_DIRS}
 "${doppia_root}/src/applications/bootstrapping_lib"
  ${libpng_INCLUDE_DIRS}
  ${opencv_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

# ----------------------------------------------------------------------
# Collect source files

add_definitions(-DSTIXEL_WORLD_LIB)

set(doppia_src "${doppia_root}/src")
set(doppia_stereo "${doppia_root}/src/stereo_matching")

file(GLOB SrcCpp
  "stixel_world_lib.cpp"
  "${doppia_src}/*.cpp"

  "${doppia_src}/applications/*Application.cpp"
  "${doppia_src}/applications/AbstractGui.cpp"
  "${doppia_src}/applications/EmptyGui.cpp"

  "${doppia_src}/objects_detection/Detection2d.cpp"
  "${doppia_src}/objects_tracking/tracked_detections/TrackedDetection2d.cpp"

  "${doppia_src}/image_processing/*.cpp"
  "${doppia_src}/drawing/gil/draw_matrix.cpp"
)


file(GLOB StereoSrcCpp

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
  "${doppia_src}/video_input/preprocessing/*Preprocessor.cpp"
  "${doppia_src}/video_input/preprocessing/*Mapper.cpp"
)


file(GLOB GuiSrcCpp
  "StixelWorldLibGui.cpp"
  "${doppia_src}/applications/stixel_world/draw*.cpp"
  "${doppia_src}/drawing/gil/*.cpp"
)

file(GLOB HelpersCpp

  #"${doppia_src}/helpers/*.cpp"
  "${doppia_src}/helpers/data/*.c*"
  "${doppia_src}/helpers/any_to_string.cpp"
  "${doppia_src}/helpers/get_section_options.cpp"
  "${doppia_src}/helpers/ModuleLog.cpp"
  "${doppia_src}/helpers/Log.cpp"
  "${doppia_src}/helpers/loggers.cpp"
  "${doppia_src}/helpers/AlignedImage.cpp"
  "${doppia_src}/helpers/replace_environment_variables.cpp"

)

# ----------------------------------------------------------------------
# Create the library

add_library(stixel_world SHARED ${SrcCpp} ${MonocularSrcCpp} ${StereoSrcCpp} ${HelpersCpp} )

target_link_libraries(stixel_world
   ${Boost_LIBRARIES}
   pthread gomp
   protobuf
   ${opencv_LIBRARIES}
#   SDL Xext Xrandr X11
   ${libpng_LIBRARIES} jpeg
)

# ----------------------------------------------------------------------
# Create a version of the library with UI integrated (gui.disabled is true by default)
add_library(stixel_world_with_ui SHARED
  ${GuiSrcCpp} ${SrcCpp} ${MonocularSrcCpp} ${StereoSrcCpp} ${HelpersCpp}
)

set_property(
   TARGET stixel_world_with_ui
   PROPERTY COMPILE_DEFINITIONS STIXEL_WORLD_WITH_UI_LIB
)

target_link_libraries(stixel_world_with_ui
   ${Boost_LIBRARIES}
   pthread gomp
   protobuf
   ${opencv_LIBRARIES}
#   SDL Xext Xrandr X11
   ${libpng_LIBRARIES} jpeg
)

# ----------------------------------------------------------------------
# Create the test executable
add_executable(test_stixel_world_lib
  "test_stixel_world_lib.cpp"
  "TestStixelWorldApplication.cpp"
)

target_link_libraries(test_stixel_world_lib stixel_world)

target_link_libraries(test_stixel_world_lib
   ${Boost_LIBRARIES}
   gomp pthread
   ${opencv_LIBRARIES}
#  SDL Xext Xrandr X11
   ${libpng_LIBRARIES} jpeg
)

# ----------------------------------------------------------------------
# Create the test executable with UI
add_executable(test_stixel_world_with_ui_lib
  "test_stixel_world_lib.cpp"
  "TestStixelWorldApplication.cpp"
)

target_link_libraries(test_stixel_world_with_ui_lib stixel_world_with_ui)

target_link_libraries(test_stixel_world_with_ui_lib
   ${Boost_LIBRARIES}
   gomp pthread
   ${opencv_LIBRARIES}
   SDL Xext Xrandr X11
   ${libpng_LIBRARIES} jpeg
)

# ----------------------------------------------------------------------
