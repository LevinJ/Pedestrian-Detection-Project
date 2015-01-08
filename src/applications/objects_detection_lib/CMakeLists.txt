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

set(doppia_root "${CMAKE_CURRENT_LIST_DIR}/../../..")

set(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "./" ${doppia_root} ${CMAKE_MODULE_PATH})
#set(CMAKE_MODULE_PATH "/home/rodrigob/work/code/doppia_references/cuda/FindCUDA/CMake/cuda" ${CMAKE_MODULE_PATH})
#set(CMAKE_MODULE_PATH "/users/visics/rbenenso/code/references/cuda/FindCUDA/CMake/cuda" ${CMAKE_MODULE_PATH})
# ----------------------------------------------------------------------
# Setup the project

include(FindPkgConfig)
project (ObjectsDetectionLib)

# ----------------------------------------------------------------------
# Site specific configurations
include(${doppia_root}/common_settings.cmake)

# ----------------------------------------------------------------------
# Setup required libraries

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
# Setup link and include directories

set(local_INCLUDE_DIRS
    "${doppia_root}/libs"
    "${doppia_root}/src"
    "${doppia_root}/src/applications/boosted_learning"
    #"/usr/include/eigen2"
    #"/usr/local/include/eigen2"
    "/users/visics/mmathias/no_backup/usr/local/include"
    "${doppia_root}/src/objects_detection/integral_channels" # to avoid doppia specific paths in the library header
    "${doppia_root}/src/objects_detection" # to avoid doppia specific paths in the library header
    "/usr/local/cuda/include"
    )

set(local_LIBRARY_DIRS
  /usr/lib64
  /usr/local/lib
  /users/visics/mmathias/no_backup/usr/local/lib
  /usr/lib64/atlas
  /usr/lib/sse2/atlas
  /usr/lib/llvm-2.8/lib
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
 "${doppia_root}/libs/cudatemplates/include"
 "${doppia_root}/src/applications/bootstrapping_lib"
  ${Boost_INCLUDE_DIRS}
  ${libpng_INCLUDE_DIRS}
  ${opencv_INCLUDE_DIRS}
  ${local_INCLUDE_DIRS}
)

if(USE_GPU)
cuda_include_directories("${doppia_root}/libs/")
endif(USE_GPU)


# ----------------------------------------------------------------------
# Collect source files

add_definitions(-DOBJECTS_DETECTION_LIB)

set(doppia_src "${doppia_root}/src")
set(doppia_stereo "${doppia_root}/src/stereo_matching")

file(GLOB SrcCpp
  "objects_detection_lib.cpp"
  "${doppia_src}/*.cpp"
  "${doppia_src}/applications/*.cpp"

  "${doppia_src}/image_processing/*.cpp"
  "${doppia_src}/drawing/gil/draw_matrix.cpp"

  "${doppia_src}/visual_odometry/Abstract*.cpp"
  "${doppia_src}/objects_tracking/*.cpp"
  "${doppia_src}/objects_tracking/motion_models/*.cpp"
  "${doppia_src}/objects_tracking/detection_descriptors/*.cpp"
  "${doppia_src}/objects_tracking/tracked_detections/*.cpp"
)

file(GLOB MonocularSrcCpp

  #"${doppia_src}/objects_detection/*.c*"
  "${doppia_src}/objects_detection/Abstract*.c*"
  "${doppia_src}/objects_detection/*Converter.c*"
  "${doppia_src}/objects_detection/Base*.c*"
  "${doppia_src}/objects_detection/*Factory.c*"
  "${doppia_src}/objects_detection/Greedy*.c*"
  "${doppia_src}/objects_detection/Detection*.c*"
  "${doppia_src}/objects_detection/*Model.c*"
  "${doppia_src}/objects_detection/*Stage.c*"
  "${doppia_src}/objects_detection/Integral*.c*"
  "${doppia_src}/objects_detection/VeryFastIntegral*.c*"
  "${doppia_src}/objects_detection/SearchRange*.c*"
  "${doppia_src}/objects_detection/MultiscalesIntegral*.c*"
  "${doppia_src}/objects_detection/FastestPedestrian*.c*"
  "${doppia_src}/objects_detection/DetectorSearchRange.c*"
  "${doppia_src}/objects_detection/*.pb.c*"
  "${doppia_src}/objects_detection/cascade_stages/*.c*"
  "${doppia_src}/objects_detection/non_maximal_suppression/*.c*"
  "${doppia_src}/objects_detection/integral_channels/Integral*.cpp"
  "${doppia_src}/objects_detection/integral_channels/AbstractIntegral*.cpp"
  "${doppia_src}/objects_detection/integral_channels/AbstractChannels*.cpp"

  "${doppia_stereo}/ground_plane/Abstract*.cpp"
  "${doppia_stereo}/stixels/Abstract*.cpp"

  "${doppia_src}/video_input/Abstract*.cpp"
  "${doppia_src}/video_input/Metric*.cpp"
  "${doppia_src}/video_input/VideoFromCam.cpp"
  "${doppia_src}/video_input/calibration/*.c*"
  "${doppia_src}/video_input/preprocessing/AddBorderFunctor.cpp"
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
  #"${doppia_stereo}/stixels/draw_stixel_world.cpp"
  #"${doppia_stereo}/objects_detection/draw_the_detections.cpp"
   #"${doppia_stereo}/stixels/*.cc"
  "${doppia_src}/video_input/*.cpp"
  "${doppia_src}/video_input/calibration/*.c*"
  "${doppia_src}/video_input/preprocessing/*Preprocessor.cpp"
  "${doppia_src}/video_input/preprocessing/*Mapper.cpp"
)


file(GLOB MonocularGuiSrcCpp
  "ObjectsDetectionLibGui.cpp"
  "${doppia_src}/drawing/gil/draw_the_detections.cpp"
  "${doppia_src}/drawing/gil/colors.cpp"
)


file(GLOB StereoGuiSrcCpp
  "ObjectsDetectionLibGui.cpp"
  "${doppia_src}/applications/objects_detection/draw*.cpp"
  "${doppia_src}/applications/stixel_world/draw*.cpp"
  "${doppia_src}/drawing/gil/*.cpp"
)

file(GLOB HelpersCpp
  "FakeObjectsDetectionLibGui.cpp"
  #"${doppia_src}/helpers/*.cpp"
  "${doppia_src}/helpers/data/*.c*"
  "${doppia_src}/helpers/any_to_string.cpp"
  "${doppia_src}/helpers/get_section_options.cpp"
  "${doppia_src}/helpers/Log.cpp"
  "${doppia_src}/helpers/ModuleLog.cpp"
  "${doppia_src}/helpers/loggers.cpp"
  "${doppia_src}/helpers/AlignedImage.cpp"
  "${doppia_src}/helpers/replace_environment_variables.cpp"
  "${doppia_src}/helpers/objects_detection/*.cpp"
  "${doppia_src}/helpers/Timers.cpp"
)


file(GLOB SrcGpuCpp

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
list(APPEND opencv_LIBRARIES opencv_imgproc opencv_gpu)

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
#set(CMAKE_CXX_COMPILER g++-4.4) # compiling with GCC4.4 is required for the EUROPA project needs

add_library(objects_detection SHARED
  ${SrcCpp} ${MonocularSrcCpp} ${StereoSrcCpp} ${HelpersCpp}
)

target_link_libraries(objects_detection
   ${Boost_LIBRARIES}
   pthread gomp
   protobuf SDL
   ${opencv_LIBRARIES}
#   SDL Xext Xrandr X11
   ${libpng_LIBRARIES} jpeg
)

if(USE_GPU)
target_link_libraries(objects_detection cuda_stuff_library ${local_CUDA_LIB})
endif(USE_GPU)

# ----------------------------------------------------------------------
# Create a version of the library with UI integrated (gui.disabled is true by default)
add_library(objects_detection_with_ui SHARED
  ${StereoGuiSrcCpp} ${SrcCpp} ${MonocularSrcCpp} ${StereoSrcCpp} ${HelpersCpp}
)

set_property(
   TARGET objects_detection_with_ui
   PROPERTY COMPILE_DEFINITIONS OBJECTS_DETECTION_WITH_UI_LIB
)

target_link_libraries(objects_detection_with_ui
   ${Boost_LIBRARIES}
   pthread gomp
   protobuf SDL
   ${opencv_LIBRARIES}
#   SDL Xext Xrandr X11
   ${libpng_LIBRARIES} jpeg
)

if(USE_GPU)
target_link_libraries(objects_detection_with_ui cuda_stuff_library ${local_CUDA_LIB})
endif(USE_GPU)

# ----------------------------------------------------------------------
# Create the monocular library
add_library(monocular_objects_detection SHARED
  ${SrcCpp} ${MonocularSrcCpp} ${HelpersCpp}
)

set_property(
   TARGET monocular_objects_detection
   PROPERTY COMPILE_DEFINITIONS MONOCULAR_OBJECTS_DETECTION_LIB
)

target_link_libraries(monocular_objects_detection
   ${Boost_LIBRARIES}
   pthread gomp
   protobuf SDL
   ${opencv_LIBRARIES}
#   SDL Xext Xrandr X11
   ${libpng_LIBRARIES} jpeg
)

if(USE_GPU)
target_link_libraries(monocular_objects_detection cuda_stuff_library ${local_CUDA_LIB})
endif(USE_GPU)


# ----------------------------------------------------------------------
# Create the monocular library with UI integrated (gui.disabled is true by default)
add_library(monocular_objects_detection_with_ui SHARED
  ${MonocularGuiSrcCpp} ${SrcCpp} ${MonocularSrcCpp} ${HelpersCpp}
)

set_property(
   TARGET monocular_objects_detection_with_ui
   PROPERTY COMPILE_DEFINITIONS MONOCULAR_OBJECTS_DETECTION_LIB OBJECTS_DETECTION_WITH_UI_LIB
)

target_link_libraries(monocular_objects_detection_with_ui
   ${Boost_LIBRARIES}
   pthread gomp
   protobuf SDL
   ${opencv_LIBRARIES}
#   SDL Xext Xrandr X11
   ${libpng_LIBRARIES} jpeg
)

if(USE_GPU)
target_link_libraries(monocular_objects_detection_with_ui cuda_stuff_library ${local_CUDA_LIB})
endif(USE_GPU)

# ----------------------------------------------------------------------
# Create the test executable
add_executable(test_objects_detection_lib
  "test_objects_detection_lib.cpp"
  "TestObjectsDetectionApplication.cpp"
)


target_link_libraries(test_objects_detection_lib objects_detection)

target_link_libraries(test_objects_detection_lib
   ${Boost_LIBRARIES}
   gomp SDL
   ${opencv_LIBRARIES}
# SDL Xext Xrandr X11
   ${libpng_LIBRARIES} jpeg
)

if(USE_GPU)
target_link_libraries(test_objects_detection_lib)
endif(USE_GPU)

# ----------------------------------------------------------------------
# Create the test executable with UI
add_executable(test_objects_detection_with_ui_lib
  "test_objects_detection_lib.cpp"
  "TestObjectsDetectionApplication.cpp"
)

target_link_libraries(test_objects_detection_with_ui_lib objects_detection_with_ui)

target_link_libraries(test_objects_detection_with_ui_lib
   ${Boost_LIBRARIES}
   gomp SDL
   ${opencv_LIBRARIES}
   SDL Xext Xrandr X11
   ${libpng_LIBRARIES} jpeg
)


if(USE_GPU)
target_link_libraries(test_objects_detection_with_ui_lib)
endif(USE_GPU)


# ----------------------------------------------------------------------
# Create the monocular test executable
add_executable(test_monocular_objects_detection_lib
  "test_objects_detection_lib.cpp"
  "TestObjectsDetectionApplication.cpp"
  "${doppia_src}/video_input/ImagesFromDirectory.cpp"
)

set_property(
   TARGET test_monocular_objects_detection_lib
   PROPERTY COMPILE_DEFINITIONS MONOCULAR_OBJECTS_DETECTION_LIB
)

target_link_libraries(test_monocular_objects_detection_lib monocular_objects_detection)

target_link_libraries(test_monocular_objects_detection_lib
   ${Boost_LIBRARIES}
   gomp
   ${opencv_LIBRARIES}
# SDL Xext Xrandr X11
   ${libpng_LIBRARIES} jpeg
)


## ----------------------------------------------------------------------
# Create the monocular test executable with UI
add_executable(test_monocular_objects_detection_with_ui_lib
  "test_objects_detection_lib.cpp"
  "TestObjectsDetectionApplication.cpp"
  "${doppia_src}/video_input/ImagesFromDirectory.cpp"
)

set_property(
   TARGET test_monocular_objects_detection_with_ui_lib
   PROPERTY COMPILE_DEFINITIONS MONOCULAR_OBJECTS_DETECTION_LIB
)

target_link_libraries(test_monocular_objects_detection_with_ui_lib monocular_objects_detection_with_ui)

target_link_libraries(test_monocular_objects_detection_with_ui_lib
   ${Boost_LIBRARIES}
   gomp
   ${opencv_LIBRARIES}
   SDL Xext Xrandr X11
   ${libpng_LIBRARIES} jpeg
)



# ----------------------------------------------------------------------

# What to install where ?

install(
TARGETS objects_detection_with_ui objects_detection monocular_objects_detection monocular_objects_detection_with_ui
LIBRARY DESTINATION lib
)

# ----------------------------------------------------------------------

