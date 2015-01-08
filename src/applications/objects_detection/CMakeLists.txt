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
set(CMAKE_MODULE_PATH "/home/rodrigob/work/code/doppia_references/cuda/FindCUDA/CMake/cuda" ${CMAKE_MODULE_PATH})
#set(CMAKE_MODULE_PATH "/users/visics/mmathias/no_backup/code/references/cuda/FindCUDA/CMake/cuda" ${CMAKE_MODULE_PATH})

# ----------------------------------------------------------------------
# Setup the project

include(FindPkgConfig)
project (ObjectsDetection)

# ----------------------------------------------------------------------
# Site specific configurations
include(${doppia_root}/common_settings.cmake)

# ----------------------------------------------------------------------
# Setup required libraries

pkg_check_modules(libpng REQUIRED libpng)
#pkg_check_modules(OpenEXR REQUIRED OpenEXR)
pkg_check_modules(opencv REQUIRED opencv>=2.3)
#set(vw_LIBRARIES "-lvwCore -lvwImage -lvwStereo -lvwFileIO -lvwMath -lvwInterestPoint")

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
if(USE_GPU)
  find_package(CUDA 4.0 REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_CUT_INCLUDE_DIR})
endif(USE_GPU)

# ----------------------------------------------------------------------
# Setup link and include directories

set(local_LIBRARY_DIRS
  "/usr/local/lib"
  "/users/visics/mmathias/no_backup/usr/local/lib"
  "/usr/lib64"
  "/usr/lib64/atlas"
  "/usr/lib/sse2/atlas"
  "/usr/lib/llvm-2.8/lib"
  ${local_CUDA_LIB_DIR}
)
set(local_INCLUDE_DIRS
  "/users/visics/mmathias/no_backup/usr/local/include"
  "/usr/include/eigen2/"
   "/usr/local/include/eigen2"
   "/usr/local/cuda-5.5/include"
   ${CUDA_INCLUDE_DIRS}
)

link_directories(
  ${libpng_LIBRARY_DIRS}
  ${OpenEXR_LIBRARY_DIRS}
  ${opencv_LIBRARY_DIRS}
  ${Boost_LIBRARY_DIRS}
  ${local_LIBRARY_DIRS}
)

include_directories(
  "${doppia_root}/libs"
  "${doppia_root}/src"
  ${libpng_INCLUDE_DIRS}
  ${OpenEXR_INCLUDE_DIRS}
  ${opencv_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS} 
  ${local_INCLUDE_DIRS}
  "${doppia_root}/libs/cudatemplates/include"
)

if(USE_GPU)
cuda_include_directories("${doppia_root}/libs/")
endif(USE_GPU)

# ----------------------------------------------------------------------
# Collect source files

set(doppia_src ${doppia_root}/src)
set(doppia_stereo ${doppia_root}/src/stereo_matching)

file(GLOB SrcCpp
  "./ObjectsDetection*.cpp"
  "./draw*.cpp"
  "${doppia_src}/*.cpp"
  #"${doppia_src}/objects_detection/*.c*"
  "${doppia_src}/objects_detection/Abstract[^G]*.c*"
  "${doppia_src}/objects_detection/*Converter.c*"
  "${doppia_src}/objects_detection/Base*.c*"
  "${doppia_src}/objects_detection/*Factory.c*"
  "${doppia_src}/objects_detection/Greedy*.c*"
  "${doppia_src}/objects_detection/Detection*.c*"
  "${doppia_src}/objects_detection/*Model.c*"
  "${doppia_src}/objects_detection/*Stage.c*"
  "${doppia_src}/objects_detection/Integral*.c*"
  "${doppia_src}/objects_detection/[^(Gpu)]*Integral*.c*"
  "${doppia_src}/objects_detection/SearchRange*.c*"
  "${doppia_src}/objects_detection/MultiscalesIntegral*.c*"
  "${doppia_src}/objects_detection/integral_channels/AbstractChannelsComputer.cpp"
  "${doppia_src}/objects_detection/integral_channels/AbstractIntegralChannelsComputer.cpp"
  "${doppia_src}/objects_detection/integral_channels/[^(Gpu)|(AbstractGpu)]*.cpp"

  "${doppia_src}/objects_detection/FastestPedestrian*.c*"
  "${doppia_src}/objects_detection/DetectorSearchRange.c*"
  "${doppia_src}/objects_detection/*.pb.c*"
  "${doppia_src}/objects_detection/cascade_stages/*.c*"
  "${doppia_src}/objects_detection/non_maximal_suppression/*.c*"

  "${doppia_src}/objects_tracking/*.cpp"

  "${doppia_src}/applications/*.cpp"
  "${doppia_src}/applications/stixel_world/*Gui.cpp"
  "${doppia_src}/applications/objects_tracking/*Application.cpp"
  "${doppia_src}/visual_odometry/Abstract*.cpp"

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

  "${doppia_src}/objects_detection/Detection2d.cpp"
  "${doppia_src}/objects_tracking/tracked_detections/TrackedDetection2d.cpp"

  "${doppia_src}/video_input/*.cpp"
  "${doppia_src}/video_input/calibration/*.c*"
  "${doppia_src}/video_input/preprocessing/*.cpp"
  #"${doppia_src}/features_tracking/*.cpp"
  "${doppia_src}/image_processing/*.cpp"

  "${doppia_src}/drawing/gil/*.cpp"
)


file(GLOB HelpersCpp
  #"${doppia_src}/helpers/*.cpp"
  "${doppia_src}/helpers/data/*.c*"
  "${doppia_src}/helpers/any_to_string.cpp"
  "${doppia_src}/helpers/get_section_options.cpp"
  "${doppia_src}/helpers/Log.cpp"
  "${doppia_src}/helpers/ModuleLog.cpp"
  "${doppia_src}/helpers/AlignedImage.cpp"
  "${doppia_src}/helpers/replace_environment_variables.cpp"
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
  "${doppia_src}/objects_detection/integral_channels/gpu/*.cu"
  "${doppia_src}/objects_detection/integral_channels/gpu/*.cpp"
  "${doppia_src}/objects_detection/gpu/*.cu"
  "${doppia_src}/objects_detection/gpu/*.cpp"

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
   ${cutil_LIB}
)

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --generate-line-info) # used during profiling

endif(USE_GPU)
# ----------------------------------------------------------------------
# Create the executable
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x") # required for unrestricted unions
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -p") # add gprof information

add_library(cpp_stuff_library  ${SrcCpp} ${HelpersCpp})

add_executable(objects_detection ./objects_detection.cpp)

target_link_libraries(objects_detection

   cpp_stuff_library

   ${Boost_LIBRARIES}
   protobuf pthread
   SDL X11 Xext #Xrandr
   gomp
   ${libpng_LIBRARIES} jpeg
#  ${OpenEXR_LIBRARIES}
   ${opencv_LIBRARIES}

   #${vw_LIBRARIES}
   #csparse sparse spblas mv
   #lapack blas atlas

   #${google_perftools_LIBS} # enables profiling, see http://code.google.com/p/gperftools

   #`OcelotConfig -l`
   #ocelot
   #boost_system boost_filesystem boost_thread
   #GLEW
   #LLVMAsmParser LLVMX86Disassembler LLVMX86AsmParser LLVMX86CodeGen LLVMSelectionDAG
   #LLVMAsmPrinter LLVMMCParser LLVMX86AsmPrinter LLVMX86Info LLVMJIT
   #LLVMExecutionEngine LLVMCodeGen LLVMScalarOpts LLVMInstCombine LLVMTransformUtils LLVMipa
   #LLVMAnalysis LLVMTarget LLVMMC LLVMCore LLVMSupport LLVMSystem
)

if(USE_GPU)
target_link_libraries(objects_detection cuda_stuff_library ${local_CUDA_LIB})
endif(USE_GPU)
# ----------------------------------------------------------------------
