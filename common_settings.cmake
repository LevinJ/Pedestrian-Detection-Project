# This is a CMake build file, for more information consult:
# http://en.wikipedia.org/wiki/CMake
# and
# http://www.cmake.org/Wiki/CMake
# http://www.cmake.org/cmake/help/syntax.html
# http://www.cmake.org/Wiki/CMake_Useful_Variables
# http://www.cmake.org/cmake/help/cmake-2-8-docs.html

# This file is intended to be included by other cmake files (see src/applications/*/CMakeLists.txt)

# ----------------------------------------------------------------------

site_name(HOSTNAME)

# Use "gcc -v -Q -march=native -O3 test.c -o test" to see which options the compiler actualy uses
# using -march=native will include all sse options available on the given machine (-msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2, etc...)
# also using -march=native will imply -mtune=native
# Thus the optimization flags below should work great on all machines
# (O3 is already added by CMAKE_CXX_FLAGS_RELEASE)
set(OPT_CXX_FLAGS "-fopenmp -ffast-math -funroll-loops -march=native")
#set(OPT_CXX_FLAGS "-fopenmp -ffast-math -funroll-loops -march=native -freciprocal-math -funsafe-math-optimizations -fassociative-math -ffinite-math-only -fcx-limited-range")  # cheap -Ofast copy
#set(OPT_CXX_FLAGS "-ffast-math -funroll-loops -march=native") # disabled OpenMp, just for testing
#set(OPT_CXX_FLAGS "-fopenmp -ffast-math -funroll-loops") # disable native compilation, so we can profile with older versions of valgrind/callgrind

# enable link time optimization
# http://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
#set(OPT_CXX_FLAGS "${OPT_CXX_FLAGS} -flto")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")
#set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -flto")
#set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -flto")

#option(CUDA_HOST_SHARED_FLAGS OFF) # cuda/nvcc uses a separate set of options than gcc

set(VISICS_MACHINES
  "vesta" "nereid" # intel vpro
  "enif" # core 2 quad
  "kochab" "oculus"  "izar" "yildun" "watar" "sadr" # intel v7, core i7 860  @ 2.80GHz
  "unuk" # intel v7, cire i7 870 @ 2.93GHz
  "jabbah" "matar" # top of the line cpu and gpu
  "kochab.esat.kuleuven.be"
)

set(D2_GPU_MACHINES
  "wks-12-31" "wks-12-32" "wks-12-33"             # new high end work stations
  "wks-12-44" "wks-12-45" "wks-12-46" "wks-12-47" # new high end work stations
  "wks-12-23"
  "ruegen"
  "ganymede"
  "menorca"
)

set(D2_NO_GPU_MACHINES
  "titan"
  "triton"
  "d2kinect0"
  "infno7900"
  "wks-12-34"
)

list(FIND VISICS_MACHINES ${HOSTNAME} HOSTED_AT_VISICS)
#message(STATUS "HOSTED_AT_VISICS == ${HOSTED_AT_VISICS}")

list(FIND D2_GPU_MACHINES ${HOSTNAME} HOSTED_AT_D2_GPU)
list(FIND D2_NO_GPU_MACHINES ${HOSTNAME} HOSTED_AT_D2_NO_GPU)

if(HOSTED_AT_VISICS GREATER -1)
  message(STATUS "Using ${VISICS_MACHINES} optimisation options")

  # since gcc 4.6 the option -Ofast provides faster than -O3
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Dint_p_NULL='\(\(int*\)0\)'")
  #set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  #set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG")

  # add local compiled opencv trunk in the pkg-config paths
  #set(PKG_CONFIG_PATH ${PKG_CONFIG_PATH}:/users/visics/rbenenso/no_backup/usr/local/lib/pkgconfig)

  #set(OPT_CXX_FLAGS "-fopenmp -march=native -mtune=native -ffast-math -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2") # just for testing

  option(USE_GPU "Should the GPU be used ?" ON)
  #option(USE_GPU "Should the GPU be used ?" OFF) # set to false for testing purposes only
  set(CUDA_BUILD_EMULATION OFF CACHE BOOL "enable emulation mode")
  set(CUDA_BUILD_CUBIN OFF)
  set(local_CUDA_CUT_INCLUDE_DIRS "/usr/local/cuda-5.5/include")
  set(local_CUDA_CUT_LIBRARY_DIRS "/usr/local/cuda-5.5/lib")
  #set(local_CUDA_CUT_INCLUDE_DIRS "/users/visics/rbenenso/code/references/cuda/cuda_sdk/C/common/inc")
  #set(local_CUDA_CUT_LIBRARY_DIRS "/users/visics/rbenenso/code/references/cuda/cuda_sdk/C/lib")
  set(local_CUDA_LIB_DIR "/usr/lib64/nvidia")
  set(local_CUDA_LIB "/usr/lib64/nvidia/libcuda.so")
  set(cuda_LIBS "cuda")
  #set(cutil_LIB "cutil")

  # if you get error messages in nvcc-generated files,
  # enable the following line for debugging:
  #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--keep")
  #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --host-compilation c++ --device-compilation c++)
  #set(CUDA_NVCC_EXECUTABLE  /users/visics/rbenenso/code/references/cuda/gcc-4.4/nvcc-4.4.sh)
  set(CUDA_NVCC_EXECUTABLE  /usr/local/cuda-5.5/bin/nvcc )
  #set(CUDA_SDK_ROOT_DIR  /users/visics/rbenenso/code/references/cuda_sdk_4.0.17/C)
  set(CUDA_SDK_ROOT_DIR  /users/visics/mmathias/devel/doppia/)

  set(CUDA_NVCC_FLAGS "-arch=sm_20" CACHE STRING "nvcc flags" FORCE) # only matar, jabbah and yildun can run current code

  #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --compiler-options -D__USE_XOPEN2K8 --pre-include /users/visics/mmathias/devel/doppia/undef_atomics_int128.h) # black magic required on Visics machines


  # faster malloc, and a good profiler via http://google-perftools.googlecode.com
  #set(google_perftools_LIBS tcmalloc profiler)
  set(google_perftools_LIBS tcmalloc_and_profiler)

  #set(liblinear_INCLUDE_DIRS "/users/visics/rbenenso/code/references/machine_learning/liblinear-1.8")
  #set(liblinear_LIBRARY_DIRS "/users/visics/rbenenso/code/references/machine_learning/liblinear-1.8")
  #add_definitions(" -Dint_p_NULL=((int*)0) ")
  #add_definitions(" -Dint_p_NULL=(int*)NULL ")

elseif(${HOSTNAME} STREQUAL  "biwi-desktop")
  message(STATUS "Using biwi-desktop optimisation options")

  option(USE_GPU "Should the GPU be used ?" TRUE)
  set(CUDA_BUILD_EMULATION OFF CACHE BOOL "enable emulation mode")
  set(CUDA_BUILD_CUBIN OFF)
  set(GCC43_DIRECTORY "/home/biwi/aess/aess/projects/3rdparty/gcc-4.3/")
  set(local_CUDA_CUT_INCLUDE_DIRS "/home/biwi/NVIDIA_GPU_Computing_SDK/C/common/inc")
  set(local_CUDA_CUT_LIBRARY_DIRS "/home/biwi/NVIDIA_GPU_Computing_SDK/C/lib")
  set(local_CUDA_LIB_DIR "/usr/lib")
  set(cuda_LIBS "cuda")
  set(cutil_LIB "cutil")


elseif(${HOSTNAME} STREQUAL  "mmp-laptop")
  # Aachen @ Europa computer
  message(STATUS "Using mmp Aaechen @ Europa optimisation options")

  option(USE_GPU "Should the GPU be used ?" TRUE)
  set(CUDA_BUILD_EMULATION OFF CACHE BOOL "enable emulation mode")
  set(GCC43_DIRECTORY "/usr/bin")
  set(local_CUDA_CUT_INCLUDE_DIRS "/home/mmp/NVIDIA_GPU_Computing_SDK/C/common/inc")
  set(local_CUDA_CUT_LIBRARY_DIRS "/home/mmp/NVIDIA_GPU_Computing_SDK/C/lib")
  set(local_CUDA_LIB_DIR "/usr/lib")
  set(cuda_LIBS "cuda")
  set(cutil_LIB "cutil")


elseif(${HOSTNAME} STREQUAL  "sammy")
  # Sammy @ IURO TUM ACE robot computer
  message(STATUS "Using sammy @ IURO optimisation options")

  option(USE_GPU "Should the GPU be used ?" TRUE)
  set(CUDA_BUILD_EMULATION OFF CACHE BOOL "enable emulation mode")
  set(CUDA_BUILD_CUBIN OFF)
  set(GCC43_DIRECTORY "/usr/bin")
  set(local_CUDA_CUT_INCLUDE_DIRS "/home/sammy/cuda/NVIDIA_GPU_Computing_SDK/C/common/inc")
  set(local_CUDA_CUT_LIBRARY_DIRS "/home/sammy/cuda/NVIDIA_GPU_Computing_SDK/C/lib")
  set(local_CUDA_LIB_DIR "/usr/local/cuda-5.5/lib")
  set(cuda_LIBS "cuda")
  set(cutil_LIB "cutil_x86_64")


elseif(${HOSTNAME} STREQUAL  "rodrigob-laptop")
  message(STATUS "Using rodrigob-laptop optimisation options")

  option(USE_GPU "Should the GPU be used ?" FALSE)
  set(CUDA_BUILD_EMULATION ON CACHE BOOL "enable emulation mode")
  set(CUDA_BUILD_CUBIN OFF)
  set(local_CUDA_CUT_INCLUDE_DIRS "/home/rodrigob/work/code/doppia_references/cuda/cuda_sdk/C/common/inc")
  set(local_CUDA_CUT_LIBRARY_DIRS "/home/rodrigob/work/code/doppia_references/cuda/cuda_sdk/C/lib")
  set(cuda_LIBS "")
  set(cutil_LIB "cutil")

  set(GCC44_DIRECTORY "/home/rodrigob/work/code/doppia_references/cuda/gcc-4.4/")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-bindir ${GCC44_DIRECTORY}" CACHE STRING "nvcc flags" FORCE)

  # faster malloc, and a good profiler via http://google-perftools.googlecode.com
  set(google_perftools_LIBS tcmalloc profiler)
  set(EUROPA_SVN "/home/rodrigob/work/code/europa_svn/code")

  set(liblinear_INCLUDE_DIRS "/home/rodrigob/work/code/doppia_references/liblinear-1.8")
  set(liblinear_LIBRARY_DIRS "/home/rodrigob/work/code/doppia_references/liblinear-1.8")


elseif(${HOSTNAME} STREQUAL  "visics-gt680r")
  message(STATUS "Using visics-gt680r optimisation options")

  option(USE_GPU "Should the GPU be used ?" TRUE)
  #set(CUDA_BUILD_EMULATION OFF CACHE BOOL "enable emulation mode")
  set(CUDA_BUILD_CUBIN OFF)

  # work around to use gcc-4.4 instead of 4.5
  #set(CUDA_NVCC_EXECUTABLE "/home/rodrigob/code/references/cuda/gcc-4.4/nvcc-4.4.sh")
  set(CUDA_NVCC_FLAGS "-arch=sm_21" CACHE STRING "nvcc flags" FORCE)

  set(local_CUDA_CUT_INCLUDE_DIRS "/home/rodrigob/code/references/cuda/cuda_sdk/C/common/inc")
  set(local_CUDA_CUT_LIBRARY_DIRS "/home/rodrigob/code/references/cuda/cuda_sdk/C/lib")
  set(local_CUDA_LIB_DIR "/usr/local/cuda-5.5/lib64")
  set(cuda_LIBS "")
  set(cutil_LIB "cutil")

  # faster malloc, and a good profiler via http://google-perftools.googlecode.com
  set(google_perftools_LIBS tcmalloc profiler)
  set(EUROPA_SVN "/home/rodrigob/code/europa_svn/code")

  set(liblinear_INCLUDE_DIRS "/home/rodrigob/code/references/liblinear-1.8")
  set(liblinear_LIBRARY_DIRS "/home/rodrigob/code/references/liblinear-1.8")

elseif(${HOSTNAME} STREQUAL  "lap-12-31")
  message(STATUS "Using lap-12-31 optimisation options")

  set(CMAKE_CXX_COMPILER "/usr/bin/g++-4.7")
  set(CMAKE_C_COMPILER "/usr/bin/g++-4.7")
  # since gcc 4.6 -Ofast is faster than -O3
  set(CMAKE_C_FLAGS_RELEASE "-Ofast -DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG")

  set(CUDA_PROPAGATE_HOST_FLAGS "OFF" CACHE STRING "nvcc propagte host flags" FORCE) # since nvcc does not handle -Ofast
  set(CUDA_NVCC_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "nvcc release flags" FORCE)
  set(CUDA_NVCC_FLAGS_RELWITHDEBINFO "-O3 -DNDEBUG" CACHE STRING "nvcc relwithdebinfo flags" FORCE)

  # add local compiled opencv trunk in the pkg-config paths
  #set(PKG_CONFIG_PATH ${PKG_CONFIG_PATH}:/home/mfritz/local/lib/pkgconfig)
  #set(PKG_CONFIG_PATH /home/mfritz/local/lib/pkgconfig)

  set(opencv_INCLUDE_DIRS "/usr/local/include/opencv2")
  set(opencv_LIBRARY_DIRS "/usr/local/lib")

  option(USE_GPU "Should the GPU be used ?" TRUE)
  #set(CUDA_BUILD_EMULATION OFF CACHE BOOL "enable emulation mode")
  set(CUDA_BUILD_CUBIN OFF)

  # work around to use gcc-4.4 instead of 4.5
  #set(CUDA_NVCC_EXECUTABLE "/home/rodrigob/code/references/cuda/gcc-4.4/nvcc-4.4.sh")
  set(CUDA_NVCC_FLAGS "-arch=sm_30" CACHE STRING "nvcc flags" FORCE)

  set(local_CUDA_CUT_INCLUDE_DIRS "/home/rodrigob/code/references/cuda/cuda_sdk/C/common/inc")
  set(local_CUDA_CUT_LIBRARY_DIRS "/home/rodrigob/code/references/cuda/cuda_sdk/C/lib")
  set(local_CUDA_LIB_DIR "/usr/local/cuda/lib64")
  set(cuda_LIBS "")
  #set(cutil_LIB "cutil")
  set(cutil_LIB "") # no cutil

  # faster malloc, and a good profiler via http://google-perftools.googlecode.com
  set(google_perftools_LIBS tcmalloc profiler)
  set(EUROPA_SVN "/home/rodrigob/code/europa_svn/code")

  set(liblinear_INCLUDE_DIRS "/home/rodrigob/code/references/liblinear-1.8")
  set(liblinear_LIBRARY_DIRS "/home/rodrigob/code/references/liblinear-1.8")

  include_directories(
   # "/home/rodrigob/code/references/libwebp-0.3.0-linux-x86-64/include"
  )

  link_directories(
   # "/home/rodrigob/code/references/libwebp-0.3.0-linux-x86-64/lib"
  )

elseif(HOSTED_AT_D2_GPU GREATER -1)

  message(STATUS "Using d2 with gpu optimisation options")

  # add local compiled opencv trunk in the pkg-config paths
  #set(PKG_CONFIG_PATH ${PKG_CONFIG_PATH}:/home/mfritz/local/lib/pkgconfig)
  set(PKG_CONFIG_PATH /home/mfritz/local/lib/pkgconfig)

  option(USE_GPU "Should the GPU be used ?" TRUE)
  #set(CUDA_BUILD_EMULATION OFF CACHE BOOL "enable emulation mode")
  set(CUDA_BUILD_CUBIN OFF)

  # work around to use gcc-4.4 instead of 4.5
  #set(CUDA_NVCC_EXECUTABLE "/home/rodrigob/code/references/cuda/gcc-4.4/nvcc-4.4.sh")
  set(CUDA_NVCC_FLAGS "-arch=sm_30" CACHE STRING "nvcc flags" FORCE)

  set(local_CUDA_CUT_INCLUDE_DIRS "/home/rodrigob/code/references/cuda/cuda_sdk/C/common/inc")
  set(local_CUDA_CUT_LIBRARY_DIRS "/home/rodrigob/code/references/cuda/cuda_sdk/C/lib")
  set(local_CUDA_LIB_DIR "/usr/lib/x86_64-linux-gnu/")
  #set(local_CUDA_LIB_DIR "/usr/lib64/")
  set(cuda_LIBS "")
  #set(cutil_LIB "cutil")
  set(cutil_LIB "") # no cutil

  include_directories(
    /BS/mohomran-projects/work/Software/protobuf-2.5.0/build/include
    #/home/mfritz/local/include
  )

  link_directories(
    /BS/mohomran-projects/work/Software/protobuf-2.5.0/build/lib
    #/home/mfritz/local/lib
  )

  # faster malloc, and a good profiler via http://google-perftools.googlecode.com
  set(google_perftools_LIBS tcmalloc profiler)
  set(EUROPA_SVN "/home/rodrigob/code/europa_svn/code")

  set(liblinear_INCLUDE_DIRS "/home/rodrigob/code/references/liblinear-1.8")
  set(liblinear_LIBRARY_DIRS "/home/rodrigob/code/references/liblinear-1.8")

  if(USE_GPU)
    #set(opencv_INCLUDE_DIRS "/home/local/opencv_2.4.3/include/")
    #set(opencv_LIBRARY_DIRS "/home/mfritz/local/opencv_2.4.3/lib/")

    set(opencv_INCLUDE_DIRS "/BS/mohomran-projects/work/Software/opencv-2.4.9/install/include")
    set(opencv_LIBRARY_DIRS "/BS/mohomran-projects/work/Software/opencv-2.4.9/install/lib")
  else()
    message("Using no gpu opencv on a GPU capable machine")
    set(opencv_INCLUDE_DIRS "/home/benenson/projects/work/local/opencv_2.4.3_no_gpu/include")
    set(opencv_LIBRARY_DIRS "/home/benenson/projects/work/local/opencv_2.4.3_no_gpu/lib")
  endif()


elseif(HOSTED_AT_D2_NO_GPU GREATER -1)
  message(STATUS "Using d2 no gpu optimisation options")

  # add local compiled opencv trunk in the pkg-config paths
  #set(PKG_CONFIG_PATH ${PKG_CONFIG_PATH}:/home/mfritz/local/lib/pkgconfig)
  set(PKG_CONFIG_PATH /home/mfritz/local/lib/pkgconfig)

  option(USE_GPU "Should the GPU be used ?" FALSE)
  #set(CUDA_BUILD_EMULATION OFF CACHE BOOL "enable emulation mode")
  set(CUDA_BUILD_CUBIN OFF)

  set(opencv_INCLUDE_DIRS "/home/benenson/projects/work/local/opencv_2.4.3_no_gpu/include")
  set(opencv_LIBRARY_DIRS "/home/benenson/projects/work/local/opencv_2.4.3_no_gpu/lib")

  #set(opencv_INCLUDE_DIRS "/BS/mohomran-projects/work/Software/opencv-2.4.1/install/include")
  #set(opencv_LIBRARY_DIRS "/BS/mohomran-projects/work/Software/opencv-2.4.1/install/lib")

  include_directories(
    /home/mfritz/software/protobuf-read-only
    /home/mfritz/local/include
    /home/benenson/code/references/libwebp-0.3.0-linux-x86-64/include
  )

  link_directories(
    /home/mfritz/local/lib
    /home/benenson/code/references/libwebp-0.3.0-linux-x86-64/lib
  )

  # faster malloc, and a good profiler via http://google-perftools.googlecode.com
  set(google_perftools_LIBS tcmalloc profiler)

  set(liblinear_INCLUDE_DIRS "/home/rodrigob/code/references/liblinear-1.8")
  set(liblinear_LIBRARY_DIRS "/home/rodrigob/code/references/liblinear-1.8")

elseif(${HOSTNAME} STREQUAL  "the_name_of_your_machine")
  # change the_name_of_your_machine to what /bin/hostname returns

  message(STATUS "Using the_name_of_your_machine compilation options")
  # start with an empty section, and see what fails as you go through the readme.text instructions

else ()
  message(FATAL_ERROR, "Unknown machine, please add your configuration inside doppia/common_settings.cmake")
  # see the elseif section above

endif ()

# ----------------------------------------------------------------------
# enable compilation for shared libraries
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fpic" CACHE STRING "nvcc flags" FORCE)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  # enable cuda debug information, to use with cuda-dbg
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G" CACHE STRING "nvcc flags" FORCE)
else()
# FIXME disabled only for testing
#  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 --use_fast_math"  CACHE STRING "nvcc flags" FORCE) # speed up host and device code
endif()

# ----------------------------------------------------------------------
# set default compilation flags and default build

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g") # add debug information, even in release mode
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -DNDEBUG -DBOOST_DISABLE_ASSERTS ${OPT_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")


if(USE_GPU)
  add_definitions(-DUSE_GPU)
endif(USE_GPU)


# set default cmake build type (None Debug Release RelWithDebInfo MinSizeRel)
if( NOT CMAKE_BUILD_TYPE )
   set( CMAKE_BUILD_TYPE "Release" )
endif()

# ----------------------------------------------------------------------
