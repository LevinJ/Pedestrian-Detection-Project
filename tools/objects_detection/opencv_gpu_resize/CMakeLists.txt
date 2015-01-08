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
project (OpenCvGpuResize)

# ----------------------------------------------------------------------
# Setup the compiler options

site_name(HOSTNAME)

if(${HOSTNAME} STREQUAL  "lap-12-31")
  set(CMAKE_CXX_COMPILER "/usr/bin/g++-4.7")
  set(CMAKE_C_COMPILER "/usr/bin/g++-4.7")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") # enable C++11 support

  # since gcc 4.6 -Ofast is faster than -O3
  set(CMAKE_C_FLAGS_RELEASE "-Ofast -DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG")
  set(OPT_CXX_FLAGS "-fopenmp -ffast-math -funroll-loops -march=native")

endif()
#set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
#set(OPT_CXX_FLAGS "-fopenmp -funroll-loops -march=native")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -DBOOST_DISABLE_ASSERTS ${OPT_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

# ----------------------------------------------------------------------
# Setup required libraries

find_package(PythonLibs REQUIRED)
find_package(Boost COMPONENTS python REQUIRED)

set(NUMPY_INCLUDE_DIRS
   "/BS/mohomran-projects/work/Software/Boost.NumPy-master"
   "/home/mfritz/local/opencv_2.4.3/include"
)

include_directories(
  ${PYTHON_INCLUDE_DIRS}
  ${NUMPY_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

set(local_LIBRARY_DIRS
  "/BS/mohomran-projects/work/Software/Boost.NumPy-master/build/lib"
  "/home/mfritz/local/opencv_2.4.3/lib"
)

link_directories(
  ${local_LIBRARY_DIRS}
)


pkg_check_modules(opencv REQUIRED opencv>=2.3)

# ----------------------------------------------------------------------
# Collect source files

set(doppia_src "${doppia_root}/src")


file(GLOB SrcCpp
  "./opencv_gpu_resize.cpp"
)


# ----------------------------------------------------------------------
# Create the python module

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,/usr/local/lib") # HACK !


python_add_module(opencv_gpu_resize  ${SrcCpp})

target_link_libraries(opencv_gpu_resize
   boost_program_options-mt
   boost_serialization-mt
   boost_filesystem-mt boost_system-mt
   boost_thread-mt

   boost_python
   boost_numpy

   gomp

  ${PYTHON_LIBRARY}

  opencv_core opencv_gpu

   #${google_perftools_LIBS} # enables profiling, see http://code.google.com/p/gperftools
)

# ----------------------------------------------------------------------
