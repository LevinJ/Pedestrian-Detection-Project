# This is a CMake build file, for more information consult:
# http://en.wikipedia.org/wiki/CMake
# and
# http://www.cmake.org/Wiki/CMake
# http://www.cmake.org/cmake/help/syntax.html
# http://www.cmake.org/Wiki/CMake_Useful_Variables
# http://www.cmake.org/cmake/help/cmake-2-8-docs.html

# to compile the local code you can use: cmake ./ && make -j2

cmake_minimum_required (VERSION 2.6)

include(FindPkgConfig)
project (TestDataSequence)

set(doppia_root "../../../")

# ----------------------------------------------------------------------
set(local_INCLUDE_DIRS 
    "${doppia_root}/libs" 
    "${doppia_root}/src"
    "/users/visics/rbenenso/no_backup/usr/local/include"
    )
    
include_directories(${local_INCLUDE_DIRS})
link_directories(${libpng_LIBRARY_DIRS} ${opencv_LIBRARY_DIRS})
# ----------------------------------------------------------------------

site_name(HOSTNAME)

# could use CMAKE_SYSTEM_PROCESSOR to define the optimization flags automagically
if (${HOSTNAME} STREQUAL "vesta")
  message(STATUS "Using vesta optimisation options")
  #set(OPT_CXX_FLAGS "-O3 -fopenmp -funroll-loops --fast-math -mtune=core2 -mfpmath=sse -mssse3")
  # no optimizations when debugging

else ()
  message(STATUS "Using core2 optimisation options")
  #set(OPT_CXX_FLAGS "-O3 -fopenmp -funroll-loops --fast-math -mtune=core2 -mfpmath=sse -mssse3")

endif ()

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


link_directories(${local_LIBRARY_DIRS})
include_directories("${doppia_root}/libs/" "${doppia_root}/src" ${local_INCLUDE_DIRS})

# ----------------------------------------------------------------------

set(doppia_src "${doppia_root}/src")

file(GLOB SrcCpp  "./*.c*")

file(GLOB HelpersCpp
  #"${doppia_src}/helpers/*.cpp"
  "${doppia_src}/helpers/data/*.c*" 
  #"${doppia_src}/helpers/any_to_string.cpp"
  #"${doppia_src}/helpers/get_section_options.cpp"
  #"${doppia_src}/helpers/Log.cpp"
  #"${doppia_src}/helpers/loggers.cpp"
)

# ----------------------------------------------------------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_TEST_DYN_LINK -Wall -W -g -p ${OPT_CXX_FLAGS}")
add_executable (test_data_sequence ${SrcCpp}  ${HelpersCpp})

target_link_libraries (test_data_sequence
boost_unit_test_framework-mt
boost_program_options-mt boost_filesystem-mt boost_system-mt
boost_thread-mt
gomp protobuf pthread
)

# ----------------------------------------------------------------------
