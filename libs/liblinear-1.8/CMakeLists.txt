# This is a CMake build file, for more information consult:
# http://en.wikipedia.org/wiki/CMake
# and
# http://www.cmake.org/Wiki/CMake
# http://www.cmake.org/cmake/help/syntax.html
# http://www.cmake.org/Wiki/CMake_Useful_Variables
# http://www.cmake.org/cmake/help/cmake-2-8-docs.html

# to compile the local code you can use: cmake ./ && make -j2

cmake_minimum_required (VERSION 2.6)

set(doppia_root "../../")

include(FindPkgConfig)
project (LibLinear)

# ----------------------------------------------------------------------
# Site specific configurations
include(${doppia_root}/common_settings.cmake)

# ----------------------------------------------------------------------
set(local_INCLUDE_DIRS 
    )
    
include_directories(
)

link_directories(
)


# ----------------------------------------------------------------------
# Collect source files

file(GLOB SrcCpp 
  "tron.cpp"
  "linear.cpp"
)


file(GLOB BlasSrc
  "blas/*.c"
)


# ----------------------------------------------------------------------
# Create the libraries

add_library( blas SHARED ${BlasSrc} )


add_library( linear SHARED ${SrcCpp} )


target_link_libraries( linear
  blas
)


# ----------------------------------------------------------------------
# Create the test executable
add_executable(train "train.c")
target_link_libraries(train linear blas)

add_executable(predict "predict.c")
target_link_libraries(predict linear blas)
# ----------------------------------------------------------------------
