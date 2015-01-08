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
project (TestLinesDetection)


set(doppia_root "../../../")

pkg_check_modules(opencv REQUIRED opencv>=2.1)
pkg_check_modules(libpng REQUIRED libpng)

# ----------------------------------------------------------------------
set(local_INCLUDE_DIRS 
    "${doppia_root}/libs" 
    "${doppia_root}/src"
    #"/usr/include/eigen2"
    #"/usr/local/include/eigen2"
    "/users/visics/rbenenso/no_backup/usr/local/include"
    )
    
include_directories(${local_INCLUDE_DIRS})
link_directories(${libpng_LIBRARY_DIRS} ${opencv_LIBRARY_DIRS})
# ----------------------------------------------------------------------

site_name(HOSTNAME)

# could use CMAKE_SYSTEM_PROCESSOR to define the optimization flags automagically
if (${HOSTNAME} STREQUAL "vesta")
  message(STATUS "Using vesta optimisation options")
  set(OPT_CXX_FLAGS "-O3 -fopenmp -funroll-loops --fast-math -mtune=core2 -mfpmath=sse -mssse3")
  # no optimizations when debugging

else ()
  message(STATUS "Using core2 optimisation options")
  set(OPT_CXX_FLAGS "-O3 -fopenmp -funroll-loops --fast-math -mtune=core2 -mfpmath=sse -mssse3")

endif ()

# ----------------------------------------------------------------------

file(GLOB SrcCpp  "./*.*pp" 
                    #"${doppia_root}/src/moving_objects_detection/tracks_categorisation/*.*pp" 
                    #"${doppia_root}/src/model_parameters_estimation/*.*pp"   
                    #"${doppia_root}/src/features_tracking/TrackedPoint.*pp"
		    "${doppia_root}/src/image_processing/Abstract*.*pp"
		    "${doppia_root}/src/image_processing/Open*.*pp"
                    "${doppia_root}/src/image_processing/Irls*.*pp"
                    )

# ----------------------------------------------------------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_TEST_DYN_LINK -Wall -W -g -p ${OPT_CXX_FLAGS}")
add_executable (test_lines_detection ${SrcCpp} )

target_link_libraries (test_lines_detection  
${opencv_LIBRARIES} ${libpng_LIBRARIES}
boost_unit_test_framework-mt
boost_program_options-mt
boost_filesystem-mt
gomp)

# ----------------------------------------------------------------------
