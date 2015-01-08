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
project (TestStixelsEstimation)


set(doppia_root "../../../")

pkg_check_modules(opencv REQUIRED opencv>=2.1)
pkg_check_modules(libpng REQUIRED libpng)

# ----------------------------------------------------------------------
set(local_INCLUDE_DIRS 
    "${doppia_root}/libs" 
    "${doppia_root}/src"
    "/usr/include/eigen2"
    "/usr/local/include/eigen2"
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

set(doppia_src "${doppia_root}/src")
set(doppia_stereo "${doppia_root}/src/stereo_matching")

file(GLOB SrcCpp  "./*.*pp" 
                 "${doppia_stereo}/*CostVolume.cpp"
                 "${doppia_stereo}/CensusTransform.cpp"
                 "${doppia_stereo}/CensusCostFunction.cpp"
                 "${doppia_stereo}/cost_functions.cpp"
                 "${doppia_stereo}/ground_plane/*.cpp"
                 "${doppia_stereo}/stixels/*.cpp"
                 "${doppia_src}/video_input/calibration/*.c*"
                 "${doppia_src}/video_input/preprocessing/*.cpp"
                 "${doppia_src}/video_input/Metric*Camera.cpp"
                 "${doppia_src}/video_input/AbstractVideoInput.cpp"
                 "${doppia_src}/image_processing/*.cpp"
                 "${doppia_src}/drawing/gil/draw_matrix.cpp"
                 "${doppia_src}/drawing/gil/colors.cpp"
                    #"${doppia_root}/src/moving_objects_detection/tracks_categorisation/*.*pp"
                    #"${doppia_root}/src/model_parameters_estimation/*.*pp"   
                    #"${doppia_root}/src/features_tracking/TrackedPoint.*pp"
                    #"${doppia_root}/src/image_processing/Abstract*.*pp"
                    #"${doppia_root}/src/image_processing/Open*.*pp"
                    )

file(GLOB HelpersCpp
  #"${doppia_src}/helpers/*.cpp"
  "${doppia_src}/helpers/any_to_string.cpp"
  "${doppia_src}/helpers/get_section_options.cpp"
  "${doppia_src}/helpers/Log.cpp"
  "${doppia_src}/helpers/loggers.cpp"
)

# ----------------------------------------------------------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_TEST_DYN_LINK -Wall -W -g -p ${OPT_CXX_FLAGS}")
add_executable (test_stixels_estimation ${SrcCpp}  ${HelpersCpp})

target_link_libraries (test_stixels_estimation
${opencv_LIBRARIES} ${libpng_LIBRARIES}
boost_unit_test_framework-mt
boost_program_options-mt boost_filesystem-mt boost_system-mt
boost_thread-mt
gomp protobuf pthread
)

# ----------------------------------------------------------------------
