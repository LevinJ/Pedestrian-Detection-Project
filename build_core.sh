#!/bin/sh
# Do not use this script unless you know what you are doing,
# start by reading readme.text to know which are the first step to do get this code base to compile.

set -x  # will print excecuted commands

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# boosted learning component
BUILDDIR="/src/applications/boosted_learning"
cd $DIR$BUILDDIR
make clean
cmake -D CMAKE_BUILD_TYPE=RelWithDebInfo ./
make


# object detection component
BUILDDIR="/src/applications/objects_detection/build"
cd $DIR$BUILDDIR
make clean
cmake -D CMAKE_BUILD_TYPE=RelWithDebInfo ../
make

cp ./objects_detection ../objects_detection




