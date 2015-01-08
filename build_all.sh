#!/bin/sh
# Do not use this script unless you know what you are doing,
# start by reading readme.text to know which are the first step to do get this code base to compile.

set -x  # will print excecuted commands

# change num threads depending on available RAM and Cpu cores.
NUM_THREADS=6

build_all()
{

#for dirname in src/applications/*/ src/tests/*/ ;
for dirname in src/applications/*/ ;
do
  cd ${dirname}
  cmake -D CMAKE_BUILD_TYPE=RelWithDebInfo .
  make -j${NUM_THREADS}
  cd -
done

}

while true; do
		read -p "Do you _really_ know what you are doing ? [y/N] " yn
		case $yn in
		[Yy]* ) build_all; break;;
		[Nn]* ) exit ;;
		#* ) echo "Please answer yes or no.";;
		* ) exit ;;
		esac
	done

