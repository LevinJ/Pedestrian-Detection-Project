#!/bin/sh

set -x  # will print excecuted commands

echo "(Re)Compiling boosted_learning..."
cmake -D CMAKE_BUILD_TYPE=RelWithDebInfo . 
make -j5

if [ ! -d "InriaPerson_octave_0_train" ]; then

  DATA_URL="http://transfer.d2.mpi-inf.mpg.de/benenson/InriaPerson_octave_0_train.tar"
  #DATA_URL="file:///home/rodrigob/data/INRIAPerson_multiscales_crisp/InriaPerson_octave_0_train.tar"

  echo "Downloading the training data ..."
  curl -o InriaPerson_octave_0_train.tar ${DATA_URL} 

  echo "Extracting the training data ..."
  mkdir InriaPerson_octave_0_train
  tar xf InriaPerson_octave_0_train.tar --directory InriaPerson_octave_0_train
fi

echo "Launching training"

./boosted_learning -c inria_speedy_training_config.ini

echo "Training finished."

