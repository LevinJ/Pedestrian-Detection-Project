#!/bin/sh

set -x  # will print excecuted commands

echo "Generating objects detection files..."
cd src/objects_detection/
protoc --cpp_out=./ detector_model.proto detections.proto
protoc --python_out=../../tools/objects_detection/ detector_model.proto detections.proto
cd ../..


cd src/stereo_matching/ground_plane/
protoc --cpp_out=./ plane3d.proto
protoc --python_out=../../../tools/stixels_evaluation plane3d.proto
cd ../../..


cd src/stereo_matching/stixels/
protoc --cpp_out=./ -I. -I../ground_plane --include_imports  stixels.proto ground_top_and_bottom.proto
protoc --python_out=../../../tools/stixels_evaluation -I. -I../ground_plane --include_imports  stixels.proto ground_top_and_bottom.proto
cd ../../..


cd src/video_input/calibration
protoc --cpp_out=./ calibration.proto
cd ../../..


cd src/helpers/data
protoc --cpp_out=./ DataSequenceHeader.proto
protoc --python_out=../../../tools/data_sequence DataSequenceHeader.proto
cd ../../..


cd src/helpers
#protoc --cpp_out=./ stereo_rectification_homographies.proto
cd ../..


cd src/tests/data_sequence/
protoc --cpp_out=./ TestData.proto
cd ../../..

echo "End of game. Have a nice day!"
