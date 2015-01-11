# Pedestrian-Detection-Project



#Project Goal:

The goal of this project is to explore how better feature reprentations and various visual cues can be used to improve detection quality in computer vision area.

Specifically, this project targets the fascinating and meaningful real world problem "pedestrian detection" as a playground. Using current state of the art open source pedestrian detector "SquaresChnFtrs" as a baseline, I am using two methods to increase detection accuracy. Expand 10 HOG+LUV channels into 40 channels by using DCT (discrete cosine transform); Encode the optical flow using SDt features (image difference between current frame T and coarsely aligned T-4 and T-8); 


Note that this project is largely to reproduce observations/discovery in “Benenson etc., 2014 EECV” paper.The baseline detecor's miss rate on Inria pedestiran dataset is 34.81%. The dct mehtod is expected to have 3.53% percent improvement, and the optical flow method is expected to have 4.47% improvement. 



What Has Been Done:

1. Got the baseline detector up and running
2. Got baseline  miss rate
3. Implemented the new baseline + DCT pedestrian detector. 
5. Cross verified that DCT algorithm CUDA implementation in the new detector is correct.


What's Next:

Investigate what went wrong with baseline + DCT detector, and then move on to implement the baseline + optical flow 

Current Issues:

The DCT method is not generating the 3.53% detection rate improvement as expected, instead it has a negative 20% detection rate impact. I am investigating what's going wrong and trying to come up with an conclusion on this issue.
