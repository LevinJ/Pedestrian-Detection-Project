#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Small application that will test the python module opencv_gpu_resize
"""

from __future__ import print_function

import cv2
import numpy
import opencv_gpu_resize as ogr

input_image = cv2.imread("test_image.png")

#output_width, output_height = 157, 213  # arbritrary values
output_width, output_height = 257, 171  # arbritrary values
output_image = numpy.zeros((output_height, output_width, 3),
                           dtype=numpy.uint8)

ogr.gpu_resize(input_image, output_image)

output_filename = "test_image.gpu_resized.png"
cv2.imwrite(output_filename, output_image)

print("Created %s for visual inspection" % output_filename)
