#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Will create a new synthetic_cascades_thresholds.txt
"""

import pylab

num_cascades = 5
num_stages = 2000


thresholds = [0]*num_stages

if False:
# 2011 cascade (CVPR submission time)
	score_0 = -0.01
	score_1250 = -0.03
	score_2000 = -0.01

	for i in range(num_stages):
		if i < 1250:
			t = ((score_1250 -  score_0)*i/1250) + score_0
		else:
			t = ((score_2000 -  score_1250)*(i - 1250)/(2000 - 1250)) + score_1250
		thresholds[i] = t
else:
	# 2012 cascade (ECCV workshop submission time) 
	from scipy.interpolate import interp1d as interpolate_1d
	curve_samples_x = [0, 6, 11, 45, 60, 
						180, 370, 500, 1000, 1350, 1500, 1750, 2000]
	curve_samples_y = [-0.002, -0.0078, -0.007, -0.011, -0.013, 
						-0.018, -0.02, -0.023, -0.0227, -0.0215, -0.0245, -0.0145, -0.006]
	spline_order = 1
	curve = interpolate_1d(curve_samples_x, curve_samples_y, kind=spline_order)
	for i in range(num_stages):
		thresholds[i] = curve(i)

data = [thresholds] * num_cascades
data = pylab.array(data)

filename = "synthetic_cascades_thresholds.txt"
pylab.savetxt(filename, data)

print "Created", filename





