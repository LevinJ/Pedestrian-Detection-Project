#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This program parses the INRIAPerson dataset and creates
an occlusion test set for different magnitudes of occlusion (first only from bottom to center of the bounding boxes)
"""

from __future__ import print_function
from optparse import OptionParser
from multiprocessing import Pool, Lock, cpu_count

import sys
import os, os.path, glob
#sys.path.append("..")
#sys.path.append("../data_sequence")
#sys.path.append("../helpers")
sys.path.append("/users/visics/rbenenso/no_backup/usr/local/lib/python2.7/site-packages")
 
import cv2
import create_multiscales_training_dataset as cmt
PEDESTRIAN_TRAINING_SIZE_HEIGHT=128
PEDESTRIAN_TRAINING_SIZE_WIDTH=64

def generateBottomOcclusion(h, output_path, imname, im, border, show):
			savefolder = os.path.join(output_path, "crop"+"-0-"+str(h)+"-0-0")
			savename= os.path.join(savefolder,os.path.basename(imname))
			if not os.path.exists(savefolder):
				os.mkdir(savefolder)
			im = occlude_pedestrian(im,border,0,h,0,0)

			if show and not use_multishreads:
				cv2.namedWindow("test", 1)
				cv2.imshow("test",im)
				cv2.waitKey(0)
			cv2.imwrite(savename, im)

def occlude_pedestrian(im, border, cuttop, cutbottom, cutleft, cutright):
    border_type = cv2.BORDER_REPLICATE
    if cuttop>=0:
        cuttop = border + cuttop
    if cutbottom>=0:
        cutbottom = border + cutbottom
    if cutleft>=0:
        cutleft = border + cutleft
    if cutright>=0:
        cutright = border + cutright
        
    croppedPedestrian = im[cuttop:im.shape[0]-cutbottom, cutleft:im.shape[1]-cutright,:]

    im = cv2.copyMakeBorder(croppedPedestrian, 
                               cuttop, cutbottom,
                               cutleft, cutright,
                               border_type)
#return im
    return croppedPedestrian 

class Lambda:
    
    def __init__(self, f, args_tuple):
        self.args_tuple = args_tuple
        self.f = f 
        return
        
    def __call__(self, args):
        return self.f(*(args + self.args_tuple))


def parse_arguments():
        
    parser = OptionParser()
    parser.description = \
        "This program takes the INRIA pedestrians dataset and " \
        "creates occluded pedestrians"

    parser.add_option("-i", "--input", dest="input_path",
                       metavar="PATH", type="string",
                       help="path to the dataset to be cropped")


    parser.add_option("-o", "--output", dest="output_path",
                       metavar="DIRECTORY", type="string",
                       help="path to a non existing directory where the new cropped dataset will be created")
                                                  
    parser.add_option("-s", "--show", dest="show", action="store_true", help="show images")



    (options, args) = parser.parse_args()

    if options.input_path:
        if not os.path.exists(options.input_path):
            parser.error("Could not find the input file")
        else:
            # we normalize the path
            options.input_path = os.path.normpath(options.input_path)
    else:
        parser.error("'input' option is required to run this program")

    if not options.output_path:
        parser.error("'output' option is required to run this program")
        if os.path.exists(options.input_path):
            parser.error("output path already exists")

    return options 

def getFilenames(folder):
	image_path_pattern = os.path.join(folder, "*.png")
	for image_path in glob.iglob(image_path_pattern):
		yield(image_path)
        



def main():
	options = parse_arguments()
	#use_multithreads = True 
	use_multithreads = False

	filenames= getFilenames(options.input_path)

	if not os.path.exists(options.output_path):
		os.mkdir(options.output_path)
	
	
	count = 0
	while 1:
            imname =filenames.next()
            border=20
#read image once, generate all occlusions
            im = cv2.imread(imname)
            print (imname)
            h = im.shape[0]
            w = im.shape[1]
#assert((w-2*border - PEDESTRIAN_TRAINING_SIZE_WIDTH) ==0)
#assert((h-2*border - PEDESTRIAN_TRAINING_SIZE_HEIGHT)==0)
            count +=1
            print(count)
            data = [(i*4,) for i in range(17)]
            g = Lambda(generateBottomOcclusion, (options.output_path, imname, im, border, options.show))

            if use_multithreads:
                # multithreaded processing of the files
                num_processes = cpu_count() + 1
                pool = Pool(processes = num_processes)
                chunk_size = len(data) / num_processes
                pool.map(g, data, chunk_size)
                
            else:    
                for d in data: 
                    g(d)

if __name__ == "__main__":
	main()
