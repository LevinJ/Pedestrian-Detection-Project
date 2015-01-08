#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This program parses the INRIAPerson dataset and creates
negative samples, sampled from the negative set
"""

from __future__ import print_function

import sys
#sys.path.append("..")
#sys.path.append("../data_sequence")
sys.path.append("../helpers")
sys.path.append("/users/visics/rbenenso/no_backup/"
                "usr/local/lib/python2.7/site-packages")

import os
from optparse import OptionParser
import random
import itertools
import cv2
from math import log, exp
#from multiprocessing import Pool, cpu_count

import progressbar
import create_multiscales_training_dataset as cmtd


def parse_arguments():

    parser = OptionParser()
    parser.description = \
        "This program takes the INRIA pedestrians dataset and " \
        "creates a negative samples for training a classifier"

    parser.add_option("-i", "--input", dest="input_path",
                      metavar="PATH", type="string",
                      help="path to the INRIAPerson dataset "
                      "train negatives folder")

    parser.add_option("-n", "--num_samples", dest="num_samples",
                      metavar="NUMBER", type="int", default=5000,
                      help="number of samples to extract from the image")

    parser.add_option("-o", "--output", dest="output_path",
                      metavar="DIRECTORY", type="string",
                      help="path to a non existing directory "
                      "where the new training dataset will be created")

    (options, args) = parser.parse_args()
    #print (options, args)

    if options.input_path:
        if not os.path.exists(options.input_path):
            parser.error("Could not find the input path")

        if not os.path.isdir(options.input_path):
            parser.error("Input path should be a directory")

    else:
        parser.error("'input' option is required to run this program")

    if options.output_path:
        if os.path.exists(options.output_path):
            parser.error("output_path should point to "
                         "a non existing directory")
    else:
        parser.error("'output' option is required to run this program")

    return options


def crop_random_sample_from_image(crop_width, crop_height, image, scales):

    height, width, _ = image.shape

    def check_scale(scale):
        scaled_width, scaled_height = int(width / scale), int(height / scale)
        return (scaled_width > crop_width) and (scaled_height > crop_height)

    valid_scales = [s for s in scales if check_scale(s)]

    if len(valid_scales) == 0:
        raise RuntimeError("Receive an image where "
                           "no scale would fit a sample")

    scale = random.choice(valid_scales)

    scaled_width, scaled_height = int(width / scale), int(height / scale)

    min_x = random.randint(0, scaled_width - crop_width)
    min_y = random.randint(0, scaled_height - crop_height)
    assert (min_x >= 0) and (min_y >= 0)

    # resize image, crop sample
    resized_image = cmtd.resized_image(image,
                                       scaled_width, scaled_height)
    assert resized_image.shape[:2] == (scaled_height, scaled_width)

    sample = resized_image[min_y: min_y + crop_height,
                           min_x: min_x + crop_width,
                           :]
    assert sample.shape[:2] == (crop_height, crop_width)

    return sample


def crop_random_samples_from_image(num_crops,
                                   crop_width, crop_height,
                                   image, scales):
    """
    Will random rescale the image and crop num_crops samples.
    Returns an array of (cropped) images.
    """

    f = lambda: crop_random_sample_from_image(crop_width, crop_height,
                                              image, scales)
    return [f() for i in range(num_crops)]


def compute_crop_width_and_height():

    object_width, object_height = 64, 128
    #margin_x, margin_y = 16, 16
    margin_x, margin_y = 20, 20
    crop_width = object_width + (2*margin_x)
    crop_height = object_height + (2*margin_y)

    return crop_width, crop_height


def compute_scales():

    min_scale, max_scale = 0.5, 8.6
    num_scales = 55
    log_min_scale, log_max_scale = log(min_scale), log(max_scale)

    delta_log_scale = (log_max_scale - log_min_scale) / (num_scales + 1)
    scales = [exp(i*delta_log_scale + log_min_scale)
              for i in range(num_scales)]

    return scales


def main():

    options = parse_arguments()

    # we assumes all files in the folder are images
    image_paths = [os.path.join(options.input_path, filename)
                   for filename in os.listdir(options.input_path)]

    # rounding might go up or down
    num_crops = int(float(options.num_samples) / len(image_paths))

    crop_width, crop_height = compute_crop_width_and_height()
    scales = compute_scales()

    progress = progressbar.ProgressBar(maxval=options.num_samples)
    progress.start()
    num_samples_created = 0
    for image_index, image_path in enumerate(itertools.cycle(image_paths)):
        if num_samples_created >= options.num_samples:
                break

        image = cv2.imread(image_path)  # read image

        if image is None:
            print("Failed to read image", image_path)
            raise RuntimeError("Failed to read one of the input images")

        samples = crop_random_samples_from_image(num_crops,
                                                 crop_width, crop_height,
                                                 image, scales)
        # save samples
        for i, sample in enumerate(samples):

            if num_samples_created >= options.num_samples:
                break

            if num_samples_created == 0:
                os.mkdir(options.output_path)
                print("Created output folder ", options.output_path)

            sample_filename = "%s.%i.png" % (os.path.basename(image_path),
                                             num_samples_created)
            sample_path = os.path.join(options.output_path, sample_filename)
            cv2.imwrite(sample_path, sample)
            num_samples_created += 1
        # end of "for each sample in this image"
        progress.update(num_samples_created)
    # end of "for each image path"
    progress.finish()

    print("Visited %i images to create %i samples, saved inside folder %s "
          % (min(image_index, len(image_paths)),
             num_samples_created, options.output_path))

    print("End of game, have a nice day!")
    return


if __name__ == "__main__":

    # Import Psyco if available
    try:
        import psyco
        psyco.full()
    except ImportError:
        #print("(psyco not found)")
        pass
    else:
        print("(using psyco)")

    main()

# end of file
