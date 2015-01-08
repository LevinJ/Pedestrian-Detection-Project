#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This program reads a folder with negative images and creates
the data for training a multiscales model.
Code is based on create_multiscales_training_dataset.py
"""

from __future__ import print_function

import sys
#sys.path.append("..")
#sys.path.append("../data_sequence")
#sys.path.append("../helpers")
sys.path.append(
    "/users/visics/rbenenso/no_backup/usr/local/lib/python2.7/site-packages")

import os
import os.path
import glob
from optparse import OptionParser
from multiprocessing import Pool, cpu_count

import cv2

from create_multiscales_training_dataset import resized_image

import math
import pylab

print_counter = 0  # global variable


def parse_arguments():

    parser = OptionParser()
    parser.description = \
        "This program reads a folder with negative images and" \
        "creates the data for training a multiscales model."

    parser.add_option("-i", "--input", dest="input_path",
                      metavar="PATH", type="string",
                      help="path to the negatives dataset folder")

    parser.add_option("-o", "--output", dest="output_path",
                      metavar="DIRECTORY", type="string",
                      help="path to a non existing directory where "
                      "the new training dataset will be created")

    (options, args) = parser.parse_args()
    #print (options, args)

    if options.input_path:
        if not os.path.exists(options.input_path):
            parser.error("Could not find the input file")
        else:
            # we normalize the path
            options.input_path = os.path.normpath(options.input_path)
    else:
        parser.error("'input' option is required to run this program")

    if options.output_path:
        if os.path.exists(options.output_path):
            parser.error(
                "output_path should point to a non existing directory")
    else:
        parser.error("'output' option is required to run this program")

    return options


failures_count = 0


class Lambda:

    def __init__(self, f, args_tuple):
        self.args_tuple = args_tuple
        self.f = f
        return

    def __call__(self, args):
        #print("Lambda.__call__ with args", args)
        return self.f(*(args + self.args_tuple))


def create_negatives_process_file(negatives_counter, file_path,
                                  model_width, model_height, negatives_path):

    negative_image = cv2.imread(file_path)

    # negative images are 1.5 bigger than the model
    model_to_negative_factor = 1.5
    minimum_image_shape = (model_height * model_to_negative_factor,
                           model_width * model_to_negative_factor)

    # <= because the C++ code has some weird +1's
    if (negative_image.shape[1] <= minimum_image_shape[1]) or  \
       (negative_image.shape[0] <= minimum_image_shape[0]):
        # must resize the image
        negative_filename = "negative_sample_%i.resized.png" \
                            % negatives_counter

        # width / height
        image_ratio = float(negative_image.shape[1]) / negative_image.shape[0]
        if (negative_image.shape[1] <= minimum_image_shape[1]):
            resized_size = (int(minimum_image_shape[1]),
                            int(minimum_image_shape[1] / image_ratio))

        if (negative_image.shape[0] <= minimum_image_shape[0]):
            resized_size = (int(minimum_image_shape[0] * image_ratio),
                            int(minimum_image_shape[0]))

        resized_shape = (resized_size[1],
                         resized_size[0],
                         negative_image.shape[2])
        assert resized_shape[0] >= minimum_image_shape[0]
        assert resized_shape[1] >= minimum_image_shape[1]

        negative_image_resized = resized_image(negative_image,
                                               resized_size[0],
                                               resized_size[1])
        negative_image = negative_image_resized
    else:
        # we can just copy
        negative_filename = "negative_sample_%i.png" % negatives_counter

    image_output_path = os.path.join(negatives_path, negative_filename)
    cv2.imwrite(image_output_path, negative_image)
    print(".", end="")
    global print_counter
    print_counter += 1  # no need to be thread safe
    if print_counter % 10:
        sys.stdout.flush()
    return
# end of "create_negatives_process_file"


def create_negatives(model_width, model_height, octave,
                     data_path, output_path):

    # create the output folder
    negatives_path = os.path.join(output_path,
                                  "negatives_octave_%.1f" % octave)
    if os.path.exists(negatives_path):
        raise RuntimeError(negatives_path + " should not exist")

    os.mkdir(negatives_path)
    print("Created folder", negatives_path)

    negatives_path_pattern = os.path.join(data_path, "*.*")

    files_paths = list(glob.iglob(negatives_path_pattern))

    g = Lambda(create_negatives_process_file,
               (model_width, model_height, negatives_path))

    use_multithreads = True
    #use_multithreads = False
    if use_multithreads:
        # multithreaded processing of the files
        num_processes = cpu_count() + 1
        pool = Pool(processes=num_processes)
        pool.map(g, enumerate(files_paths))
        #parallel_map(process_file, files_paths, cpu_count() + 1)
    else:
        for counter, file_path in enumerate(files_paths):
            g((counter, file_path))

    print()

    resized_path_pattern = os.path.join(negatives_path, "*.resized.png")
    num_resized = len(list(glob.iglob(resized_path_pattern)))
    print("For octave %.1f created %i negatives images (%i are resized)"
          % (octave, len(files_paths), num_resized))

    return
# end of "def create_negatives"


def create_all_negatives(data_path, output_path,
                         model_width, model_height, cropping_border,
                         octave):

    model_width, model_height = int(model_width * (2 ** octave)), \
        int(model_height * (2 ** octave))
    octave_cropping_border = int(cropping_border * (2 ** octave))

    # make sure model size is pair, so we have a well defined center
    if (model_width % 2) == 1:
        model_width += 1
    if (model_height % 2) == 1:
        model_height += 1

    print(
        "At octave %.1f, "
        "model (width, height) == %s and cropping_border == %i" %
        (octave, str((model_width, model_height)), octave_cropping_border))

    create_negatives(model_width, model_height, octave,
                     data_path, output_path)
    return
# end of "def create_positives_and_negatives"


def create_multiscales_hard_negatives_dataset(data_path, output_path):

    # check thet data path is either Train or Test inside INRIAPerson
    base_path, subfolder_name = os.path.split(data_path)
    assert os.path.isdir(data_path)

    # define the scales we care about
    # this is the scales range used for INRIA
    min_scale, max_scale = 0.6094, 8.6
    delta_octave = 1.0
    #delta_octave = 0.5

    model_width, model_height = 64, 128
    # how many pixels around the model box to define a test image ?
    cropping_border = 20

    if False:
        # just for testing
        cropping_border = 20
        min_scale, max_scale = 4, 4

    min_octave = int(round(math.log(min_scale)/math.log(2)))
    max_octave = int(round(math.log(max_scale)/math.log(2)))
    #max_octave=-1
    octaves = list(pylab.frange(min_octave, max_octave, delta_octave))
     # for debugging, it is easier to look at big pictures first
    octaves.reverse()

    if os.path.exists(output_path):
        raise RuntimeError(output_path + " should not exist")
    else:
        os.mkdir(output_path)
        print("Created folder", output_path)

    print("Will create data for octaves", octaves)

    for octave in octaves:
        create_all_negatives(data_path, output_path,
                             model_width, model_height, cropping_border,
                             octave)
    # end of "for each octave"

    return
# end of "def create_multiscales_training_dataset"


def main():
    options = parse_arguments()

    create_multiscales_hard_negatives_dataset(options.input_path,
                                              options.output_path)

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