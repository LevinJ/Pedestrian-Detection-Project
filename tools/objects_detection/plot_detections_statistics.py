#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Small utility to visualize the detections statistics
"""


from __future__ import print_function

import sys
sys.path.append("..")
sys.path.append("../data_sequence")
sys.path.append("../helpers")

from detections_pb2 import Detections
from data_sequence import DataSequence

import os.path
from optparse import OptionParser

import pylab


def open_data_sequence(data_filepath):

    assert os.path.exists(data_filepath)

    the_data_sequence = DataSequence(data_filepath, Detections)

    def data_sequence_reader(data_sequence):
        while True:
            data = data_sequence.read()
            if data is None:
                raise StopIteration
            else:
                yield data

    return data_sequence_reader(the_data_sequence)


def parse_arguments():

        parser = OptionParser()
        parser.description = \
            "This program takes a detections.data_sequence " \
            "created by ./objects_detection and plots its statistics"

        parser.add_option("-i", "--input", dest="input_path",
                          metavar="FILE", type="string",
                          help="path to the .data_sequence file")

        (options, args) = parser.parse_args()
        #print (options, args)

        if options.input_path:
            if not os.path.exists(options.input_path):
                parser.error("Could not find the input file")
        else:
            parser.error("'input' option is required to run this program")

        return options


def new_blank_figure():

    pylab.figure()  # create new figure
    pylab.clf()  # clear the figure
    pylab.gcf().set_facecolor("w")  # set white background
    pylab.grid(True)

    return


def plot_scores_and_heights(scores, heights):

    # -=-=-=-=-=-=-=-=-=-=-=-=-
    new_blank_figure()

    print("Min score ==", min(scores))
    print("Max score ==", max(scores))
    #num_bins = int(max(scores) * 10)
    num_bins = 100
    pylab.hist(scores, num_bins)
    pylab.title("Scores distribution")

    # -=-=-=-=-=-=-=-=-=-=-=-=-
    new_blank_figure()

    num_bins = max(heights)
    pylab.hist(heights, num_bins)
    pylab.title("heights distribution")

    # -=-=-=-=-=-=-=-=-=-=-=-=-
    new_blank_figure()

    pylab.scatter(heights, scores)
    pylab.xlabel("Heights")
    pylab.ylabel("Scores")
    pylab.title("Scores versus heights")

    return


def main():

    options = parse_arguments()

    # get the input data sequence
    detections_sequence = open_data_sequence(options.input_path)

    detections_height = []
    detections_score = []

    # extract data of interest
    for detections in detections_sequence:
        for detection in detections.detections:
            bb = detection.bounding_box
            height = bb.max_corner.y - bb.min_corner.y
            detections_height.append(height)
            detections_score.append(detection.score)
        # end of "for each detection the frame"
    # end of "for each frame in the sequence"

    plot_scores_and_heights(detections_score, detections_height)
    pylab.show()

    print("Have a nice day")
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



