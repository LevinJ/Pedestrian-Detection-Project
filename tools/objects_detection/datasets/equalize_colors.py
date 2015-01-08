#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Color calibrate images in a directory using selected algorithms

Requires colorcorrect package
http://pypi.python.org/pypi/colorcorrect

You can install the colorcorrect package via pip.
For example: pip install --user colorcorrect
"""

from __future__ import print_function

import os.path
import sys
#sys.path.append("..")
#sys.path.append("../data_sequence")
sys.path.append(os.path.join(os.path.dirname(__file__), "../../helpers"))

import os
from optparse import OptionParser

import progressbar
from query_yes_no import query_yes_no
from PIL import Image

try:

    import colorcorrect.algorithm as cca
    from colorcorrect.util import from_pil, to_pil
except:
    raise RuntimeError("colorcorrect package seems not installed")


methods = {
    "grey_world": cca.grey_world,
    "luminance_weighted_grey_world": cca.luminance_weighted_gray_world,
    "automatic": cca.automatic_color_equalization
}


def parse_arguments():

    parser = OptionParser()
    parser.description = \
        "Will equalize the colors of " \
        "all the images in a folder using the indicated algorithm"

    parser.add_option("-i", "--input", dest="input_path",
                       metavar="PATH", type="string",
                       help="path to the folder containing images")

    parser.add_option("-f", "--input_file", dest="input_file",
                       metavar="PATH", type="string",
                       help="path to the folder containing images")

    parser.add_option("-m", "--method", dest="method",
                       metavar="METHOD", type="string",
                       default = "luminance_weighted_grey_world",
                       help= \
"""which method to use?\n
Options are: \ngrey_world,
luminance_weighted_grey_world,
or automatic\n
[default: %default]""")
    parser.add_option("-c", "--on_condor", dest="on_condor",
                       action="store_true",
                       help="This option disables " \
                       "the security check for overriding images")

    (options, args) = parser.parse_args()
    #print (options, args)

    if not (options.input_path or options.input_file):
        parser.error("'input' option is required to run this program")

    if options.method not in methods.keys():
        parser.error("Received an unknow method option '%s', "
        "possible values are %s" % (options.method, methods.keys()))

    return options


def equalize_colors_file(input_file, method_name, on_condor=False):


    equalize_image = methods[method_name]

    print (input_file)

    try:
        image = Image.open(input_file)
    except:
        print("Failed to open image %s, skipping." % input_file)

    new_image = to_pil(equalize_image(from_pil(image)))
    new_image.save(input_file)


    if on_condor:
        f = open(input_file+ ".done", "w")
        f.writelines("test\n")
        f.close()

    # end of "for each image in the folder"

    return
def equalize_colors(input_path, method_name, on_condor=False):

    if not on_condor:
        yes_should_continue = query_yes_no(
            "All the images in %s will be modified.\n"
            "No backup will be created, ARE YOU SURE?" % input_path,
            default="no")

        if not yes_should_continue:
            print("End of game. No file has been edited.")
            return

    equalize_image = methods[method_name]

    filenames = os.listdir(input_path)

    progress_bar = None
    if not on_condor:
        progress_bar = progressbar.ProgressBarWithMessage(len(filenames),
                                                          " Processing images")

    files_counter = 0
    for filename in filenames:
        print (filename)

        file_path = os.path.join(input_path, filename)
        try:
            image = Image.open(file_path)
        except:
            print("Failed to open image %s, skipping." % file_path)
            continue

        new_image = to_pil(equalize_image(from_pil(image)))
        new_image.save(file_path)

        files_counter += 1

        if progress_bar:
            progress_bar.update(files_counter)
        else:
            # on condor
            f = open(file_path + ".done", "w")
            f.writelines("test\n")
            f.close()

    # end of "for each image in the folder"

    if progress_bar:
        progress_bar.finish()

    print("%i images inside %s have been equalized" % (files_counter,
                                                       input_path))

    return


def main():
    options = parse_arguments()

    if options.input_path:
        equalize_colors(options.input_path, options.method, options.on_condor)

    if options.input_file:
        equalize_colors_file(options.input_file, options.method, options.on_condor)

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

