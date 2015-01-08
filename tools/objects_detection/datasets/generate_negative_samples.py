
from __future__ import print_function

import Image
import glob
import random
import os


from optparse import OptionParser


def parse_arguments():

    parser = OptionParser()
    parser.description = \
        "This program takes the INRIA pedestrians dataset and " \
        "creates occluded pedestrians"

    parser.add_option("-i", "--input", dest="input_path",
                       metavar="PATH", type="string",
                       help="path to the INRIAPerson dataset of negatives for Test or Train ")


    parser.add_option("-o", "--output", dest="output_path",
                       metavar="DIRECTORY", type="string",
                       help="path to a non existing directory where the new training dataset will be created")

    parser.add_option("-t", "--target_size", dest="target_size",
                       type="string",default="70,134",
					   help="Target size of the sampeled rectangles: for inria Test 70,134")
    parser.add_option("-n", "--number_of_samples", dest="number_of_samples",
                       type="int",default=1000,
                       help="Number of samples to be randomly choosen")
    parser.add_option( "--minsize", dest="minsize",
                       type="int",default=200,
                       help="Minsize of the rectangle to be considered as candidate")
    parser.add_option( "--maxsize", dest="maxsize",
                       type="int",default=110000,
                       help="Maxsize of the rectangle to be considered as candidate")
    parser.add_option("--sample_ratio", dest="sample_ratio", type="float",default=0.5, help="ratio of the element to sample: width/height")

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
        if os.path.exists(options.output_path):
			parser.error("output_path already exists")

    return options


def findBackgroundSamples(imsize, minSize, maxSize, samples_per_image, ratio):

    reclist = []
    while (len(reclist) < samples_per_image):
        w, h = imsize

        #sample new rectangle of minsize
        x1 = random.randint(0, w-1)
        x2 = random.randint(x1, w)
        y1 = random.randint(0, h-1)
        w1 = x2 - x1
        h1 = int(round(1.0/ratio * w1))
        y2 = y1 + h1
        if (((x2-x1) * (y2-y1)) < minSize) \
                or (y2 > h) \
                or (((x2-x1) * (y2-y1)) > maxSize):
            continue
        rect = [x1, y1, x2, y2]
        reclist.append(rect)

    return reclist


def main():
    options = parse_arguments()
    os.mkdir(options.output_path)
    minSize = options.minsize
    maxSize = options.maxsize
    number_of_samples = options.number_of_samples
    target_size = options.target_size.strip()
    target_size = target_size.split(",")
    target_size = [int(i.strip()) for i in target_size]
    counter = 0
    image_file_names = glob.glob(os.path.join(options.input_path, '*.png'))
    samples_per_image = int(number_of_samples / float(len(image_file_names)))
    print(samples_per_image)

    for elem in image_file_names:
        im = Image.open(elem)
        print(elem)
        reclist = findBackgroundSamples(im.size, minSize, maxSize,
                                        samples_per_image,
                                        options.sample_ratio)

        for rec in reclist:
            counter += 1
            if (counter % 1000 == 0 and options.show):
                print (counter)

            cropim = im.crop([rec[0], rec[1], rec[2], rec[3]])

            raise Exception("PIL resize should not be used, "
                            "only opencv_gpu_resize")
            reszim = cropim.resize(target_size, Image.ANTIALIAS)
            filename = "neg%08d.png" % counter
            reszim.save(os.path.join(options.output_path, filename))
        # end of "for each rectangle"

if __name__ == "__main__":
	main()

