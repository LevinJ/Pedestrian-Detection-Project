#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Small utility to visualize the feature vector and the learned model
"""


from __future__ import print_function

import os.path #, sys
from optparse import OptionParser
#import types

from pylab import *


def parse_arguments():
        
        parser = OptionParser()
        parser.description = \
            "This takes liblinear training, test or model files and " \
            "plots a visualization of the corresponding integral channels"
    
        parser.add_option("-i", "--input", dest="input_path",
                           metavar="FILE", type="string",
                           help="path to the training, test or model liblinear file")
        parser.add_option("-n", "--line_number", dest="line_to_read", 
                          metavar="NUMBER", type="int", default=-200,
                          help="line number (feature number) to read. " \
                          "If a negative number is given a random number between 0 and NUMBER is chose." \
                          "Only used if a training or test file is given")        
        parser.add_option("-c", "--class", dest="class_to_read", 
                          metavar="NUMBER", type="int", 
                          help="integer identifing the desired class. if not provided any class is acceptable")
        parser.add_option("-a", "--average", dest="do_average",
                           action="store_true", default = False,
                           #metavar="BOOL", type="int", default=False,
                           help="if set to true will plot the average of the feature vector for each class")
        
        (options, args) = parser.parse_args()
        #print (options, args)
    
        if options.input_path:
            if not os.path.exists(options.input_path):
                parser.error("Could not find the input file")
        else:
            parser.error("'input' option is required to run this program")
    
        
        if options.line_to_read == 0:
                parser.error("'line' number must be != 0")
        elif options.line_to_read > 0:
            pass
        else: # options.line_to_read < 0:
            options.line_to_read = int(random()*-options.line_to_read)
            
        
        return options 


def is_liblinear_model(input_file):
    """
    Is the file an liblinear trained SVM model or a training/test set ?
    """
    
    first_line = input_file.readline()
    input_file.seek(0) # reset
    is_model = "solver_type" in first_line    
    
    return is_model
    

def get_model_matrix(input_file):
    
    print("Going to plot linear svm model")    
    
    line = input_file.readline()
    while "w\n" not in line:
        line = input_file.readline()
    w_vector = []
    for line in input_file.readlines():
        w_vector.append(float(line))
    
    return feature_vector_to_matrix(w_vector, True), w_vector
    
    
def from_line_tokens_to_feature_vector(tokens):
    feature_vector = []
    i = 1
    for token in tokens[1:]:
        index, value = token.split(':')
        index = int(index); value = float(value)
        while index < i:
            feature_vector.append(0.0)
        feature_vector.append(value)
        i += 1

    return feature_vector  
    
def create_samples_iterator(input_file, class_to_read, loop = True):

    def samples_iterator(input_file, class_to_read, should_loop):
        
        class_value = None
        yield_once = False
        while True:
            for line in input_file:
                tokens = line.split()
                class_value = int(tokens[0])
                if type(class_to_read) is int:                    
                    if class_to_read != class_value:
                        continue # skip tis line
                
                # class_to_read == class_value or
                # all classes counts
                yield_once = True
                try:
                    yield class_value, from_line_tokens_to_feature_vector(tokens)
                except Exception as e:
                    print("Something went wrong when parsing the line. Aborting samples reading.")
                    print(line)
                    print(e)
                    break
            # end of for each line left in the file
            if should_loop:
                if yield_once:
                    print("Looping over the input file")
                    input_file.seek(0)
                else:
                    print("The input file does not contain samples of the desired class")
                    break
            else:
                break
        # end of while True
        return

    return samples_iterator(input_file, class_to_read, loop)  
    
def get_feature_matrix(input_file, class_to_read, line_number):
    
    print("Going to read feature", line_number)    

    print("class_to_read ==", class_to_read)    
    assert line_number > 0   
    if type(class_to_read) is int:
	    assert class_to_read in [-1, 1]
    
    i = 0
    class_value = None
    for line in input_file:
        if type(class_to_read) is int:
            class_value = int(line.split()[0])
            #print("class_value ==", class_value)    
            if class_to_read == class_value:
                i+=1
        else:
            i+=1 # all classes counts
        if i >= line_number:
	    class_value = int(line.split()[0])	
            break
        
    print("Read feature %i of class %s" % (i, str(class_value)))    
    
    assert line
    tokens = line.split()
    feature_vector = from_line_tokens_to_feature_vector(tokens)
        
    return feature_vector_to_matrix(feature_vector)
    
def feature_vector_to_matrix(v, is_model = False):

    v_len = len(v)    
    assert (v_len % 10) == 0
    
    resize_factor = int(sqrt(64*128*10/v_len))    
    assert (resize_factor % 2) == 0
        
    window_size = (64/resize_factor, 128/resize_factor)
    the_matrix = zeros((window_size[0]*10, window_size[1]))

    window_len = window_size[0]*window_size[1]  
    
    for i in range(10):
        sub_vector = v[i*window_len:(i+1)*window_len]        
        sub_matrix = matrix(sub_vector).reshape(window_size[1], window_size[0]).transpose()
        if is_model:
            #do_normalize = False
            do_normalize = True
            if do_normalize:
                # normalize positive and negative elements separatelly
                sub_matrix_pos = clip(sub_matrix, 0, sub_matrix.max())
                sub_matrix_neg = clip(sub_matrix, sub_matrix.min(), 0)
                sub_matrix = zeros(sub_matrix.shape)            
                
                if sub_matrix_pos.max() > 0:
                    sub_matrix_pos *= 1.0 / sub_matrix_pos.max()
                
                if sub_matrix_neg.min() < 0:
                    sub_matrix_neg *= 1.0 / sub_matrix_neg.min()
                
                # we invert positives and negatives to map to RbBu colormap
                sub_matrix -= sub_matrix_pos 
                sub_matrix += sub_matrix_neg
            else:
                # we invert positives and negatives to map to RbBu colormap
                sub_matrix = -sub_matrix                        
        else:    
            # normalize it all
            sub_matrix -= sub_matrix.min()
            sub_matrix *= 1.0 / sub_matrix.max()
        #print(the_matrix[i*window_size[0]:(i+1)*window_size[0], :].shape)
        #print(sub_matrix.shape)
        the_matrix[i*window_size[0]:(i+1)*window_size[0], :] = sub_matrix
    
    
    return the_matrix
    
def plot_vector(vector, the_title):
    
    # create figure    
    figure(0)
    clf() # clear the figure
    gcf().set_facecolor("w") # set white background            
    #grid(True)
    
    # draw the figure
    plot(vector)
    
    
    #legend(loc="upper right", fancybox=True)
    #legend(loc="best", fancybox=True)
    xlabel("Vector index")
    ylabel("Value")
    title(the_title)
    draw() 
    
    return
    
def plot_matrix(m, is_model, file_path, samples_iterator):
    
    
    # create figure    
    fig = figure(1)
    clf() # clear the figure
    gcf().set_facecolor("w") # set white background            
    #grid(True)
    
    colormap = cm.gray
    if is_model:
        colormap = cm.RdBu
        
    # draw the figure
    imshow(m.transpose(), cmap=colormap, interpolation="nearest")    
    
    #legend(loc="upper right", fancybox=True)
    #legend(loc="best", fancybox=True)
    #xlabel("Number of rectangles")
    #ylabel("l1 residual energy")
    title("%s" % os.path.basename(file_path))
    draw() 
    
    def on_key_press(event):
        if event.key in ["right", "down", " ", "enter"]:
            #print("Moving to next training sample")            
            class_value, feature_vector = samples_iterator.next()                    
            m = feature_vector_to_matrix(feature_vector)
            imshow(m.transpose(), cmap=colormap, interpolation="nearest")
            draw()
        elif event.key in ["left", "up", "backspace"]:
            print("Moving to previous training sample is not yet implemented")
        else:
            #print("Key ommited")
            #print("you pressed", event.key, event.xdata, event.ydata)
            pass

    if samples_iterator:
        #cid = 
        fig.canvas.mpl_connect("key_press_event", on_key_press)

    return


def compute_average(data):
    """
    Based on http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#On-line_algorithm
    """
    
    print("Computing the average feature vector and its variance. (this may take a while)")
    
    class_value, feature_vector = data.next()
    n = 0
    mean_vector = zeros(len(feature_vector))
    m2_vector = zeros(len(feature_vector))
 
    for class_value, feature_vector in data:
        x = feature_vector
        n = n + 1
        delta = x - mean_vector
        mean_vector = mean_vector + delta/n
        m2_vector = m2_vector + delta*(x - mean_vector)  # This expression uses the new value of mean
    # end of "for x in data"
    
    variance = m2_vector/n 
    #variance = m2_vector/(n - 1) # unbiased estimate
    
    return mean_vector, variance


def plot_average(mean, variance):
    
    
    mean = feature_vector_to_matrix(mean)
    variance = feature_vector_to_matrix(variance)
    
    colormap = cm.gray
    #colormap = cm.RdBu
    
    # create figure 
    figure(2)
    clf() # clear the figure
    gcf().set_facecolor("w") # set white background            
    #grid(True)
                        
    subplot(211)        
    # draw the figure
    imshow(mean.transpose(), cmap=colormap, interpolation="nearest")    
        
    #xlabel("Number of rectangles")
    #ylabel("l1 residual energy")
    title("mean")
    
    
    subplot(212)
    # draw the figure
    imshow(variance.transpose(), cmap=colormap, interpolation="nearest")    
        
    #xlabel("Number of rectangles")
    #ylabel("l1 residual energy")
    title("variance")
    draw() 

    return

def main():
    
    options =  parse_arguments()    
    
    # get the input file
    input_file = open(options.input_path, "r")

    # is it a model or a feature vector ?    
    is_model = is_liblinear_model(input_file)
    
    # create the matrix to visualize
    matrix_to_plot = None
    samples_iterator = None
    if is_model:
        matrix_to_plot, w_vector = get_model_matrix(input_file)
        plot_vector(w_vector, "w vector")
    else:
        matrix_to_plot = get_feature_matrix(input_file, options.class_to_read, options.line_to_read)
        should_loop = True
        samples_iterator = create_samples_iterator(input_file, options.class_to_read, should_loop)
        
    # plot
    plot_matrix(matrix_to_plot, is_model, input_file.name, samples_iterator)    
    
    if options.do_average:
        input_file = open(options.input_path, "r") # open second reader
        should_loop = False
        samples_iterator = create_samples_iterator(input_file, options.class_to_read, should_loop)
        mean, variance = compute_average(samples_iterator)
        plot_average(mean, variance)
    
    show()
    
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



