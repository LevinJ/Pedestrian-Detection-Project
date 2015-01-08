#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Companion script to doppia/src/tests/objects_detection/test_objects_detection

Will plot the content of very_fast_detector_scales_statistics.txt
"""

from __future__ import print_function

import pylab


def plot_statistics_file(filename, title, xlabel, ylabel):
  # parse the content --
    data = pylab.loadtxt(filename)
    
    num_points = data[:, -1]
    print("Used at least %i samples per plot point" % min(num_points))    

    delta_scales = data[:, 0] 
    print("min/max delta_scale ==", (min(delta_scales), max(delta_scales)))    
    
    # plot the content --
    pylab.figure() # new figure
    pylab.gcf().set_facecolor("w") # set white background
    pylab.grid(True)
 
    pylab.spectral() # set the default colormap to pylab.cm.Spectral

    labels = ["delta_scale", 
              "$\Delta$score mean", 
              "$\Delta$score min", 
              "$\Delta$score max", 
              "$\Delta$score 1%",
              "$\Delta$score 5%",
              "$\Delta$score 50%",
              "$\Delta$score 95%",
              "$\Delta$score 99%"]

    for index in range(1, len(labels)):
        x = data[:, 0]
        x_log = [pylab.sign(s)*pylab.log(abs(s)) for s in x]
        y = data[:, index]        
        #pylab.plot(x, y, label=labels[index])        
        pylab.plot(x_log, y, label=labels[index], marker=".")        


    #delta_xtick = (max(delta_scales) - min(delta_scales)) / min(10, len(delta_scales))
    #pylab.xticks(pylab.arange(min(delta_scales), max(delta_scales), delta_xtick))    
    pylab.legend(loc ="upper right", fancybox=True)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)
    pylab.title(title)

    pylab.draw()
    return

def main():
    
    if False:
        filename = "very_fast_detector_scale_statistics.txt"
        plot_statistics_file(filename,
                             "$\Delta$score versus $\Delta$scale", "$\Delta$scale", "$\Delta$score")

    if True:
        filename = "very_fast_detector_scale_centered_statistics.txt"
        plot_statistics_file(filename, "Score versus $\Delta$scale", "log($\Delta$scale)", "score")

  
    pylab.show() # blocking call
    
    #pylab.savefig("very_fast_detector_scale_statistics.pdf")
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


