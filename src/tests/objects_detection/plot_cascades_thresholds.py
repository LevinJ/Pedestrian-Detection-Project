#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Companion script to doppia/src/tests/objects_detection/test_objects_detection

Will plot the content of cascade_threshold.txt
"""

from __future__ import print_function

import pylab


def plot_cascades_thresholds():
   # parse the content --
    filename = "cascade_threshold.txt"
    #filename = "../../../tools/objects_detection/synthetic_cascades_thresholds.txt"
    data = pylab.loadtxt(filename)
    
    # plot the content --
    pylab.figure() # new figure
    pylab.gcf().set_facecolor("w") # set white background
    pylab.grid(True)
 
    pylab.spectral() # set the default colormap to pylab.cm.Spectral


    labels = ["Scale 0.5",
              "Scale 1",
              "Scale 2",
              "Scale 4",
              "Scale 8",
              "Scale 16" ]

    for index in range(1, data.shape[0]):
        x = data[index, :]
        if x[0] < 1E10:
            pylab.plot(x, label=labels[index])        
        else:
            # has max_float value
            pass

    #delta_xtick = (max(delta_scales) - min(delta_scales)) / min(10, len(delta_scales))
    #pylab.xticks(pylab.arange(min(delta_scales), max(delta_scales), delta_xtick))    
    pylab.legend(loc ="upper left", fancybox=True)
    pylab.xlabel("Cascade stage")
    pylab.ylabel("Score threshold")
    pylab.title("Cascade thresholds")

    pylab.draw()
    return

def main():

    plot_cascades_thresholds()
  
    pylab.show() # blocking call
    
    #pylab.savefig("cascades_thresholds.pdf")
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


