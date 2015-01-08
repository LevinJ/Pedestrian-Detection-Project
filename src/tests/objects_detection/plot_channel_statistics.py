#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Companion script to doppia/src/tests/objects_detection/test_objects_detection

Will plot the content of channel_statistics.txt
"""

from __future__ import print_function
import pylab

def main():
    
    filename = "channel_statistics.txt"
    
    # parse the content --
    data = pylab.loadtxt(filename)
    scales = data[0,:]
    print("num_scales == ", len(scales))
    for i in range(len(scales)):
        if scales[i] > 1:
            break
    down_scales_slice = slice(0, i)
    up_scales_slice = slice(i, None)
    num_channels = (len(data) - 1)/2
        
    # do regressions --
    do_regressions = True
    # channels to be estimated together
    # this values where set after observing the data a first time    
    regression_groups = {"HOG":range(0,7), "L":[7], "UV":range(8,10)}   
    down_scales_regressions = {}
    up_scales_regressions = {}
    if do_regressions:
        log_data = pylab.log(data)       
        log_scales = log_data[0,:]                
        for name, channels_indices in regression_groups.items():
            down_scales_x = list(log_scales[down_scales_slice])*len(channels_indices)       
            up_scales_x = list(log_scales[up_scales_slice])*len(channels_indices)       
            down_scales_y = []
            up_scales_y = []
            for c in channels_indices:
                down_scales_y.extend(log_data[c*2 + 1, down_scales_slice])
                up_scales_y.extend(log_data[c*2 + 1, up_scales_slice])            
                
            # r = a*(k**b) => log(r) = b*log(k) + log(a)
            down_b, down_log_a = pylab.polyfit(down_scales_x, down_scales_y, 1)
            up_b, up_log_a = pylab.polyfit(up_scales_x, up_scales_y, 1)
            
            down_scales_regressions[name] = [down_b, pylab.exp(down_log_a)]
            up_scales_regressions[name] = [up_b, pylab.exp(up_log_a)]
            print("%s\tfor downscaling r = %.3f*(x**%.3f), "
                  "for upscaling  r = %.3f*(x**%.3f)" % (name, 
                  down_scales_regressions[name][1], down_scales_regressions[name][0], 
                  up_scales_regressions[name][1], up_scales_regressions[name][0]))
    
    #print(regression_groups)    
    
    # plot the content --
    pylab.figure(0)
    pylab.gcf().set_facecolor("w") # set white background
    pylab.grid(True)
 
    colormap = pylab.cm.Spectral
    #colormap = pylab.cm.gist_rainbow
    #colormap = pylab.cm.brg
       
    for channel_index in range(num_channels):
            color = colormap( channel_index / float(num_channels) )
            label = "channel %i" % channel_index
            #label = None
            
            # mean down
            pylab.subplot(2,2,1)
            x = scales[down_scales_slice]
            y = data[channel_index*2 + 1, down_scales_slice]            
            pylab.plot(x,y, color=color)#, label=label)

            # std dev down
            pylab.subplot(2,2,3)
            x = scales[down_scales_slice]
            y = data[channel_index*2 + 2, down_scales_slice]            
            pylab.plot(x,y, color=color)#, label=label)
        
            # mean up
            pylab.subplot(2,2,2)
            x = scales[up_scales_slice]
            y = data[channel_index*2 + 1, up_scales_slice]            
            pylab.plot(x,y, color=color)#, label=label)
        
            # std dev up
            pylab.subplot(2,2,4)
            x = scales[up_scales_slice]
            y = data[channel_index*2 + 2, up_scales_slice]            
            pylab.plot(x,y, color=color, label=label)

    for label, b_a in down_scales_regressions.items():
        b,a = b_a        
        # mean down
        pylab.subplot(2,2,1)
        x = scales[down_scales_slice]
        y = [a*(k**b) for k in x]
        color = colormap( regression_groups[label][0] / float(num_channels) )
        pylab.plot(x,y, 
                   color=color, label=label, 
                   linewidth=1.5, linestyle="--")

    for label, b_a in up_scales_regressions.items():
        b,a = b_a        
        # mean down
        pylab.subplot(2,2,2)
        x = scales[up_scales_slice]
        y = [a*(k**b) for k in x]
        color = colormap( regression_groups[label][0] / float(num_channels) )
        pylab.plot(x,y, 
                   color=color, label=label, 
                   linewidth=1.5, linestyle="--")

    pylab.subplot(2,2,1)
    pylab.xlabel("scales")
    pylab.ylabel("mean ratio")
    pylab.title("Mean ratio when downscaling")

    pylab.subplot(2,2,3)
    #pylab.legend(loc ="lower right", fancybox=True)
    pylab.xlabel("scales")
    pylab.ylabel("Standard deviation of ratio")
    pylab.title("Standard deviation of when downscaling")
        
    pylab.subplot(2,2,2)
    pylab.legend(loc ="lower right", fancybox=True)
    pylab.xlabel("scales")
    pylab.ylabel("mean ratio")
    pylab.title("Mean ratio when upscaling")

    pylab.subplot(2,2,4)
    pylab.legend(loc ="lower right", fancybox=True)
    pylab.xlabel("scales")
    pylab.ylabel("Standard deviation of ratio")
    pylab.title("Standard deviation of when upscaling")
        
    pylab.suptitle("Channel statistics")

    pylab.draw()
    pylab.show() # blocking call
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


