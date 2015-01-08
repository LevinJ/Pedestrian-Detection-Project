#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mini VISICS specific script to plot the caltech evaluation results on the INRIA dataset
"""

from __future__ import print_function

import os.path
import sys
local_dir = os.path.dirname(sys.argv[0])
sys.path.append(os.path.join(local_dir, "../data_sequence"))
sys.path.append(os.path.join(local_dir, "../helpers")) 

import os, os.path, shutil, subprocess, glob
from optparse import OptionParser

from detections_to_caltech import detections_to_caltech

at_visics = True
#at_visics = False

def plot_caltech_evaluation(results_path, disable_show, exp_idx ):

    results_path = os.path.normpath(results_path)
    results_name = os.path.split(results_path)[-1]

    data_sequence_path = os.path.join(results_path, "detections.data_sequence")
    vxxx_path  = os.path.join(results_path, "Vxxx")
    print (results_name)

    if os.path.exists(vxxx_path):
        print(vxxx_path, "already exists, skipping the generation step")
    else:
        # convert data sequence to caltech data format
        detections_to_caltech(data_sequence_path, vxxx_path)

    caltech_evaluation_dir = "/esat/kochab/mmathias/caltech_pedestrian/evaluation/code3.0.0/"
    caltech_results_dir = os.path.join(caltech_evaluation_dir, "data-USA/res/Ours-wip_"+results_name)
    print(caltech_results_dir)
   
    
    set01_path = os.path.join(caltech_results_dir, "setxxx_" + results_name)
    for f in glob.glob(os.path.join(vxxx_path,"*.txt")):
        [p,txtname] = os.path.split(f)
        [a, b, c] = txtname.split('_')
        resfolder = os.path.join(caltech_results_dir, a, b)
        if not os.path.exists(resfolder):
            print(resfolder)
            os.makedirs(resfolder)
        shutil.copy(f, os.path.join(resfolder,c))




    # update the set01 symbolic link    
    ln_path = os.path.join(caltech_evaluation_dir, "data-USA/res/Ours-wip")
    os.remove(ln_path)
    os.symlink(caltech_results_dir, ln_path)

    # clear previous evaluations    
    caltech_eval_dir = os.path.join(caltech_evaluation_dir, "data-USA/eval")    
    paths_to_remove = glob.glob(os.path.join(caltech_eval_dir, "*Ours-wip*"))
    for path in paths_to_remove:
        os.remove(path)
    

    # run the matlab code
    current_dir = os.getcwd()
    os.chdir(caltech_evaluation_dir)
    if at_visics:
        #matlab_command = 'matlab -nodisplay -r "dbEval ('+str(exp_idx)+'); quit"'
        matlab_command = 'matlab -nodisplay -r "dbEval ; quit"'
    else:
        matlab_command = 'octave --eval "dbEvalOctave; quit"'
        
    print("Running:", matlab_command)
    subprocess.call(matlab_command, shell=True)
    os.chdir(current_dir)
    
	# copy and rename the the curve of our detector    
    #result_mat_path = os.path.join(set01_path, results_name + "Ours-wip.mat")    
    #shutil.copy(os.path.join(caltech_eval_dir, "Ours-wip.mat"), result_mat_path)

    # copy and rename the resulting pdf    
    result_pdf_path = os.path.join(caltech_results_dir, "output_roc.pdf")    
    print("result pdf path", result_pdf_path)
    pdfname = "UsaTest roc exp=reasonable.pdf"
    print ("pdfName: ", pdfname)
    print ("from: " , os.path.join(caltech_evaluation_dir, "results", pdfname))
    shutil.copy(os.path.join(caltech_evaluation_dir, "results", pdfname), result_pdf_path)

    
    # done
    print("Done creating the evaluation. Check out %s." % result_pdf_path)
    if not disable_show:
		subprocess.call("gnome-open %s" % result_pdf_path, shell=True)    
    return

def getExperimentName(index):
    exps = {}

    exps[1] ='all'
    exps[2] ='reasonable'
    exps[3] ='scale=near'
    exps[4] ='scale=medium'
    exps[5] ='scale=far'
    exps[6] ='occ=none'
    exps[7] ='occ=partial'
    exps[8] ='occ=heavy'
    exps[9] ='ar=all'
    exps[10] ='ar=typical'
    exps[11] ='ar=atypical'
    exps[12] ='octave=-1'
    exps[13] ='octave=0'
    exps[14] ='octave=1'
    exps[15] ='octave=2'
    exps[16] ='octave=3'
    return exps[index]

def main():

    parser = OptionParser()
    parser.description = \
        """
        Reads the recordings of objects_detection over Caltech version of INRIA dataset, 
        calls the Caltech evaluation toolbox, and shows the resulting plot
        """

    parser.add_option("-i", "--input", dest="input_path",
                       metavar="FILE", type="string",
                       help="path to the recording directory")
    parser.add_option("-d", "--disable_show", dest="disable_show",
                        action="store_true",default=False,
                       help="disable showing of the resulting pdf file")
    parser.add_option("-e", "--exp_idx", dest="exp_idx",
                        type="int",default=1,
                       help="specify which experiment setup to use")

    (options, args) = parser.parse_args()
    #print (options, args)
    
    if options.input_path:
        if not os.path.exists(options.input_path):
            parser.error("Could not find the input directory")
    else:
        parser.error("'input' option is required to run this program")

    if not os.path.isdir(options.input_path):
        parser.error("the 'input' option should point towards " \
                     "the recording directory of the objects_detection application")

    results_path = options.input_path

    print ("show result disabled=" , options.disable_show)
    plot_caltech_evaluation(results_path, options.disable_show, options.exp_idx)

    return
        

if __name__ == '__main__':
    main()
    
    
