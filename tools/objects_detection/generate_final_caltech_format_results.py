'''
Created on Dec 4, 2014

@author: Zhirong Jian
''' 
import os.path
import glob
import shutil
import re


def write_file_force(resultFilePath, lines):
    """Write data to a file, create the file and corresponding direcotry if they do not already exist."""
    resultfiledir = os.path.dirname(resultFilePath)
    if not os.path.exists(resultfiledir):
        os.makedirs(resultfiledir)
    #text_file = open(resultFilePath, "a+") # append to the file
    with open(resultFilePath,"a+",) as f:
        f.writelines(lines)

def get_path_and_id(infile):
    """Use Regular expression to extract result file path and image id. 
    
    For example, the input could be  ./convertedcaltechformat/set06_V000_I00029.txt
    After the processing of this function, three values will be extracted from base name of the file
    set06, V000, 29
    and the  output will be
    {"result_file_path": "./convertedcaltechformat/set06/V000.txt", "imageid_to_insert":"29"}
    """
    filebasename = os.path.basename(infile)
    p = re.compile("(^set\d*)_(V\d*)_I(.*)\.txt$")
    m = p.match(filebasename)
    if m:
        # for caltech pedestrain images
        setfolderName = m.group(1)
        VfileName = m.group(2)
        imageid = str(int (m.group(3)) + 1)
        result_file_path = os.path.dirname(infile)+ "/res/" + setfolderName + "/"+VfileName+ ".txt"
    else:
        #for Inria pedestrian images
        p = re.compile("^I(.*)\.txt$")
        m = p.match(filebasename)
        VfileName = "V000"
        imageid = str(int (m.group(1)) + 1)
        result_file_path = os.path.dirname(infile)+ "/res/" + VfileName+ ".txt"
    return {"result_file_path": result_file_path, "imageid_to_insert": imageid}
    
    
def generate_result_files(input_path):
    """Generate detection result files which are required by Caltech pedestrian detection evaluation tools."""
    filelist = glob.glob(os.path.join(input_path, '*.txt'))
    for infile in sorted(filelist): 
        #do some fancy stuff
        print str(infile) 
        respathid = get_path_and_id(infile)
        with open(infile) as f:
            linelist =[]
            for line in f:
                linelist.append(respathid["imageid_to_insert"] + "," + line)
            print(respathid["result_file_path"],linelist)
            write_file_force(respathid["result_file_path"],linelist)

def debug_this_script():   
    """A helper function that allows for easy debugging of this script"""     
    input_path= "./detection_result/baseline/Inriamodel-Inriaimages/2014_12_09_79209_recordings/caltechformat"  
    #output_path= "./detection_result/baseline/Inriamodel-Caltechimages/2014_12_09_77307_recordings/caltechformat" 
    res_output_path =  input_path + "/res"
    if os.path.exists(res_output_path):
        #parser.error("output_path should point to a non existing directory")
        shutil.rmtree(res_output_path)
    generate_result_files(input_path)
    
#debug_this_script()