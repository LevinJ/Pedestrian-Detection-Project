#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os.path
from collections import namedtuple

from pyparsing import ParseException

def get_idl_line_parser():
    """
    Based on http://pyparsing.wikispaces.com/file/view/parsePythonValue.py
    """
    from pyparsing import \
    Word, ZeroOrMore, OneOrMore, Optional, oneOf, StringEnd, Suppress, Group, Combine, \
    nums, dblQuotedString, removeQuotes
    
    s = Suppress
    int_number = Combine(Optional(oneOf("+ -")) + Word(nums)).setParseAction(lambda tokens: int(tokens[0])).setName("integer")
    float_number = \
        Combine(Optional(oneOf("+ -")) + Word(nums) + Optional("." +
                   Optional(Word(nums)) +
                   Optional(oneOf("e E")+Optional(oneOf("+ -")) +Word(nums)))) \
        .setName("float") \
        .setParseAction( lambda tokens: float(tokens[0]) )
               
    bounding_box = s('(') + OneOrMore( int_number | s(',') ) + s(')')
    bounding_box_with_score = Group(bounding_box + Optional( ( s(":") | s("~") ) +  float_number ))
    #filename =  s('"') + Word(alphanums + "/_.~") +  s('"')
    quoted = dblQuotedString.setParseAction(removeQuotes)
    filename = quoted
    idl_line = filename + Optional(s(':') +  ZeroOrMore(bounding_box_with_score | s(','))) + ( s(";") | s(".") ) + StringEnd() 

    #print( filename.parseString("\"left/image_00000004_0.png\"") )
    #print( bounding_box.parseString("(221, 183, 261, 289)") )
    
    return idl_line.parseString


def fix_bounding_box(box):
    "Check and fix the box if necessary"
    
    if box[3] < box[1]:
        temp = box[1]
        box[1] = box[3]
        box[3] = temp
        
    if box[2] < box[0]:
        temp = box[0]
        box[0] = box[2]
        box[2] = temp
    
    return box

IdlLineData = namedtuple("IdlLineData", "filename bounding_boxes")

def open_idl_file(filepath):
    "Helper method returns an idl parser generator"
    
    file_basename, file_extension = os.path.splitext(filepath)
    if file_extension != ".idl":
        raise ValueError("open_idl_file expect to receive an .idl file")
    
    the_idl_file = open(filepath, "r")        
        
    idl_line_parser = get_idl_line_parser()
    
    def idl_file_parser(idl_file):
        for line in idl_file.readlines():
            try:
                elements = idl_line_parser(line)
                line_data = IdlLineData(elements[0], [fix_bounding_box(box) for box in elements[1:]])
                #print(line)
                #print(line_data)
                yield line_data            
            except ParseException as e:
                print("Failed to parse '%s' at the following line:\n%s" %(filepath, e.line))
                raise e               
    
    return idl_file_parser(the_idl_file)
    
def get_empty_idl_data_generator():
    
    def empty_idl_data_generator():
        yield IdlLineData("empty_idl_data", [])        

    return empty_idl_data_generator()   


if __name__ == '__main__':

    test_line1 = "\"left/image_00000004_0.png\": (221, 183, 261, 289), (217, 210, 234, 263);"
    test_line2 = "\"left/image_00000004_0.png\": (166, -20,183,272):0.0141471,(599,198,619,254):0.195687;"    
    test_line3 = "\"left/image_00000703_0.png\": (52, 288, 89, 178), (110, 286, 137, 190), (157, 272, 139, 199), (187, 309, 225, 178), (289, 165, 236, 322), (330, 190, 372, 303), (296, 210, 313, 265), (552, 185, 588, 310), (509, 188, 554, 317), (423, 198, 449, 297);"
    test_line4 = "\"left/image_00000020_0.png\": (246, 40, 298, 171):364.2, (394, 68, 446, 198):244, (90, 172, 139, 295):243.116;"
    idl_line_parser = get_idl_line_parser()
    print( test_line1, "\n->\n", idl_line_parser(test_line1), "\n\n" )
    print( test_line2, "\n->\n", idl_line_parser(test_line2), "\n\n" )
    print( test_line3, "\n->\n", idl_line_parser(test_line3), "\n\n" )
    print( test_line4, "\n->\n", idl_line_parser(test_line4), "\n\n" )
