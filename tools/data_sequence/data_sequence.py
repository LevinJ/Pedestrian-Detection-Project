#!/usr/bin/env python


from __future__ import print_function

from DataSequenceHeader_pb2 import DataSequenceHeader
from google.protobuf.message import DecodeError

import os.path
from struct import pack, unpack

class DataSequence:
    """
    This is the python mirror of the c++ DataSequence class
    See doppia/src/helpers/data/DataSequence.hpp for more information
    """
    def __init__(self, filename, data_type, attributes = {}):
        """
        if no attributes are given then the file is opened in read more,
        else it is opened in write mode
        """

        self.data_type = data_type

        if attributes:
            # write mode
            self.file = open(filename, "wb")
            self.attributes = attributes
            self._write_header()
        else:
            # read mode
            assert os.path.exists(filename)
            self.file = open(filename, "rb")
            self.attributes = {}
            self._read_header()

        return

    def _read_header(self):
        
        assert self.file.mode == "rb"
        
        # size is saved using WriteLittleEndian64, 64 / 8 == 8
        # see http://docs.python.org/library/struct.html#byte-order-size-and-alignment

        raw_little_endian_int64 = self.file.read(8)                
        size = unpack("<Q", raw_little_endian_int64)[0]

        header_string = self.file.read(size)    
        self.header = DataSequenceHeader()    
        self.header.ParseFromString(header_string)

        for attribute in self.header.attributes:
            self.attributes[attribute.name] = attribute.value     

        return

    def _write_header(self):

        assert self.file.mode == "wb"

        # set the header content --        
        self.header = DataSequenceHeader()

        for attribute_name, attribute_value in self.attributes.items():
            attribute = self.header.attributes.add()
            attribute.name = attribute_name
            attribute.value = attribute_value  
        
        # serialize --
        header_string = self.header.SerializeToString()
        size = len(header_string)             
        raw_little_endian_int64 = pack("<Q", size)
        
        # write the data to the file --
        self.file.write(raw_little_endian_int64)
        self.file.write(header_string)

        return


    def get_attributes(self):
        return self.attributes

    def read(self):
        """
        Reads a new message
        """
        
        assert self.file.mode == "rb"
         
        # size is saved using WriteLittleEndian64, 64 / 8 == 8
        # see http://docs.python.org/library/struct.html#byte-order-size-and-alignment

        raw_little_endian_int64 = self.file.read(8)        
        if not raw_little_endian_int64:
            # end of file
            return None
        size = unpack("<Q", raw_little_endian_int64)[0]
        data_string = self.file.read(size)    
       
        if raw_little_endian_int64 and data_string:
            data = self.data_type()
            try:
                data.ParseFromString(data_string)
            except DecodeError as error:
                print("Ending read on google.protobuf.message.DecodeError", error)
                #print("Last read data_string == '%s'" % data_string)
                print("Last read data_string has size == ", len(data_string))
                data = None
        else:
            data = None
        return data

    def write(self, data):
        """
        Writes a new message into the data sequence file
        """
        
        assert isinstance(data, self.data_type)
        assert self.file.mode == "wb"

        # serialize --
        data_string = data.SerializeToString()
        size = len(data_string)             
        raw_little_endian_int64 = pack("<Q", size)
        
        # write the data to the file --
        self.file.write(raw_little_endian_int64)
        self.file.write(data_string)
        
        return

    def flush(self):
        self.file.flush()
        return

