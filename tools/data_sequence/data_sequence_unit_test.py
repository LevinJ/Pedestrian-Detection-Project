#!/usr/bin/env python


from __future__ import print_function

from TestData_pb2 import TestData
from data_sequence import DataSequence
import os, os.path

import unittest

class TestDataSequence(unittest.TestCase):
    """
    This test is a mirror of the C++ unit test at
    doppia/src/tests/data_sequence/test_data_sequence.cpp
    """

    def setUp(self):
        self.test_filename = "test.sequence"

        self.attributes = {}
        self.attributes["name"] = "test_data_sequence"
        self.attributes["created_by"] = "data_sequence_unit_test.py"

        self.data1 = TestData()
        self.data1.int_value = 15
        self.data1.float_value = 15
        self.data1.string_value = "15"

        self.data2 = TestData()
        self.data2.int_value = 16
        self.data2.float_value = 16

        self.data3 = TestData()
        self.data3.string_value = "17"
        
        return
    
        
    def test_read_cpp_sequence(self):    
        """
        Test that python can read the file created via cpp
        """
        self.test_filename = "../../src/tests/data_sequence/test.sequence"
        self.attributes["created_by"] = "test_data_sequence.cpp"
            
        self.read_and_check()
        return

    def read_and_check(self):
        
        self.assertTrue(os.path.exists(self.test_filename))    
        self.assertTrue(os.path.getsize(self.test_filename) > 0)

        data_sequence_in = DataSequence(self.test_filename, TestData)

        read_attributes = data_sequence_in.get_attributes()
        read_data1 = data_sequence_in.read()
        read_data2 = data_sequence_in.read()
        read_data3 = data_sequence_in.read()

        self.assertEqual(read_attributes, self.attributes)
        self.assertEqual(read_data1, self.data1)
        self.assertEqual(read_data2, self.data2)
        self.assertEqual(read_data3, self.data3)
        return
    
    def test_read_write_sequence(self):
        """
        Test data sequence creation and reading
        """
        
        if os.path.exists(self.test_filename):
            os.remove(self.test_filename)
            
        data_sequence_out = \
            DataSequence(self.test_filename, TestData, self.attributes)

        data_sequence_out.write(self.data1)
        data_sequence_out.write(self.data2)
        data_sequence_out.write(self.data3)

        data_sequence_out.flush()
    
        self.read_and_check()
        return
            
if __name__ == '__main__':
    unittest.main()

