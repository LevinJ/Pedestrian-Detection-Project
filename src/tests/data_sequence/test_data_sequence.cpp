

/// To be used as
/// cmake ./ && make -j2 && ./test_data_sequence

#define BOOST_TEST_MODULE TestDataSequence
#include <boost/test/unit_test.hpp>

#include <boost/filesystem.hpp>
#include "helpers/data/DataSequence.hpp"

#include "TestData.pb.h"

#include <string>
#include <cstdio>
#include <iostream>


using namespace doppia;
using namespace std;
using doppia_protobuf::TestData;

typedef DataSequence<TestData> TestDataSequence;

BOOST_AUTO_TEST_CASE(WriteAndReadSmallSequenceTestCase)
{

    const string test_filename = "test.sequence";
    //if(boost::filesystem::exists(test_filename))
    {
        boost::filesystem::remove(test_filename);
    }

    // set data --
    TestDataSequence::attributes_t attributes;
    TestData data1, data2, data3;

    {
        attributes.insert(std::make_pair("name", "test_data_sequence"));
        attributes.insert(std::make_pair("created_by", "test_data_sequence.cpp"));

        data1.set_int_value(15);
        data1.set_float_value(15);
        data1.set_string_value("15");
        data2.set_int_value(16);
        data2.set_float_value(16);
        data3.set_string_value("17");
    }

    // write output data --
    {
        TestDataSequence output_sequence(test_filename, attributes);

        output_sequence << data1;
        output_sequence << data2;
        output_sequence << data3;
    }

    BOOST_REQUIRE(boost::filesystem::exists(test_filename));
    BOOST_REQUIRE(boost::filesystem::file_size(test_filename) > 0);

    // read input data --
    {
        TestDataSequence input_sequence(test_filename);

        TestData read_data1, read_data2, read_data3;
        TestDataSequence::attributes_t read_attributes;

        read_attributes = input_sequence.get_attributes();
        input_sequence >> read_data1;
        input_sequence >> read_data2;
        input_sequence >> read_data3;

        BOOST_CHECK(read_attributes == attributes);

        BOOST_CHECK(read_data1.int_value() == data1.int_value());
        BOOST_CHECK(read_data1.float_value() == data1.float_value());
        BOOST_CHECK(read_data1.string_value() == data1.string_value());

        BOOST_CHECK(read_data2.int_value() == data2.int_value());
        BOOST_CHECK(read_data2.float_value() == data2.float_value());
        BOOST_CHECK(read_data2.has_string_value() == data2.has_string_value());

        BOOST_CHECK(read_data3.has_int_value() == data3.has_int_value());
        BOOST_CHECK(read_data3.has_float_value() == data3.has_int_value());
        BOOST_CHECK(read_data3.string_value() == data3.string_value());

        //BOOST_REQUIRE_MESSAGE(estimation_is_correct, "estimated boundary should be identical to expected_boundary boundary");
    }


    return;
} // end of "BOOST_AUTO_TEST_CASE"



