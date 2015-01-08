#ifndef ADD_TO_PARSED_OPTIONS_HPP
#define ADD_TO_PARSED_OPTIONS_HPP

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <vector>
#include <string>

template <typename T>
        void add_to_parsed_options(boost::program_options::parsed_options &the_parsed_options, const std::string key, const T value)
{

    std::vector<std::string> t_vector;
    t_vector.clear();
    t_vector.push_back(boost::lexical_cast<std::string>(value));

    // FIXME should check that key does not already exists
    the_parsed_options.options.push_back( boost::program_options::option(key, t_vector) );

    return;
}


#endif // ADD_TO_PARSED_OPTIONS_HPP
