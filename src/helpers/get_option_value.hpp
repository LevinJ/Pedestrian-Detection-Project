#ifndef GET_OPTION_VALUE_HPP
#define GET_OPTION_VALUE_HPP

#include <boost/program_options.hpp>
#include <string>
#include <cstdio>

template<typename T>
const T get_option_value(const boost::program_options::variables_map &options, const std::string key) {

    if(options.count(key) != 0) {
        return options[key].as<T>();
    }
    else
    {
        T dummy_value;
        printf("Could not find required program option '%s'\n", key.c_str());
        printf("Use the \033[1;32m--help\033[0m option for more details.\n");
        throw std::runtime_error("Could not find a required program option");
        return dummy_value;
    }
}

#endif // GET_OPTION_VALUE_HPP
