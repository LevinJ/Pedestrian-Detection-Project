#ifndef TEST_STIXEL_WORLD_APPLICATION_HPP
#define TEST_STIXEL_WORLD_APPLICATION_HPP

#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <string>

namespace doppia {
// forward declarations
class AbstractVideoInput;
}


namespace stixel_world {

class TestStixelWorldApplication
{

protected:
    // processing modules
    boost::shared_ptr<doppia::AbstractVideoInput> video_input_p;

public:
    static boost::program_options::options_description get_args_options();

    std::string get_application_title() const;

    TestStixelWorldApplication();
    ~TestStixelWorldApplication();

    int main(int argc, char *argv[]);

    boost::program_options::variables_map parse_arguments(int argc, char *argv[]);

    void setup_problem(const boost::program_options::variables_map &options);

    void compute_solution();

protected:

    bool should_save_detections;
};


} // end of namespace stixel_world

#endif // TEST_STIXEL_WORLD_APPLICATION_HPP
