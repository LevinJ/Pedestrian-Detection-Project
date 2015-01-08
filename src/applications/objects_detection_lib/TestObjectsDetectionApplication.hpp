#ifndef OBJECTS_DETECTION_TESTOBJECTSDETECTIONAPPLICATION_HPP
#define OBJECTS_DETECTION_TESTOBJECTSDETECTIONAPPLICATION_HPP

#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <string>

namespace doppia {
// forward declarations
class AbstractVideoInput;
class ImagesFromDirectory;
}
namespace objects_detection {

// forward declaration
class FakeAbstractVideoInput;

class TestObjectsDetectionApplication
{

protected:
    // processing modules
#if defined(MONOCULAR_OBJECTS_DETECTION_LIB)
    boost::shared_ptr<FakeAbstractVideoInput> video_input_p; // dummy definition
#else // MONOCULAR_OBJECTS_DETECTION_LIB is not defined
    boost::shared_ptr<doppia::AbstractVideoInput> video_input_p;
#endif // MONOCULAR_OBJECTS_DETECTION_LIB is defined or not
    boost::scoped_ptr<doppia::ImagesFromDirectory> directory_input_p;

public:
    static boost::program_options::options_description get_args_options();

    std::string get_application_title() const;

    TestObjectsDetectionApplication();
    ~TestObjectsDetectionApplication();

    int main(int argc, char *argv[]);

    boost::program_options::variables_map parse_arguments(int argc, char *argv[]);

    void setup_problem(const boost::program_options::variables_map &options);

    void compute_solution();

protected:

    bool should_save_detections;
};


} // end of namespace objects_detection

#endif // OBJECTS_DETECTION_TESTOBJECTSDETECTIONAPPLICATION_HPP
