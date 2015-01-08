
#include "bootstrapping_lib.hpp"

#include "helpers/get_option_value.hpp"
#include "helpers/Log.hpp"

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/format.hpp>
#include <boost/foreach.hpp>

#include <string>
#include <iostream>
#include <cstdlib>

#include <omp.h>

namespace {

using namespace std;
using namespace boost;
using boost::filesystem::path;
using namespace bootstrapping;


class TestSimpleBootstrappingApplication
{

    /// storage used for the input images
    gil::gray8_image_t left_image, right_image;
    gil::gray8_image_t left_disparity_map;

    string output_depth_map_filename;

public:
    static program_options::options_description get_args_options(void);

    string get_application_title() const;

    TestSimpleBootstrappingApplication();
    ~TestSimpleBootstrappingApplication();

    int main(int argc, char *argv[]);

    program_options::variables_map parse_arguments(int argc, char *argv[]);

    void setup_problem(const program_options::variables_map &options);

    void compute_solution(const program_options::variables_map &options);

    void save_solution();

protected:

    path classifier_model_filepath;
    std::vector<std::string> negative_image_paths_to_explore;
    int max_false_positives, max_false_positives_per_image;
    std::vector<meta_datum_t> meta_data;
    std::vector<integral_channels_t> integral_images;
};


//  ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
string TestSimpleBootstrappingApplication::get_application_title() const
{
    return "Simple test program for bootstrapping_lib. Rodrigo Benenson @ KULeuven. 2011.";
}


TestSimpleBootstrappingApplication::TestSimpleBootstrappingApplication()
{
    // nothing to do here
    return;
}


TestSimpleBootstrappingApplication::~TestSimpleBootstrappingApplication()
{
    // nothing to do here
    return;
}


int TestSimpleBootstrappingApplication::main(int argc, char *argv[])
{
    cout << get_application_title() << endl;

    // obtain:
    // - left and right rectified images
    program_options::variables_map options = parse_arguments(argc, argv);
    setup_problem(options);

    compute_solution(options);

    save_solution();

    cout << "End of game, have a nice day." << endl;

    return EXIT_SUCCESS;
}


program_options::options_description TestSimpleBootstrappingApplication::get_args_options(void)
{
    program_options::options_description desc("TestSimpleBootstrappingApplication options");
    desc.add_options()

            ("model,m", program_options::value<string>(),
             "classifer model file path")

            ("images,i", program_options::value<string>(),
             "path to a directory with negative images")

            ("max_false_positives,f", program_options::value<int>()->default_value(1000),
             "maximum number of false positives to collect")
            ;

    return desc;
}


program_options::variables_map TestSimpleBootstrappingApplication::parse_arguments(int argc, char *argv[])
{

    program_options::options_description desc("Allowed options");
    desc.add_options()("help", "produces this help message");

    desc.add(TestSimpleBootstrappingApplication::get_args_options());

    program_options::variables_map options;

    try
    {
        program_options::store(program_options::parse_command_line(argc, argv, desc), options);
        program_options::notify(options);
    }
    catch (std::exception & e)
    {
        cout << "\033[1;31mError parsing the command line options:\033[0m " << e.what () << endl << endl;
        cout << desc << endl;
        exit(EXIT_FAILURE);
    }


    if (options.count("help"))
    {
        cout << desc << endl;
        exit(EXIT_SUCCESS);
    }
/*
    if ((options.count("model") + options.count("images") ) < 2 )
    {
        cout << "No classifier model or images directory specified" << endl;
        cout << desc << endl;
        exit(EXIT_FAILURE);
    }
*/
    return options;
}


void TestSimpleBootstrappingApplication::setup_problem(const program_options::variables_map &options)
{
    const bool print_log_messages = false;
    if(print_log_messages)
    {
        logging::get_log().clear(); // we reset previously existing options

        logging::LogRuleSet logging_rules;
        logging_rules.add_rule(logging::DebugMessage, "*");
        logging_rules.add_rule(logging::WarningMessage, "*");
        logging_rules.add_rule(logging::InfoMessage, "*");
        logging_rules.add_rule(logging::EveryMessage, "*");

        logging::get_log().add(std::cout, logging_rules);
    }

    using namespace boost::filesystem;

    classifier_model_filepath = get_option_value<string>(options, "model");
    // the file existence will be checked inside bootstrapping::bootstrap

    max_false_positives = get_option_value<int>(options, "max_false_positives");
    max_false_positives_per_image = -1;

    path images_directory_path = get_option_value<string>(options, "images");

    if(boost::filesystem::is_directory(images_directory_path) == false)
    {
        throw std::invalid_argument("Expected input option 'images' to point to a directory");
    }
    else
    {
        const size_t max_num_images = 10; // arbritary test number
        directory_iterator
                directory_it( images_directory_path ),
                end_it; // default construction yields past-the-end
        for ( ; directory_it != end_it and (negative_image_paths_to_explore.size() < max_num_images); ++directory_it )
        {
            const string file_extension = extension(directory_it->path().filename());
            if( (file_extension == ".png") or (file_extension == ".jpg") or (file_extension == ".jpeg"))
            {
                negative_image_paths_to_explore.push_back( directory_it->path().string() );
            }
            else
            {
                continue;
            }
        } // end of "for each file in the directory"

        printf("Selected %zi images from the directory for test processing\n",
               negative_image_paths_to_explore.size());
    }

    return;
}


void TestSimpleBootstrappingApplication::compute_solution(const program_options::variables_map &options)
{
    printf("Processing, please wait...\n");

/*    const float
            min_scale = 0.5,
            max_scale = 3;
    const int num_scales = 10;

    const float
            min_ratio = 0.7,
            max_ratio = 2;
    const int num_ratios = 10;

*/

    // same parameters as when training/testing on INRIA
    const float
            min_scale = 0.6,
            max_scale = 8.6;
            //max_scale = 3;
    const int num_scales = 55;


    const float
            min_ratio = 1,
            max_ratio = 1;
    const int num_ratios = 1;

    const bool use_less_memory = false;


    const size_t original_size = integral_images.size();
    bootstrapping::bootstrap(
                classifier_model_filepath,
                negative_image_paths_to_explore,
                max_false_positives, max_false_positives_per_image,
                min_scale, max_scale, num_scales,
                min_ratio, max_ratio, num_ratios,
                use_less_memory,
                meta_data,
                integral_images,
options);

    if(meta_data.size() != integral_images.size())
    {
        throw std::runtime_error("bootstrapping::bootstrap has an bug, meta_data.size() != integral_images.size()");
    }


    const bool save_integral_images = false;
    if(save_integral_images)
    {
        boost::filesystem::path storage_path = "/tmp/";
        boost::format filename_pattern("false_positive_%i.png");

        const int max_images_to_save = 100;

        int false_positive_counter = 0;
        BOOST_FOREACH(const integral_channels_t &integral_channels, integral_images)
        {
            const boost::filesystem::path file_path =
                    storage_path / boost::str( filename_pattern % false_positive_counter);
            doppia::save_integral_channels_to_file(integral_channels, file_path.string());

            false_positive_counter += 1;
            if(false_positive_counter  > max_images_to_save)
            {
                break; // stop the for loop
            }
        }

        printf("Saved %i false positives integral channels images inside %s\n",
               false_positive_counter, storage_path.string().c_str());

        //throw std::runtime_error("Stopping everything so you can look at the false positive integral channels images");
    }


    printf("Found %zi false positives in %zi images\n",
           integral_images.size() - original_size,
           negative_image_paths_to_explore.size());

    return;
}


void TestSimpleBootstrappingApplication::save_solution()
{
    // nothing to save

    return;
}


} // end of anonymous namespace

// -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
int main(int argc, char *argv[])
{
    int ret = EXIT_SUCCESS;

    try
    {
        boost::scoped_ptr<TestSimpleBootstrappingApplication>
                application_p( new TestSimpleBootstrappingApplication() );

        ret = application_p->main(argc, argv);
    }
    // on linux re-throw the exception in order to get the information
    catch (std::exception & e)
    {
        cout << "\033[1;31mA std::exception was raised:\033[0m " << e.what () << endl;
        ret = EXIT_FAILURE; // an error appeared
        throw;
    }
    catch (...)
    {
        cout << "\033[1;31mAn unknown exception was raised\033[0m " << endl;
        ret = EXIT_FAILURE; // an error appeared
        throw;
    }

    return ret;
}


