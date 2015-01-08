#include "BoostedLearningApplication.hpp"

#include "boosted_learning/ModelIO.hpp"

#include "applications/EmptyGui.hpp"
#include "objects_detection/integral_channels/AbstractIntegralChannelsComputer.hpp"
#include "objects_detection/integral_channels/IntegralChannelsComputerFactory.hpp"
#include "objects_detection/integral_channels/IntegralChannelsForPedestrians.hpp"
#include "video_input/ImagesFromDirectory.hpp"

#include "helpers/get_option_value.hpp"
#include "helpers/ModuleLog.hpp"
#include "helpers/progress_display_with_eta.hpp"
#include "helpers/ModuleLog.hpp"
//#include "helpers/geometry.hpp"

#include <boost/gil/extension/io/png_io.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/static_assert.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/format.hpp>
#include <boost/array.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/timer.hpp>
#include <boost/format.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <omp.h>
#include <string>
#include <stdexcept>
#include <sstream>
#include <ctime>

namespace doppia {

using namespace std;
using namespace boost;
using namespace boosted_learning;

MODULE_LOG_MACRO("BoostedLearningApplication")
//  ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

std::string BoostedLearningApplication::get_application_title()
{
    return  "Trains a classifier with bootstrapping "
            "(M. Mathias, R. Benenson, M. Omran) @ KULeuven & MPI-Inf. 2011-2014.";
}


BoostedLearningApplication::BoostedLearningApplication()
    : BaseApplication()
{
    // nothing to do here
    return;
}


BoostedLearningApplication::~BoostedLearningApplication()
{
    // nothing to do here
    return;
}



TrainingData::point_t get_model_windowSize(const program_options::variables_map &options)
{
    const std::string modelWindowString= get_option_value<std::string>(options, "train.model_window");

    //get model window
    std::vector<std::string> modelWindowStrings;
    boost::split(modelWindowStrings, modelWindowString, boost::is_any_of(","));
    assert(modelWindowStrings.size() == 2);
    TrainingData::point_t modelWindow(boost::lexical_cast<int>(modelWindowStrings[0]),
            boost::lexical_cast<int>(modelWindowStrings[1]));

    return modelWindow;
}


TrainingData::point_t getTrainingDataOffset(const program_options::variables_map &options)
{
    const int
            offsetX = get_option_value<int>(options, "train.offset_x"),
            offsetY = get_option_value<int>(options, "train.offset_y");

    TrainingData::point_t testOffset(offsetX, offsetY);
    return testOffset;
}


TrainingData::point_t getTestingDataOffset(const program_options::variables_map &options)
{
    const int
            offsetX = get_option_value<int>(options, "test.offset_x"),
            offsetY = get_option_value<int>(options, "test.offset_y");

    TrainingData::point_t testOffset(offsetX, offsetY);
    return testOffset;
}


TrainingData::rectangle_t get_object_window(const program_options::variables_map &options)
{
    //read files for training
    const std::string objectWindowString = get_option_value<std::string>(options, "train.object_window");

    //get object window
    std::vector<std::string> objectWindowStrings;
    boost::split(objectWindowStrings, objectWindowString, boost::is_any_of(","));
    assert(objectWindowStrings.size() == 4);
    doppia::geometry::point_xy<int> mincorner(boost::lexical_cast<int>(objectWindowStrings[0]),
            boost::lexical_cast<int>(objectWindowStrings[1]));
    doppia::geometry::point_xy<int> maxcorner(mincorner.x() + boost::lexical_cast<int>(objectWindowStrings[2]),
            mincorner.y() + boost::lexical_cast<int>(objectWindowStrings[3]));
    TrainingData::rectangle_t  objectWindow(mincorner, maxcorner);

    return objectWindow;
}

void retrieveFilesNames(const boost::filesystem::path directoryPath, std::vector<std::string> &filesNames)
{
    using namespace boost::filesystem;

    directory_iterator
            directoryIterator = directory_iterator(directoryPath),
            directoryEnd;
    while(directoryIterator != directoryEnd)
    {
        const string
        #if BOOST_VERSION <= 104400
                fileName = directoryIterator->path().filename();
#else
                fileName = directoryIterator->path().filename().string();
#endif

        // should we check the filePath extension ?
        const path filePath = directoryPath / fileName;
        filesNames.push_back(filePath.string());

        ++directoryIterator;
    } // end of "for each file in the directory"

    return;
}


void getImageFileNames(
        const string nameFile,
        const int backgroundClassLabel,
        std::vector<std::string> &positiveExamplesPaths,
        std::vector<std::string> &negativeExamplesPaths)
{
    using boost::filesystem::path;
    using boost::filesystem::is_directory;

    path inputPath(nameFile);
    if(is_directory(inputPath))
    {
        // handling the INRIAPerson case http://pascal.inrialpes.fr/data/human
        const path
                inriaPersonPositivesPath = inputPath / "pos",
                inriaPersonNegativesPath = inputPath / "neg";

        const bool isInriaPedestriansDirectory = is_directory(inriaPersonPositivesPath) and is_directory(inriaPersonNegativesPath);

        const string
                positivePathPrefix = "positives_octave_",
        #if BOOST_VERSION <= 104400
                inputPathFilename = inputPath.filename();
#else
                inputPathFilename = inputPath.filename().string();
#endif
        float octave_number = 0;
        if(boost::algorithm::starts_with(inputPathFilename, positivePathPrefix))
        {
            // inputPathFilename should be something like "positives_octave_-2.0"
            const string number_string =
                    inputPathFilename.substr(positivePathPrefix.size(),
                                             inputPathFilename.size() - positivePathPrefix.size());
            octave_number = boost::lexical_cast<float>(number_string);
        }

        const path
                multiScalesPositivesPath = inputPath,
                multiScalesNegativesPath = \
                inputPath.parent_path() / boost::str(boost::format("negatives_octave_%.1f") % octave_number);

        const bool isMultiScalesDirectory =
                is_directory(multiScalesPositivesPath) and
                boost::algorithm::starts_with(inputPathFilename, positivePathPrefix) and
                is_directory(multiScalesNegativesPath);

        if(true and (not isInriaPedestriansDirectory) and (not isMultiScalesDirectory))
        { // just for debugging
            printf("is_directory(multiScalesPositivesPath) == %i\n", is_directory(multiScalesPositivesPath));
            printf("is_directory(%s) == %i\n",
                   multiScalesNegativesPath.string().c_str(), is_directory(multiScalesNegativesPath));
#if BOOST_VERSION <= 104400
            printf("starts_with(%s, %s) == %i\n",
                   std::string(multiScalesPositivesPath.filename()).c_str(), positivePathPrefix.c_str(),
                   boost::algorithm::starts_with(multiScalesPositivesPath.filename(), positivePathPrefix));
#else
            printf("starts_with(%s, %s) == %i\n",
                   multiScalesPositivesPath.filename().string().c_str(), positivePathPrefix.c_str(),
                   boost::algorithm::starts_with(multiScalesPositivesPath.filename().string(), positivePathPrefix));
#endif
        }

        if(isInriaPedestriansDirectory)
        {
            retrieveFilesNames(inriaPersonPositivesPath, positiveExamplesPaths);
            retrieveFilesNames(inriaPersonNegativesPath, negativeExamplesPaths);


        }
        else if(isMultiScalesDirectory)
        {
            retrieveFilesNames(multiScalesPositivesPath, positiveExamplesPaths);
            retrieveFilesNames(multiScalesNegativesPath, negativeExamplesPaths);
        }
        else
        {
            printf("LabeledData::getImageFileNames is trying to read %s\n", nameFile.c_str());
            throw std::invalid_argument("LabeledData::getImageFileNames received a directory, but "
                                        "could not recognize it as an INRIAPerson Train/Test directory "
                                        "nor as multiscales_inria_person/data_set/positives_octave_* directory");
        }

    }
    else
    { // input path is a list file

        ifstream inFile(nameFile.c_str());

        if (!inFile.is_open())
        {
            printf("LabeledData::getImageFileNames is trying to read %s\n", nameFile.c_str());
            throw std::invalid_argument("LabeledData::getImageFileNames could not open the indicated file");
        }


        // just read the file
        while (!inFile.eof())
        {
            //get next file

            int  tmpClassNum;
            inFile >> tmpClassNum; // store class

            // additional check to avoid problems in the case of an empty line at the end
            // of the file
            if (inFile.eof())
            {
                break;
            }

            string filename;
            inFile >> filename;

            //if (_verbose > 2)
            if(false)
            {
                std::cout << filename << std::endl;
            }

            if (tmpClassNum != backgroundClassLabel)
            {
                positiveExamplesPaths.push_back(filename);
            }
            else
            {
                negativeExamplesPaths.push_back(filename);
            }

        } // end of "while reading file"

    } // end of "if nameFile is a directory or not"

    return;
} // end of getImageFileNames


void getImageFileNames(
        const string nameFile,
        const int backgroundClassLabel,
        std::vector<std::string> &positiveExamplesPaths,
        std::vector<std::string> &negativeExamplesPaths,
        std::vector<std::string> &hardNegativeExamplesPaths,
        const program_options::variables_map &options)
{
    using boost::filesystem::path;
    using boost::filesystem::is_directory;
    const bool useHardNegatives = get_option_value<bool>(options, "train.use_hard_negatives");

    path inputPath(nameFile);
    if(is_directory(inputPath))
    {
        // handling the INRIAPerson case http://pascal.inrialpes.fr/data/human
        const path
                inriaPersonPositivesPath = inputPath / "pos",
                inriaPersonNegativesPath = inputPath / "neg";

        const bool isInriaPedestriansDirectory = \
                is_directory(inriaPersonPositivesPath) and is_directory(inriaPersonNegativesPath);

        const string
                positivePathPrefix = "positives_octave_",
        #if BOOST_VERSION <= 104400
                inputPathFilename = inputPath.filename();
#else
                inputPathFilename = inputPath.filename().string();
#endif
        float octave_number = 0;
        if(boost::algorithm::starts_with(inputPathFilename, positivePathPrefix))
        {
            // inputPathFilename should be something like "positives_octave_-2.0"
            const string number_string =
                    inputPathFilename.substr(positivePathPrefix.size(),
                                             inputPathFilename.size() - positivePathPrefix.size());
            octave_number = boost::lexical_cast<float>(number_string);
        }

        const path
                multiScalesPositivesPath = inputPath,
                multiScalesNegativesPath = \
                inputPath.parent_path() / boost::str(boost::format("negatives_octave_%.1f") % octave_number),
                multiScalesHardNegativesPath = \
                inputPath.parent_path() / boost::str(boost::format("hard_negatives_octave_%.1f") % octave_number);

        const bool isMultiScalesDirectory =
                is_directory(multiScalesPositivesPath) and
                boost::algorithm::starts_with(inputPathFilename, positivePathPrefix) and
                is_directory(multiScalesNegativesPath);

        if(true and (not isInriaPedestriansDirectory) and (not isMultiScalesDirectory))
        { // just for debugging
            printf("is_directory(multiScalesPositivesPath) == %i\n", is_directory(multiScalesPositivesPath));
            printf("is_directory(%s) == %i\n",
                   multiScalesNegativesPath.string().c_str(), is_directory(multiScalesNegativesPath));
#if BOOST_VERSION <= 104400
            printf("starts_with(%s, %s) == %i\n",
                   std::string(multiScalesPositivesPath.filename()).c_str(), positivePathPrefix.c_str(),
                   boost::algorithm::starts_with(multiScalesPositivesPath.filename(), positivePathPrefix));
#else
            printf("starts_with(%s, %s) == %i\n",
                   multiScalesPositivesPath.filename().string().c_str(), positivePathPrefix.c_str(),
                   boost::algorithm::starts_with(multiScalesPositivesPath.filename().string(), positivePathPrefix));
#endif
        }

        if(isInriaPedestriansDirectory)
        {
            retrieveFilesNames(inriaPersonPositivesPath, positiveExamplesPaths);
            retrieveFilesNames(inriaPersonNegativesPath, negativeExamplesPaths);
        }
        else if(isMultiScalesDirectory)
        {
            retrieveFilesNames(multiScalesPositivesPath, positiveExamplesPaths);
            retrieveFilesNames(multiScalesNegativesPath, negativeExamplesPaths);
            if(useHardNegatives)
            {
                if (is_directory(multiScalesHardNegativesPath))
                {
                    retrieveFilesNames(multiScalesHardNegativesPath, hardNegativeExamplesPaths);
                }
                else
                {
                    std::cout << "directory: " << multiScalesHardNegativesPath << std::endl << std::flush;
                    throw runtime_error("requested hard negatives, but no hard negatives found!");
                }
            }
        }
        else
        {
            printf("LabeledData::getImageFileNames is trying to read %s\n", nameFile.c_str());
            throw std::invalid_argument("LabeledData::getImageFileNames received a directory, but "
                                        "could not recognize it as an INRIAPerson Train/Test directory "
                                        "nor as multiscales_inria_person/data_set/positives_octave_* directory");
        }

    }
    else
    { // input path is a list file
        if(useHardNegatives)
            throw std::runtime_error("useHardNegatives is only implemented when using octave_specific directory structure!");
        ifstream inFile(nameFile.c_str());

        if (!inFile.is_open())
        {
            printf("LabeledData::getImageFileNames is trying to read %s\n", nameFile.c_str());
            throw std::invalid_argument("LabeledData::getImageFileNames could not open the indicated file");
        }


        // just read the file
        while (!inFile.eof())
        {
            //get next file

            int  tmpClassNum;
            inFile >> tmpClassNum; // store class

            // additional check to avoid problems in the case of an empty line at the end
            // of the file
            if (inFile.eof())
            {
                break;
            }

            string filename;
            inFile >> filename;

            //if (_verbose > 2)
            if(false)
            {
                std::cout << filename << std::endl;
            }

            if (tmpClassNum != backgroundClassLabel)
            {
                positiveExamplesPaths.push_back(filename);
            }
            else
            {
                negativeExamplesPaths.push_back(filename);
            }

        } // end of "while reading file"

    } // end of "if nameFile is a directory or not"

    return;
} // end of getImageFileNames


program_options::options_description BoostedLearningApplication::get_options_description()
{
    using namespace boost::program_options;

    options_description desc("BoostedLearningApplication options");

    const std::string application_name = "boosted_learning";
    BaseApplication::add_args_options(desc, application_name);

    desc.add_options()

            ("task,t",
             value<std::string>()->default_value("bootstrap_train"),
             "the task to execute (you probably want to use bootstrap_train)")

            ("silent_mode",
             value<bool>()->default_value(false),
             "if true, no status information will be printed at run time (use this for speed benchmarking)")
            ;

    {
        options_description training_options("Training task options");

        training_options.add_options()

                ("train.training_set_name",
                 value<std::string>()->default_value("INRIAPerson"),
                 "String identifying the training set used. This for human consuption.")

                ("train.training_set",
                 value<std::string>(),
                 "File containing the Training set, each line <class> <image_path>")

                ("train.background_class_label",
                 value<int>()->default_value(0),
                 "label for the background class")

                ("train.test_set",
                 value<std::string>(),
                 "File containing the Testing set, each line <class> <image_path>")

                ("train.extended_training_set",
                 value<std::string>()->default_value(""),
                 "File containing the a bigger training set used starting from the first "
                 "bootstrapping stage, each line <class> <image_path>")

                ("train.features_configurations_file",
                 value<std::string>()->default_value("/users/visics/mmathias/devel/doppia/tools/objects_detection/bestFeat.txt"),
                 "If feature type == file, then this file is used for the featurepool")

                ("train.validation_set",
                 value<std::string>()->default_value(""),
                 "File containing the Testing set, each line <class> <image_path>")

                ("train.model_window",
                 value<std::string>()->default_value("64,128"),
                 "Window size of the training images (w,h)")

                ("train.num_iterations",
                 value<int>(),
                 "number of iterations for training."
                 "For vanilla adaboost this defines the number of weak classifiers used in the strong classifier.")

                ("train.num_negative_samples",
                 value<int>()->default_value(5000),
                 "Number of negative samples to be used for the training")

                ("train.use_hard_negatives",
                 value<bool>()->default_value(false),
                 "use hard negatives for training")

                ("train.object_window",
                 value<std::string>()->default_value("8,16,48,96"),
                 "bounding box of the object in (x,y,w,h)")

                ("train.offset_x",
                 value<int>(),
                 "offset in X direction between the training image border and the model window")

                ("train.offset_y",
                 value<int>(),
                 "offset in y direction between the training image border and the model window")

                ("train.bootstrap_learner_file",
                 value<std::string>()->default_value(std::string()),
                 "File with the learner used for bootstrapping")

                ("train.type_adaboost", value<std::string>()->default_value("vanilla"),
                 "Type of the Adaboost used: vanilla or vanilla")
                // FIXME this option is not really used, should add checks in the code
                //  default, GENTLE_ADA, other=DISCRETE_ADA

                ("train.decision_tree_depth",
                 value<int>()->default_value(1),
                 "depth of the decision trees (0 equals decision stump)")

                ("train.cascade_type",
                 value<std::string>()->default_value("dbp"),
                 "Type of the soft cascade: none, dbp or mip."
                 "dbp stands for Direct Backward Prunning (see C. Zang and P. Viola 2007).")

                ("train.min_feature_width",
                 value<int>()->default_value(1),
                 "minimal width of the features to train on (in pixels of the original image)")

                ("train.min_feature_height",
                 value<int>()->default_value(1),
                 "minimal height of the features to train on (in pixels of the original image)")

                ("train.max_feature_width",
                 value<int>()->default_value(-1),
                 "maximum width of the features to train on (in pixels of the original image). "
                 "If negative value, no limit is imposed.")

                ("train.max_feature_height",
                 value<int>()->default_value(-1),
                 "maximum height of the features to train on (in pixels of the original image). "
                 "If negative value, no limit is imposed.")

                ("train.pushUpBias", value<weights_t::value_type>()->default_value(0.0),
                 "biasing the features to be located in the upper half of the object window: 0.0 means not pushing up at all 1.0 results in a half (top) classifier")

                ("train.pushLeftBias", value<weights_t::value_type>()->default_value(0.0),
                 "biasing the features to be located in the left half of the object window: 0.0 means not pushing left at all 1.0 results in a half (left) classifier")


                // FIXME is this ever used ?
                //("train.max_feature_size_ratio", value<float>(),
                // "defines the maximal size of a feature: 0.6 means max 60% of the training image size")

                ("train.feature_pool_size",
                 value<int>(),
                 "Size of the set of features used to build up weak classifiers at each iteration of the boosting algorithm")

                ("train.feature_pool_type",
                 value<std::string>()->default_value("random"),
                 "specifies the type of sampling for the feature pool. "
                 "Possible values are:\n"
                 "random; please take care of which random seed is being used\n"
                 "file; will read the feature pool from a file (ignores featuresPoolSize)\n"
                 "HOG-like; all regularly spaced squares of fixed size (ignores featuresPoolSize)\n"
                 "HOG-multiScale; regularly spaced squares of all possible sizes (ignores featuresPoolSize)\n"
                 "channelInvariant; randomly sample squares on one channel, copy to all other ones\n"
                 "gradientMirrored; ??\n"
                 "gradientMirroredCopy; ??\n"
                 "verticalSymmetry; each random rectangle has it vertical mirror\n"
                 "allRectangles; please be carefull, this option will be very memory hungry (60 Gb or more)"
                 "(ignores featuresPoolSize, min/max feature width/height/ratio)\n")

                ("train.use_stumpsets",
                 value<bool>()->default_value(false),
                 "enables stumpset training instead of standard adaboost")

                ("train.use_dropout",
                 value<bool>()->default_value(false),
                 "randomly dropout classifiers to prevent training from overfitting")

                ("train.reset_beta",
                 value<bool>()->default_value(true),
                 "set true to reweight the features, after removing some")

                ("train.removeHurtingFeatures",
                 value<bool>()->default_value(true),
                 "set true to remove as well features, that generate a training error > 0.5")

                ("train.feature_pool_random_seed",
                 value<boost::uint32_t>()->default_value(0),
                 "random seed used to generate the features pool. If the value is 0 the current time is used as seed (recommended)."
                 "Fixing the value to a value > 0 allows to repeat trainings with the same set of features "
                 "(but negative samples still randomly sampled)")

                ("train.output_model_filename",
                 value<std::string>(),
                 "file to write the trained detector into")

                ("train.channels_folder_path",
                 value<std::string>(),
                 "path to the folder containing the channel features computed for the train and test set (if any)")

                ;

        desc.add(training_options);
    }

    {
        options_description testing_options("Testing task options");

        testing_options.add_options()
                ("test.classifier_name",
                 value<std::string>(),
                 "filename of the classifier to test on")

                ("test.test_set",
                 value<std::string>(),
                 "File containing the Testing set, each line <class> <image_path>")

                ("test.offset_x",
                 value<int>()->default_value(3), // default value fits INRIAPerson
                 "offset in X direction between the testing image border and the model window")

                ("test.offset_y",
                 value<int>()->default_value(3), // default value fits INRIAPerson
                 "offset in y direction between the testing image border and the model window")

                ;
        desc.add(testing_options);
    }

    {
        options_description boostrapping_options("bootstrap_train task options");

        boostrapping_options.add_options()
                ("bootstrap_train.classifiers_per_stage",
                 value<std::vector<int> >()->multitoken(),
                 "List of number of classifiers trained per stage.")

                ("bootstrap_train.max_num_samples_per_image",
                 value<std::vector<int> >()->multitoken(),
                 "List of number of samples per image allowed for bootstrapping")

                ("bootstrap_train.num_bootstrapping_samples",
                 value<int>()->default_value(5000),
                 "Number of samples to be searched by bootstrapping")

                ("bootstrap_train.min_scale",
                 value<float>()->default_value(0.3),
                 "minimum detection window scale explored for detections")

                ("bootstrap_train.max_scale",
                 value<float>()->default_value(5.0),
                 "maximum detection window scale explored for detections")

                ("bootstrap_train.num_scales",
                 value<int>()->default_value(10),
                 "number of scales to explore. (this is combined with num_ratios)")

                ("bootstrap_train.min_ratio",
                 value<float>()->default_value(1),
                 "minimum ratio (width/height) of the detection window used to explore detections")

                ("bootstrap_train.max_ratio",
                 value<float>()->default_value(1),
                 "max ratio (width/height) of the detection window used to explore detections")

                ("bootstrap_train.num_ratios",
                 value<int>()->default_value(1),
                 "number of ratios to explore. (this is combined with num_scales)")

                ("bootstrap_train.frugal_memory_usage",
                 value<bool>()->default_value(false),
                 "By default we use as much GPU memory as useful for speeding things up. "
                 "If frugal memory usage is enabled, we will reduce the memory usage, "
                 "at the cost of longer computation time.")

                ;

        desc.add(boostrapping_options);
    }

    return desc;
} // end of BoostedLearningApplication::get_args_options


void BoostedLearningApplication::get_all_options_descriptions(program_options::options_description &desc)
{
    desc.add(BoostedLearningApplication::get_options_description());
    desc.add(IntegralChannelsComputerFactory::get_options_description());
    desc.add(IntegralChannelsForPedestrians::get_options_description());
    return;
}

/// helper method called by setup_problem
void BoostedLearningApplication::setup_logging(std::ofstream &log_file,
                                               const program_options::variables_map &options)
{
    if(log_file.is_open())
    {
        // the logging is already setup
        return;
    }

    // set base logging rules --
    BaseApplication::setup_logging(log_file, options);

    const bool silent_mode = get_option_value<bool>(options, "silent_mode");
    if(silent_mode == false)
    {
        // set our own stdout rules --
        logging::LogRuleSet &rules_for_stdout = logging::get_log().console_log().rule_set();
        rules_for_stdout.add_rule(logging::InfoMessage, "BoostedLearningApplication");
#if defined(DEBUG)
        rules_for_stdout.add_rule(logging::DebugMessage, "*"); // we are debugging this application
#else
        // "production mode"
        rules_for_stdout.add_rule(logging::InfoMessage, "BoostedLearningApplication");
        rules_for_stdout.add_rule(logging::InfoMessage, "*Factory");
        rules_for_stdout.add_rule(logging::InfoMessage, "WeakLearner");
        rules_for_stdout.add_rule(logging::InfoMessage, "SoftCascadeOverIntegralChannelsModel");
#endif

    }

    return;
}


/*
int getNumChannels(TrainingData::integral_channels_computer_shared_ptr_t &integral_channels_computer_p,
                   const std::string &training_example_path)
{
    boost::gil::rgb8_image_t image;
    boost::gil::rgb8c_view_t image_view = doppia::open_image(training_example_path.c_str(), image);

    const boost::filesystem::path file_path = training_example_path;
#if BOOST_VERSION <= 104400
    const std::string filename = file_path.filename();
#else
    const std::string filename = file_path.filename().string();
#endif
    integral_channels_computer_p->set_image(image_view, filename);
    integral_channels_computer_p->compute();

    const doppia::IntegralChannelsFromFiles::integral_channels_t &
            t_integral_channels = integral_channels_computer_p->get_integral_channels();

    const int numChannels = t_integral_channels.shape()[0];

    if(numChannels <= 0)
    {
        printf("getNumChannels thinks that we will compute %i channels, which seems wrong\n",
               numChannels);
        throw std::runtime_error("Failed to estimate the number of integral channels");
    }

    return numChannels;
}*/



void BoostedLearningApplication::setup_problem(const program_options::variables_map &options)
{

    // parse the application specific options --
    silent_mode = get_option_value<bool>(options, "silent_mode");
    task = get_option_value<std::string>(options, "task");

    //integral_channels_computer_p.reset(ChannelsComputerFactory::new_instance(options));

    /*if(task == "train")
    {
        const std::string channels_folder_path = get_option_value<std::string>(options, "train.channels_folder_path");
        integral_channels_computer_p(new doppia::IntegralChannelsFromFiles(channels_folder_path));
    }*/


    // setting up training parameters

    // used in bootstrapTrain(), bootstrapTrainResume(), read files for training
    trainSetPath = get_option_value<std::string>(options, "train.training_set"),
            testSetPath = get_option_value<std::string>(options, "train.test_set"),
            validationSetPath = get_option_value<std::string>(options, "train.validation_set");
    extendedTrainSetPath = get_option_value<std::string>(options, "train.extended_training_set");
    numIterations = get_option_value<int>(options, "train.num_iterations");
    trainNumNegativeSamples = get_option_value<int>(options, "train.num_negative_samples");
    initialBootstrapFileName = get_option_value<std::string>(options, "train.bootstrap_learner_file");
    //channels_folder_path = get_option_value<std::string>(options, "train.channels_folder_path");
    featuresPoolSize = get_option_value<int>(options, "train.feature_pool_size");
    featurePoolType = get_option_value<std::string>(options, "train.feature_pool_type");
    useStumpSets = get_option_value<bool>(options, "train.use_stumpsets");
    typeAdaboost = get_option_value<string>(options, "train.type_adaboost");

    // setting up bootstrapping parameters

    // used in bootstrapTrain(), bootstrapTrainResume()
    numBootstrappingSamples = get_option_value<int>(options, "bootstrap_train.num_bootstrapping_samples");
    classifiersPerStage = get_option_value<std::vector<int> >(options, "bootstrap_train.classifiers_per_stage");
    maxNumSamplesPerImage = get_option_value<std::vector<int> >(options, "bootstrap_train.max_num_samples_per_image");

    backgroundClassLabel = get_option_value<int>(options, "train.background_class_label");




    // setting up testing parameters

    // used in test(), getTestingDataOffset() (<-- possibly redundant, when is it called?)
    test_offsetX = get_option_value<int>(options, "test.offset_x");
    test_offsetY = get_option_value<int>(options, "test.offset_y");

    // used in test(), read files for training
    //testSetPath = get_option_value<std::string>(options, "test.test_set"),
    classifierName = get_option_value<std::string>(options, "test.classifier_name");


    if(task == "test")
    {
        // AdaboostLearner Learner(silent_mode,labeledData); //TODO actually uses "verbose", adjust!!!

        ModelIO modelReader;
        modelReader.read_model(classifierName);
        StrongClassifier classifier = modelReader.read();
        testTimeClassifier = &classifier;
        //classifier._learners[classifier._learners.size()-1]._cascadeThreshold = 0;
        //classifier.writeClassifier("0803cascade5000.firstIter.proto.bin");

        const TrainingData::point_t modelWindow = get_model_windowSize(options);
        //const TrainingData::rectangle_t objectWindow = get_object_window(options);

        //get Data

        std::vector<std::string> filenamesPositives, filenamesBackground;
        getImageFileNames(testSetPath, backgroundClassLabel, filenamesPositives, filenamesBackground);
        // above param originally called testSetPaths
        LabeledData labeledData(silent_mode, backgroundClassLabel);

        labeledData.createIntegralImages(filenamesPositives, filenamesBackground,
                                         modelWindow, test_offsetX, test_offsetY,
                                         options);

        labeledTestData = &labeledData;

    }

    else if(task == "train")
    {
        const double start_preparation_of_training = omp_get_wtime();        //std::vector<int> classifiersPerStage, maxNumSamplesPerImage;

        bool doBootstrap = true; //TODO: MAKE PARAMETER!!!
        // changed if-else to if-block with same outcome
        if(!doBootstrap)
        {
            classifiersPerStage.empty();
            classifiersPerStage.push_back(numIterations);
            maxNumSamplesPerImage.empty();
            maxNumSamplesPerImage.push_back(-1);
        }

        if (classifiersPerStage.size() != maxNumSamplesPerImage.size())
        {
            printf("classifiersperStage.size() == %zi, maxNumSamplesPerImage.size() == %zi\n",
                   classifiersPerStage.size(), maxNumSamplesPerImage.size());
            throw runtime_error("Size miss match between the vectors classifiersperStage and maxNumSamplesPerImage");
        }

        //      TrainingData::point_t testingDataOffset;
        modelWindowSize = get_model_windowSize(options);
        trainDataOffset = getTrainingDataOffset(options);

        const TrainingData::rectangle_t objectWindow = get_object_window(options);

        getImageFileNames(trainSetPath, backgroundClassLabel, filenamesPositives, filenamesBackground, filenamesHardNegatives, options);

        assert(filenamesPositives.empty() == false);

        //const int trainNumNegativeSamples previously retrieved here

        size_t maxNumExamples = filenamesPositives.size() + trainNumNegativeSamples + filenamesHardNegatives.size();

        //float posNegRatio = filenamesPositives.size()/(float) trainNumNegativeSamples;
        if(classifiersPerStage.size() > 1)
        { // thus doBootstrap == true
            maxNumExamples += numBootstrappingSamples*(classifiersPerStage.size() - 1);
        }

        //const std::string initialBootstrapFileName previously retrieved here
        if (!initialBootstrapFileName.empty())
        {
            maxNumExamples += numBootstrappingSamples;
        }

        // integral_channels_computer_p was previously declared here

        // we retrieve the number of channels --
        //const std::string channels_folder_path previously retrieved here

        //throw std::runtime_error("integral_channels_computer_p should be instanciated "
        //                         "in setup_training using IntegralChannelsComputerFactory::new_instance");

        //TrainingData::integral_channels_computer_shared_ptr_t
        //       integral_channels_computer_p(new doppia::IntegralChannelsFromFiles(channels_folder_path));
        // const int numChannels = getNumChannels(integral_channels_computer_p, filenamesPositives[0]);

        integral_channels_computer_p.reset(doppia::IntegralChannelsComputerFactory::new_instance(options));
        const int numChannels = integral_channels_computer_p->get_num_channels();

        printf("The selected integral channels computer will generate %i channels per image\n", numChannels);

        // computes all feature configurations available for training.
        // "size_t featuresPoolSize" & "std::string featurePoolType" previously retrieved here
        FeaturesSharedPointer featuresConfigurations(new Features());
        getFeaturesConfigurations(modelWindowSize, featuresPoolSize, numChannels, *featuresConfigurations, featurePoolType, options);
        featuresPoolSize = featuresConfigurations->size();
        std::cout << "Current size of the feature Pool: " << featuresPoolSize << std::endl;

        //writeFeaturesToFile("features.txt", *featuresConfigurations); return;
        //getFeatureConfigureationsFromFile(*featuresConfigurations); return;

        std::vector<bool> valid_features(featuresConfigurations->size(), false);
        fill (valid_features.begin(), valid_features.begin()+featuresPoolSize, true);

        trainingData.reset(new TrainingData(featuresConfigurations, valid_features, maxNumExamples,
                                            modelWindowSize, objectWindow, integral_channels_computer_p));
        trainingData->addPositiveSamples(filenamesPositives, modelWindowSize, trainDataOffset);

        //std::cout << (int)round(t.elapsed());

        { // normal channel features
            trainingData->addNegativeSamples(filenamesBackground, modelWindowSize, trainDataOffset, trainNumNegativeSamples);
            trainingData->addHardNegativeSamples(filenamesHardNegatives, modelWindowSize, trainDataOffset);
        }

        const bool dumpfeature_responses = false;
        if (dumpfeature_responses)
        {
            trainingData->dumpfeature_responses("feature_responses.txt");
            throw std::runtime_error("stop here for now");
        }
        if (!initialBootstrapFileName.empty())
        {
            const int maxFalsePositivesPerImage = 5;
            trainingData->addBootstrappingSamples(initialBootstrapFileName, filenamesBackground,
                                                  modelWindowSize, trainDataOffset,
                                                  numBootstrappingSamples, maxFalsePositivesPerImage,
                                                  options);
        }

        trainingData->setupBins(1000);

        const std::string outputModelPath = get_option_value<std::string>(options, "train.output_model_filename");

        BoostedLearner.reset(new AdaboostLearner(silent_mode, trainingData, typeAdaboost, numIterations, outputModelPath,
                                                 options));

        if (not testSetPath.empty())
        {
            printf("\nCollecting test data...\n");
            testingDataOffset = getTestingDataOffset(options);
            std::vector<std::string> filenamesPositives, filenamesBackground;
            getImageFileNames(testSetPath, backgroundClassLabel, filenamesPositives, filenamesBackground);
            size_t testingNumNegativeSamples = trainNumNegativeSamples;
            size_t maxTestingNumExamples = filenamesPositives.size() + testingNumNegativeSamples;;

            TrainingData::shared_ptr_t testingData(new TrainingData(featuresConfigurations, valid_features, maxTestingNumExamples,
                                                                    modelWindowSize, objectWindow, integral_channels_computer_p));
            testingData->addPositiveSamples(filenamesPositives, modelWindowSize, testingDataOffset);
            testingData->addNegativeSamples(filenamesBackground, modelWindowSize, testingDataOffset, testingNumNegativeSamples);

            BoostedLearner->set_test_data(testingData);
        }

        if (not validationSetPath.empty())
        {
            printf("\nCollecting validation data...\n");
            //validationset has the same offset as testset
            std::vector<std::string> filenamesPositives;
            getImageFileNames(validationSetPath, backgroundClassLabel, filenamesPositives, filenamesBackgroundValidation);
            size_t testingNumNegativeSamples = 2000;
            size_t maxValidationNumExamples = filenamesPositives.size() + 2000;

            if(classifiersPerStage.size() > 1)
            { // thus doBootstrap == true
                maxValidationNumExamples += 2000*(classifiersPerStage.size() - 1);
            }

            validationData.reset(new TrainingData(featuresConfigurations, valid_features, maxValidationNumExamples,
                                                  modelWindowSize, objectWindow, integral_channels_computer_p));
            validationData->addPositiveSamples(filenamesPositives, modelWindowSize, testingDataOffset);
            validationData->addNegativeSamples(filenamesBackgroundValidation, modelWindowSize, testingDataOffset, testingNumNegativeSamples);

            BoostedLearner->set_validation_data(validationData);
        }

        boost::posix_time::seconds delta_time(omp_get_wtime() - start_preparation_of_training);
        printf("Time elapsed while setting up data: %s\n",
               boost::posix_time::to_simple_string(delta_time).c_str());
    }

    return;
} // end of BoostedLearningApplication::setup_problem


AbstractGui* BoostedLearningApplication::create_gui(const program_options::variables_map &options)
{

    //const bool use_empty_gui = get_option_value<bool>(options, "gui.disabled");

    bool use_empty_gui = true;
    if(options.count("gui.disabled") > 0)
    {
        use_empty_gui = get_option_value<bool>(options, "gui.disabled");
    }

    AbstractGui *gui_p=NULL;
    if(use_empty_gui)
    {
        gui_p = new EmptyGui(options);
    }
    else
    {
        //gui_p = new ComputeWhiteningMatrixGui(*this, options);
        throw std::runtime_error("BoostedLearningGui not yet implemented");
    }

    return gui_p;
}


int BoostedLearningApplication::get_current_frame_number() const
{
    int current_frame_number = 0;
    if(directory_input_p)
    {
        current_frame_number = directory_input_p->get_current_frame_number();
    }

    return current_frame_number;
}


int detect(bool silent_mode)
{
    throw std::runtime_error("Detect task is not implemented. "
                             "You can use doppia objects_detection application instead");

    return 0;
}

std::string getSoftCascadeFileName(const program_options::variables_map &options, std::string outputModelFileName = std::string())
{
    if (outputModelFileName.empty()){
        using namespace boost::posix_time;
        const ptime current_time(second_clock::local_time());

        boost::filesystem::path outputModelPath = get_option_value<std::string>(options, "train.output_model_filename");
        outputModelFileName =
                (outputModelPath.parent_path() /
                 boost::str( boost::format("%i_%02i_%02i_%i_%s")
                             % current_time.date().year()
                             % current_time.date().month().as_number()
                             % current_time.date().day()
                             % current_time.time_of_day().total_seconds()
                     #if BOOST_VERSION <= 104400
                             % outputModelPath.filename()
                     #else
                             % outputModelPath.filename().string()
                     #endif
                             )).string();
    }
    const boost::filesystem::path
            outputModelPath(outputModelFileName),
        #if BOOST_VERSION <= 104400
            softCascadeFilePath =
            outputModelPath.parent_path() / (outputModelPath.stem() + ".softcascade" + outputModelPath.extension());
#else
            softCascadeFilePath =
            outputModelPath.parent_path() / (outputModelPath.stem().string() + ".softcascade" + outputModelPath.extension().string());
#endif


    return softCascadeFilePath.string();
}


void toSoftCascade(const bool silent_mode, const program_options::variables_map &options,
                   std::string inputModelFileName = std::string(), std::string softCascadeFileName= std::string())
{
    if(inputModelFileName.empty())
    {
        // if the input is empty, then we use the best guess
        inputModelFileName = get_option_value<std::string>(options, "train.output_model_filename");
    }

    if (softCascadeFileName.empty())
    {
        softCascadeFileName = getSoftCascadeFileName(options);
    }

    //read files for training
    const std::string
            trainSetPath = get_option_value<std::string>(options, "train.training_set");

    const int backgroundClassLabel = get_option_value<int>(options, "train.background_class_label");

    ModelIO modelReader;
    modelReader.read_model(inputModelFileName);


    const TrainingData::point_t
            modelWindowSize = get_model_windowSize(options),
            modelWindowSizeFromModel = modelReader.get_model_window_size(),
            trainDataOffset = getTrainingDataOffset(options);
    const TrainingData::rectangle_t
            objectWindow = get_object_window(options),
            objectWindowFromModel = modelReader.get_object_window();


    if(modelWindowSize != modelWindowSizeFromModel)
    {
        std::cerr<< " model window size from model: : " << modelWindowSizeFromModel.x() << " " << modelWindowSizeFromModel.y() << std::endl;
        throw std::invalid_argument("The modelWindowSize specified in the .ini file does not match the model content");
    }

    if(objectWindow != objectWindowFromModel)
    {
        std::cerr<< " object window size from model: " << objectWindowFromModel.min_corner().x() << " " << objectWindowFromModel.min_corner().y() <<
                    objectWindowFromModel.max_corner().x() << " " << objectWindowFromModel.max_corner().y()<< std::endl;
        throw std::invalid_argument("The modelWindowSize specified in the .ini file does not match the model content");
    }

    printf("Starting computing the soft cascade.\n");

    std::vector<std::string> filenamesPositives, filenamesBackground;
    getImageFileNames(trainSetPath, backgroundClassLabel, filenamesPositives, filenamesBackground);

    LabeledData::shared_ptr labeledTrainData(new LabeledData(silent_mode, backgroundClassLabel));
    labeledTrainData->createIntegralImages(filenamesPositives, filenamesBackground,
                                           modelWindowSize, trainDataOffset.x(), trainDataOffset.y(),
                                           options);


    // computed all feature configurations available for training.
    AdaboostLearner::to_soft_cascade_dbp(labeledTrainData,
                                         inputModelFileName, softCascadeFileName,
                                         modelWindowSize, objectWindow);

    printf("Finished computing the softcascade.\n"
           "Final soft cascade model saved at %s\n",
           softCascadeFileName.c_str());

    return;
}  // end of toSoftCascade


int printModel(const program_options::variables_map &options)
{
    std::string outputModelFileName = get_option_value<std::string>(options, "train.output_model_filename");
    ModelIO modelReader(10);
    modelReader.read_model(outputModelFileName);
    StrongClassifier learner = modelReader.read();

    modelReader.print();
    return 0;

}

void BoostedLearningApplication::test(const bool silent_mode)
{

    boost::timer timer;

    // do classification
    int tp, fp, fn, tn;
    (*testTimeClassifier).classify((*labeledTestData), tp, fp, fn, tn);

    const float time_in_seconds = timer.elapsed();

    std::cout << "Time required for execution: " << time_in_seconds << " seconds." << "\n\n";
    std::cout << "FrameRate: " << (*labeledTestData).get_num_examples() / time_in_seconds << std::endl;


    std::cout << "Classification Results (TestData): " << std::endl;
    std::cout << "Detection Rate: " << float(tp + tn) / (tp + tn + fp + fn) * 100 << " %" <<  std::endl;
    std::cout << "Error Rate: " << float(fp + fn) / (tp + tn + fp + fn) * 100 << " %" <<  std::endl;
    std::cout << "Error Positives: " <<  float(fn) / (tp + fn) * 100 << " %" <<  std::endl;
    std::cout << "Error Negatives: " <<  float(fp) / (tn + fp) * 100 << " %" <<  std::endl;
    std::cout << "\n";

    return;
}

void BoostedLearningApplication::train(const bool silent_mode, const bool doBootstrap = true)
{
    boost::posix_time::seconds delta_time(0);
    const double start_total_training_time = omp_get_wtime();

    const std::string baseOuputModelFilename = BoostedLearner->get_output_model_filename();
    std::cout << BoostedLearner->get_output_model_filename() << " - model name" << std::endl;
    bool useAdditionalNegatives = false;
    // const std::string extendedTrainset previously retrieved here;
    if (extendedTrainSetPath != ""){
        useAdditionalNegatives = true;
    }

    for (size_t k = 0; k < classifiersPerStage.size(); ++k)
    {


        // if not first round, bootstrap new negatives
        if (k != 0)
        {
            const double start_bootstrapping_time = omp_get_wtime();
            const std::string bootstrapFile =
                    boost::str(boost::format("%s.bootstrap%i") % baseOuputModelFilename % (k - 1));
            std::cout << bootstrapFile << " - bootstrap file" << std::endl;

            // use more samples form pascal VOC
            if (useAdditionalNegatives){
                useAdditionalNegatives = false;
                getImageFileNames(extendedTrainSetPath, backgroundClassLabel, filenamesPositives, filenamesBackground);
            }

            // sample new (hard) negatives using bootstrapping
            trainingData->addBootstrappingSamples(bootstrapFile, filenamesBackground,
                                                  modelWindowSize, trainDataOffset,
                                                  numBootstrappingSamples, maxNumSamplesPerImage[k],
                                                  options);
            trainingData->setupBins(1000);

            if (not validationSetPath.empty())
            {
                validationData->addBootstrappingSamples(bootstrapFile, filenamesBackgroundValidation,
                                                        modelWindowSize, testingDataOffset,
                                                        2000, maxNumSamplesPerImage[k],
                                                        options);
            }

            delta_time = boost::posix_time::seconds(omp_get_wtime() - start_bootstrapping_time);
            printf("Time elapsed in seconds: %d\n", delta_time.total_seconds());
            printf("Time elapsed while mining hard negatives for training round %zu: %s\n",
                   k, boost::posix_time::to_simple_string(delta_time).c_str());

        }

        const double start_training_time = omp_get_wtime();

        BoostedLearner->set_num_training_rounds(classifiersPerStage[k]);
        BoostedLearner->set_output_model_filename(boost::str(boost::format("%s.bootstrap%i") % baseOuputModelFilename % (k)));
        const int decisionTreeDepth = get_option_value<int>(options, "train.decision_tree_depth");
        const std::string datasetName = get_option_value<std::string>(options, "train.training_set_name");
        // const bool useDropout = get_option_value<bool>(options, "train.use_dropout");

        //const bool useStumpSets  previously retrieved here
        if(useStumpSets)
        {
            BoostedLearner->train(decisionTreeDepth, datasetName, STUMP_SET);
        }
        else
        {
            BoostedLearner->train(decisionTreeDepth, datasetName, DECISION_TREE);
        }

        delta_time = boost::posix_time::seconds(omp_get_wtime() - start_training_time);
        printf("Time elapsed in seconds: %d\n", delta_time.total_seconds());
        printf("Time elapsed while producing strong classifier for training round %zu: %s\n",
               k, boost::posix_time::to_simple_string(delta_time).c_str());
    } // end of "for each training round"

    boost::filesystem::copy_file(BoostedLearner->get_output_model_filename(), baseOuputModelFilename);

    printf("Finished the %zi bootstrapping stages. Model was trained over %zi samples (%zi positives, %zi negatives).\n"
           "Final model saved at %s\n",
           classifiersPerStage.size(),
           trainingData->get_num_examples(),
           trainingData->get_num_positive_examples(), trainingData->get_num_negative_examples(),
           //BoostedLearner->getOuputModelFileName().c_str()
           baseOuputModelFilename.c_str());


    const bool computeSoftcascade = false; // FIXME should be an option
    if(computeSoftcascade)
    {
        toSoftCascade(silent_mode, options, baseOuputModelFilename, getSoftCascadeFileName(options, baseOuputModelFilename));
    }

    delta_time = boost::posix_time::seconds(omp_get_wtime() - start_total_training_time);
    printf("Time elapsed in seconds: %d\n", delta_time.total_seconds());
    printf("Time elapsed for all training and boostrapping rounds: %s\n",
           boost::posix_time::to_simple_string(delta_time).c_str());

    return;
} // end of bootstrapTrain


void BoostedLearningApplication::main_loop()
{
    // get task
    if(silent_mode)
    {
        printf("The application is running in silent mode. "
               "No information will be printed until training is complete.\n");
    }

    std::cout << "Task: " << task << std::endl;

    if (task == "train")
    {
        train(silent_mode);
    }
    else if (task == "test")
    {
        test(silent_mode);
    }
    else
    {
        throw std::invalid_argument("unknown task given");
    }

    printf("End of game. Have a nice day !\n");
    return;
} // end of void BoostedLearningApplication::main_loop



}
