#ifndef BOOSTEDLEARNINGAPPLICATION_HPP
#define BOOSTEDLEARNINGAPPLICATION_HPP

#include "applications/BaseApplication.hpp"
#include "boosted_learning/LabeledData.hpp"
#include "boosted_learning/AdaboostLearner.hpp"
#include "boosted_learning/TrainingData.hpp"

#include <boost/filesystem/path.hpp>
#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>

namespace doppia {

using namespace boosted_learning;
// forward declarations
// if needed: class <ClassName>
class ImagesFromDirectory;
class AbstractIntegralChannelsComputer;
class IntegralChannelsComputer;

class BoostedLearningApplication : public doppia::BaseApplication
{

    // processing modules
    boost::scoped_ptr<ImagesFromDirectory> directory_input_p;
    //boost::shared_ptr<IntegralChannelsForPedestrians> integral_channels_computer_p;
    boost::shared_ptr<AbstractIntegralChannelsComputer> integral_channels_computer_p;

public:

    static boost::program_options::options_description get_options_description();

    std::string get_application_title();

    BoostedLearningApplication();
    virtual ~BoostedLearningApplication();

protected:

    void get_all_options_descriptions(boost::program_options::options_description &desc);

    void setup_logging(std::ofstream &log_file, const boost::program_options::variables_map &options);
    void setup_problem(const boost::program_options::variables_map &options);

    /// @returns a newly created gui object (can be NULL)
    AbstractGui* create_gui(const boost::program_options::variables_map &options);

    int get_current_frame_number() const;

    void main_loop();

    void train(const bool silent_mode, const bool doBootstrap);
    void test(const bool silent_mode);

protected:

    bool silent_mode;
    std::string task;
    //size_t num_files_to_process;
    //boost::filesystem::path output_path;

    //training parameters
    std::string trainSetPath, testSetPath, validationSetPath, extendedTrainSetPath;
    int numIterations, trainNumNegativeSamples;
    std::string initialBootstrapFileName;
    std::string channels_folder_path;
    size_t featuresPoolSize;
    std::string featurePoolType;
    std::string typeAdaboost;
    bool useStumpSets;

    //bootstrapping parameters
    int numBootstrappingSamples;
    std::vector<int> classifiersPerStage, maxNumSamplesPerImage;

    //testing parameters
    int test_offsetX, test_offsetY;
    std::string classifierName;

    int backgroundClassLabel;

    // required for test()
    StrongClassifier *testTimeClassifier;
    LabeledData *labeledTestData;
    // required for train()
    boost::shared_ptr<AdaboostLearner> BoostedLearner;
    TrainingData::shared_ptr_t trainingData, validationData;
    std::vector<std::string> filenamesPositives, filenamesBackgroundValidation, filenamesBackground, filenamesHardNegatives;
    TrainingData::point_t testingDataOffset, trainDataOffset, modelWindowSize;


};

} // end of namespace doppia

#endif // BOOSTEDLEARNINGAPPLICATION_HPP
