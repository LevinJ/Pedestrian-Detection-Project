#include "TrainingData.hpp"

#include "applications/bootstrapping_lib/bootstrapping_lib.hpp"

//#include "objects_detection/integral_channels/IntegralChannelsFromFiles.hpp"
#include "objects_detection/integral_channels/IntegralChannelsForPedestrians.hpp"

#include "video_input/ImagesFromDirectory.hpp" // for the open_image helper method
#include "integral_channels_helpers.hpp"

#include <boost/format.hpp>
#include "helpers/progress_display_with_eta.hpp"

#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/nvp.hpp>


#include <cstdio>

namespace boosted_learning {

using namespace boost;

TrainingData::TrainingData(
        ConstFeaturesSharedPointer featuresConfigurations,
        const std::vector<bool> &valid_features,
        const size_t maxNumExamples, const point_t modelWindow, const rectangle_t objectWindow,
        integral_channels_computer_shared_ptr_t integral_channels_computer_p)
    :
      _backgroundClassLabel(-1), // FIXME hardcoded value
      _featuresConfigurations(featuresConfigurations),
      _validFeatures(valid_features),
      _modelWindow(modelWindow),
      _objectWindow(objectWindow),
      _numPositivesExamples(0),
      _numNegativesExamples(0),
      _integralChannelsComputer(integral_channels_computer_p)

{
    // we allocated the full data memory at the begining
    _feature_responses.resize(boost::extents[_featuresConfigurations->size()][maxNumExamples]);
    _featureBinResponses.resize(boost::extents[_featuresConfigurations->size()][maxNumExamples]);
    _metaData.resize(maxNumExamples);
    printf("Allocated features responses for %zi features and a maximum of %zi samples\n",
           _featuresConfigurations->size(), maxNumExamples);

    if(_integralChannelsComputer.get() == NULL)
    {
       throw std::runtime_error("TrainingData should receive as input an AbstractIntegralChannelsComputer instance");
    }

    return;
}

TrainingData::~TrainingData()
{
    // nothing to do here
    return;
}


size_t TrainingData::get_feature_pool_size() const
{
    // size_t ret = 0;
    // for (size_t k =0; k< _validFeatures.size(); ++k)
    // {
    //     if (_validFeatures[k] == true)
    //         ret++;
    // }
    // return ret;

    return  _feature_responses.shape()[0];
}

size_t TrainingData::getMaxNumExamples() const
{
    return _feature_responses.shape()[1];
}

size_t TrainingData::get_num_examples() const
{
    return _numPositivesExamples + _numNegativesExamples;
}

size_t TrainingData::get_num_positive_examples()const
{
    return _numPositivesExamples;
}

size_t TrainingData::get_num_negative_examples() const
{
    return _numNegativesExamples;
}

const TrainingData::rectangle_t & TrainingData::get_object_window() const
{
    return _objectWindow;
}

const TrainingData::point_t & TrainingData::get_model_window() const
{
    return _modelWindow;
}

TrainingData::objectClassLabel_t TrainingData::get_class_label_for_sample(const size_t Index) const
{
    return _metaData[Index].imageClass;
}

const string &TrainingData::get_file_name(const size_t Index) const
{
    return _metaData[Index].filename;
}

const ConstFeaturesSharedPointer TrainingData::getFeaturesConfigurations() const
{
    return _featuresConfigurations;
}

const FeaturesResponses &TrainingData::get_feature_responses() const
{
    return _feature_responses;
}

const FeaturesBinResponses &TrainingData::get_feature_bin_responses() const
{
    return _featureBinResponses;
}


const Feature &TrainingData::get_feature(const size_t featureIndex) const
{
    return (*_featuresConfigurations)[featureIndex];
}

bool TrainingData::get_feature_validity(const size_t featureIndex) const
{
    return _validFeatures[featureIndex];
}


void TrainingData::setDatum(
        const size_t datumIndex,
        const meta_datum_t &metaDatum,
        const bootstrapping::integral_channels_t &integralImage)
{
    assert(datumIndex < getMaxNumExamples());

    for (size_t featuresIndex = 0; featuresIndex < _featuresConfigurations->size(); featuresIndex+=1)
    {
        const int feature_response = (*_featuresConfigurations)[featuresIndex].getResponse(integralImage);
        //printf("featuresIndex == %zi; datumIndex == %zi; feature_response == %i\n",
        //      featuresIndex, datumIndex, feature_response);
        //fflush(stdout);
        _feature_responses[featuresIndex][datumIndex] = feature_response;
    }

    _metaData[datumIndex] = metaDatum;

    if(metaDatum.imageClass == _backgroundClassLabel)
    {
        _numNegativesExamples += 1;
    }
    else
    {
        _numPositivesExamples += 1;
    }

    const bool save_integral_images = false;
    if(save_integral_images)
    {
        static int false_positive_counter = 0;
        const int max_images_to_save = 10;
        //const size_t startDatumIndex = 5000 + false_positives;
        const size_t startDatumIndex = 0;
        if((datumIndex > startDatumIndex) and (false_positive_counter < max_images_to_save))
        {

            boost::format filename_pattern("false_positive_%i.png");
            const boost::filesystem::path
                    storage_path = "/tmp/",
                    file_path = storage_path / boost::str( filename_pattern % false_positive_counter);
            doppia::save_integral_channels_to_file(integralImage, file_path.string());

            false_positive_counter += 1;

            printf("Saved %i false positives integral channels images inside %s\n",
                   false_positive_counter, storage_path.string().c_str());
            //throw std::runtime_error("Stopping everything so you can look at the false positive integral channels images");
        }
    } // end of "if shoudl save integral images"

    return;
}


void TrainingData::addPositiveSamples(const std::vector<std::string> &filenamesPositives,
                                      const point_t &modelWindowSize, const point_t &dataOffset)
{

    int feature_extraction_time = 0, image_loading_time = 0, tmp_time;

    const size_t
            initialNumberOfTrainingSamples = get_num_examples(),
            finalNumberOfTrainingSamples = initialNumberOfTrainingSamples + filenamesPositives.size();
    if(finalNumberOfTrainingSamples > getMaxNumExamples())
    {
        throw std::runtime_error("TrainingData::addPositiveSamples is trying to add more data than initially specified");
    }


    printf("\nCollecting %zi positive samples\n", filenamesPositives.size());
    doppia::progress_display_with_eta progress_indicator(filenamesPositives.size());


    meta_datum_t  metaDatum;
    integral_channels_t sampleIntegralChannels;

    // integralChannelsComputer is already multithreaded, so no benefit on paralelizing this for loop
    for (size_t filenameIndex = 0; filenameIndex < filenamesPositives.size(); filenameIndex +=1)
    {
//tmp_time = (int)round(omp_get_wtime());
        //std::cout << filenamesPositives[filenameIndex].c_str() << std::endl;
        gil::rgb8_image_t image;
        gil::rgb8c_view_t image_view = doppia::open_image(filenamesPositives[filenameIndex].c_str(), image);
//image_loading_time += (int)round(omp_get_wtime()) - tmp_time;
        const boost::filesystem::path file_path = filenamesPositives[filenameIndex];
#if BOOST_VERSION <= 104400
        const std::string filename = file_path.filename();
#else
        const std::string filename = file_path.filename().string();
#endif
tmp_time = (int)round(omp_get_wtime());
        _integralChannelsComputer->set_image(image_view, filename);
image_loading_time += (int)round(omp_get_wtime()) - tmp_time;
tmp_time = (int)round(omp_get_wtime());
        _integralChannelsComputer->compute();
feature_extraction_time += (int)round(omp_get_wtime()) - tmp_time;
//tmp_time = (int)round(omp_get_wtime());
        get_integral_channels(_integralChannelsComputer->get_integral_channels(),
                              modelWindowSize, dataOffset, doppia::IntegralChannelsForPedestrians::get_shrinking_factor(),
                              sampleIntegralChannels);
//image_loading_time += (int)round(omp_get_wtime()) - tmp_time;
        metaDatum.filename = filenamesPositives[filenameIndex];
        metaDatum.imageClass = 1;//classes[k];
        metaDatum.x = dataOffset.x();
        metaDatum.y = dataOffset.y();

        setDatum(initialNumberOfTrainingSamples + filenameIndex,
                 metaDatum, sampleIntegralChannels);

        ++progress_indicator;
    } // end of "for each filename"

    printf("Time elapsed while loading positive images: %02d:%02d:%02d\n",
           image_loading_time/3600, (image_loading_time%3600)/60, image_loading_time%60);
    printf("Time elapsed while extracting features from positive images: %02d:%02d:%02d\n",
           feature_extraction_time/3600, (feature_extraction_time%3600)/60, feature_extraction_time%60);


    return;
}


void TrainingData::addHardNegativeSamples(const std::vector<std::string> &filenamesHardNegatives,
                                          const point_t &modelWindowSize, const point_t &dataOffset)
{
    int feature_extraction_time = 0, image_loading_time = 0, tmp_time;

    const size_t
            initialNumberOfTrainingSamples = get_num_examples(),
            finalNumberOfTrainingSamples = initialNumberOfTrainingSamples + filenamesHardNegatives.size();
    if(finalNumberOfTrainingSamples > getMaxNumExamples())
    {
        throw std::runtime_error("TrainingData::addHardNegativeSamples "
                                 "is trying to add more data than initially specified");
    }


    printf("\nCollecting %zi hard negative samples\n", filenamesHardNegatives.size());
    doppia::progress_display_with_eta progress_indicator(filenamesHardNegatives.size());


    meta_datum_t  metaDatum;
    integral_channels_t sampleIntegralChannels;

    // integralChannelsComputer is already multithreaded, so no benefit on paralelizing this for loop
    for (size_t filenameIndex = 0; filenameIndex < filenamesHardNegatives.size(); filenameIndex +=1)
    {
        tmp_time = (int)round(omp_get_wtime());
        gil::rgb8_image_t image;
        gil::rgb8c_view_t image_view = doppia::open_image(filenamesHardNegatives[filenameIndex].c_str(), image);

        const boost::filesystem::path file_path = filenamesHardNegatives[filenameIndex];
#if BOOST_VERSION <= 104400
        const std::string filename = file_path.filename();
#else
        const std::string filename = file_path.filename().string();
#endif
        _integralChannelsComputer->set_image(image_view, filename);
        image_loading_time += (int)round(omp_get_wtime()) - tmp_time;
        tmp_time = (int)round(omp_get_wtime());
        _integralChannelsComputer->compute();
        feature_extraction_time += (int)round(omp_get_wtime()) - tmp_time;

        get_integral_channels(_integralChannelsComputer->get_integral_channels(),
                              modelWindowSize, dataOffset, doppia::IntegralChannelsForPedestrians::get_shrinking_factor(),
                              sampleIntegralChannels);

        metaDatum.filename = filenamesHardNegatives[filenameIndex];
        metaDatum.imageClass = _backgroundClassLabel;
        metaDatum.x = dataOffset.x();
        metaDatum.y = dataOffset.y();

        setDatum(initialNumberOfTrainingSamples + filenameIndex, metaDatum, sampleIntegralChannels);

        ++progress_indicator;
    } // end of "for each filename"
    printf("Time elapsed while loading images for hard negatives extraction: %02d:%02d:%02d\n",
           image_loading_time/3600, (image_loading_time%3600)/60, image_loading_time%60);
    printf("Time elapsed while extracting features from hard negatives: %02d:%02d:%02d\n",
           feature_extraction_time/3600, (feature_extraction_time%3600)/60, feature_extraction_time%60);

    return;
}


void TrainingData::addNegativeSamples(const std::vector<std::string> &filenamesBackground,
                                      const point_t &modelWindowSize, const point_t &dataOffset,
                                      const size_t numNegativeSamplesToAdd)
{

    int feature_extraction_time = 0, image_loading_time = 0, tmp_time;

    const size_t
            initialNumberOfTrainingSamples = get_num_examples(),
            finalNumberOfTrainingSamples = initialNumberOfTrainingSamples + numNegativeSamplesToAdd;
    if(finalNumberOfTrainingSamples > getMaxNumExamples())
    {
        throw std::runtime_error("TrainingData::addNegativeSamples is trying to add more data than initially specified");
    }

    printf("\nCollecting %zi random negative samples\n", numNegativeSamplesToAdd);
    doppia::progress_display_with_eta progress_indicator(numNegativeSamplesToAdd);

    meta_datum_t  metaDatum;
    integral_channels_t sampleIntegralChannels;

#if defined(DEBUG)
    srand(1);
#else
    srand(time(NULL));
#endif
    srand(1);

    const int samplesPerImage = std::max<int>(1, numNegativeSamplesToAdd / filenamesBackground.size());

    // FIXME no idea what the +1 does
    const int
            minWidth = (modelWindowSize.x()+1 + 2*dataOffset.x()),
            minHeight = (modelWindowSize.y()+1 + 2*dataOffset.y());

    const float maxSkippedFraction = 0.25;

    size_t numNegativesSamplesAdded = 0, numSkippedImages = 0, filenameIndex = 0;

    // integralChannelsComputer is already multithreaded, so no benefit on paralelizing this for loop
    while (numNegativesSamplesAdded < numNegativeSamplesToAdd)
    {
        if (filenameIndex >= filenamesBackground.size())
        {
            // force to loop until we have reached the desired number of samples
            filenameIndex = 0;
        }
        const string &background_image_path = filenamesBackground[filenameIndex];
        filenameIndex +=1;

        gil::rgb8c_view_t imageView;
        gil::rgb8_image_t image;
//tmp_time = (int)round(omp_get_wtime());
        imageView = doppia::open_image(background_image_path.c_str(), image);
//image_loading_time += (int)round(omp_get_wtime()) - tmp_time;

        if ((imageView.width() < minWidth) or (imageView.height() < minHeight))
        {
            // if input image is too small, we skip it
            //printf("Skipping negative sample %s, because it is too small\n", filename.c_str());
            numSkippedImages += 1;

            const float skippedFraction = static_cast<float>(numSkippedImages) / filenamesBackground.size();
            if (skippedFraction > maxSkippedFraction)
            {
                printf("Skipped %zi images (out of %zi, %.3f%%) because they where too small (or too big to process)\n",
                       numSkippedImages, filenamesBackground.size(), skippedFraction*100);

                throw std::runtime_error("Too many negatives images where skipped. Dataset needs to be fixed");
            }
            continue;
        }

        const int
                maxRandomX = (imageView.width() - modelWindowSize.x()+1 - 2*dataOffset.x()),
                maxRandomY = (imageView.height() - modelWindowSize.y()+1 - 2*dataOffset.y());

        try
        {
            // FIXME harcoded values
            const size_t
                    expected_channels_size = imageView.size()*10,
                    max_texture_size = 134217728; // 2**27 for CUDA capability 2.x
            if(expected_channels_size > max_texture_size)
            {
                throw std::invalid_argument("The image is monstruously big!");
            }

            const boost::filesystem::path file_path = background_image_path;
#if BOOST_VERSION <= 104400
            const std::string filename = file_path.filename();
#else
            const std::string filename = file_path.filename().string();
#endif
tmp_time = (int)round(omp_get_wtime());
            _integralChannelsComputer->set_image(imageView, filename);
image_loading_time += (int)round(omp_get_wtime()) - tmp_time;
tmp_time = (int)round(omp_get_wtime());
            _integralChannelsComputer->compute();
feature_extraction_time += (int)round(omp_get_wtime()) - tmp_time;

        }
        catch(std::exception &e)
        {
            printf("Computing integral channels of image %s \033[1;31mfailed\033[0m (size %zix%zi). Skipping it. Error was:\n%s\n",
                   background_image_path.c_str(),
                   imageView.width(), imageView.height(),
                   e.what());
            numSkippedImages += 1;
            continue; // we skip this image
        }
        catch(...)
        {
            printf("Computing integral channels of %s \033[1;31mfailed\033[0m (size %zix%zi). Skipping it. Received unknown error.\n",
                   background_image_path.c_str(),
                   imageView.width(), imageView.height());
            numSkippedImages += 1;
            continue; // we skip this image
        }

        metaDatum.filename = background_image_path;
        metaDatum.imageClass = _backgroundClassLabel;

        size_t numSamplesForImage = std::min<size_t>(samplesPerImage,
                                                     (numNegativeSamplesToAdd - numNegativesSamplesAdded));
        numSamplesForImage = 1;
        for (size_t randomSampleIndex = 0; randomSampleIndex < numSamplesForImage; randomSampleIndex += 1)
        {
            //const point_t::coordinate_t
            size_t
                    x = dataOffset.x() + rand() % maxRandomX,
                    y = dataOffset.y() + rand() % maxRandomY;
            //printf("random x,y == %i, %i\n", x,y);
            const point_t randomOffset(x,y);
            metaDatum.x = randomOffset.x(); metaDatum.y = randomOffset.y();
//tmp_time = (int)round(omp_get_wtime());
            get_integral_channels(_integralChannelsComputer->get_integral_channels(),
                                  modelWindowSize, randomOffset, doppia::IntegralChannelsForPedestrians::get_shrinking_factor(),
                                  sampleIntegralChannels);
//image_loading_time += (int)round(omp_get_wtime()) - tmp_time;
            setDatum(initialNumberOfTrainingSamples + numNegativesSamplesAdded,
                     metaDatum, sampleIntegralChannels);

            numNegativesSamplesAdded += 1;
            ++progress_indicator;
        }

    } // end of "for each background image"



    if (numSkippedImages > 0)
    {
        const float skippedFraction = static_cast<float>(numSkippedImages) / filenamesBackground.size();
        printf("Skipped %zi images (out of %zi, %.3f%%) because they where too small (or too big to process)\n",
               numSkippedImages, filenamesBackground.size(), skippedFraction*100);
    }
    printf("Time elapsed while loading negative images: %02d:%02d:%02d\n",
           image_loading_time/3600, (image_loading_time%3600)/60, image_loading_time%60);
    printf("Time elapsed while extracting features from negative images: %02d:%02d:%02d\n",
           feature_extraction_time/3600, (feature_extraction_time%3600)/60, feature_extraction_time%60);

    return;
}

// ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

class AppendDatumFunctor
{
public:
    AppendDatumFunctor(TrainingData &trainingData);
    ~AppendDatumFunctor();

    void operator()(const TrainingData::meta_datum_t &metaDatum,
                    const TrainingData::integral_channels_t &integralImage);

protected:

    TrainingData &trainingData;
    size_t datumIndex;
};


AppendDatumFunctor::AppendDatumFunctor(TrainingData &trainingData_)
    : trainingData(trainingData_),
      datumIndex(trainingData_.get_num_examples())
{
    // nothing to do here
    return;
}

AppendDatumFunctor::~AppendDatumFunctor()
{
    // nothing to do here
    return;
}

void AppendDatumFunctor::operator()(const TrainingData::meta_datum_t &metaDatum,
                                    const TrainingData::integral_channels_t &integralImage)
{
    trainingData.setDatum(datumIndex, metaDatum, integralImage);
    datumIndex += 1;
    return;
}


// ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

void TrainingData::addBootstrappingSamples(
        const std::string classifierPath,
        const std::vector<std::string> &filenamesBackground,
        const point_t &modelWindowSize, const point_t &dataOffset,
        const size_t numNegativeSamplesToAdd, const int maxFalsePositivesPerImage,
        const boost::program_options::variables_map &options)
{

    const size_t
            initialNumberOfTrainingSamples = get_num_examples(),
            finalNumberOfTrainingSamples = initialNumberOfTrainingSamples + numNegativeSamplesToAdd;
    if (finalNumberOfTrainingSamples > getMaxNumExamples())
    {
        printf("initialNumberOfTrainingSamples == %zi\n", initialNumberOfTrainingSamples);
        printf("numNegativeSamplesToAdd == %zi\n", numNegativeSamplesToAdd);
        printf("finalNumberOfTrainingSamples %zi > getMaxNumExamples() %zi \n",
               finalNumberOfTrainingSamples, getMaxNumExamples());
        throw std::runtime_error("TrainingData::addBootstrappingSamples is trying to add more data than initially specified");
    }

    printf("Searching for hard %zi negatives given the current model (%s), please wait...\n",
           numNegativeSamplesToAdd, classifierPath.c_str());
    //doppia::progress_display_with_eta progress_indicator(numNegativeSamplesToAdd);

    const int
            numScales = get_option_value<int>(options, "bootstrap_train.num_scales"),
            numRatios = get_option_value<int>(options, "bootstrap_train.num_ratios");

    const float
            minScale = get_option_value<float>(options, "bootstrap_train.min_scale"),
            maxScale = get_option_value<float>(options, "bootstrap_train.max_scale"),
            minRatio = get_option_value<float>(options, "bootstrap_train.min_ratio"),
            maxRatio = get_option_value<float>(options, "bootstrap_train.max_ratio");

    const bool use_less_memory = get_option_value<bool>(options, "bootstrap_train.frugal_memory_usage");

    const size_t initialIntegralImagesSize = get_num_examples();
    bootstrapping::append_result_functor_t the_functor = AppendDatumFunctor(*this);
    bootstrapping::bootstrap(boost::filesystem::path(classifierPath), filenamesBackground,
                             numNegativeSamplesToAdd, maxFalsePositivesPerImage,
                             minScale, maxScale, numScales,
                             minRatio, maxRatio, numRatios,
                             use_less_memory,
                             the_functor,
options);

    const size_t numFoundFalsePositives = get_num_examples() - initialIntegralImagesSize;
    if (numFoundFalsePositives < numNegativeSamplesToAdd)
    {
        const size_t numRandomNegativesToAdd = numNegativeSamplesToAdd - numFoundFalsePositives;
        addNegativeSamples(filenamesBackground, modelWindowSize, dataOffset, numRandomNegativesToAdd);
    }

    return;
}


bintype TrainingData::get_bin_size() const
{
    return _num_bins;
}


void TrainingData::setupBins(int num_bins)
{
    this->_num_bins = num_bins;
    for (size_t featureIndex = 0; featureIndex < get_feature_pool_size(); ++featureIndex)
    {
        if (_validFeatures[featureIndex] == false)
            continue;
        int minv = std::numeric_limits<int>::max();
        int maxv = -std::numeric_limits<int>::max();

        for (size_t exampleIndex = 0; exampleIndex < get_num_examples(); ++exampleIndex)
        {
            const int feature_response = _feature_responses[featureIndex][exampleIndex];
            minv = std::min(feature_response, minv);
            maxv = std::max(feature_response, maxv);
        } // end of "for each example"




        const weights_t::value_type bin_scaling = num_bins / static_cast<weights_t::value_type>(maxv - minv);

        for (size_t exampleIndex = 0; exampleIndex < get_num_examples(); ++exampleIndex)
        {
            const int feature_response = _feature_responses[featureIndex][exampleIndex];
            _featureBinResponses[featureIndex][exampleIndex]=static_cast<bintype>( bin_scaling * (feature_response - minv));


        } // end of "for each example"




    } // end of "for each feature"
}


void TrainingData::dumpfeature_responses(const string &filename) const
{

    // save the matrix to disk --
    {

        std::ofstream output_stream(filename.c_str());
        boost::archive::text_oarchive archive(output_stream);
        std::cout << "Created recording file " << filename << std::endl;

        // write the data
        //archive & _feature_responses;

        archive << boost::serialization::make_nvp("shape",
                                                  boost::serialization::make_array(_feature_responses.shape(), _feature_responses.dimensionality));
        archive << boost::serialization::make_nvp("data",
                                                  boost::serialization::make_array(_feature_responses.data(), _feature_responses.num_elements()));
    }

    return;


}




} // namespace boosted_learning
