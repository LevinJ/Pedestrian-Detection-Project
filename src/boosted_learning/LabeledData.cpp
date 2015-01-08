#include "LabeledData.hpp"

#include "WeakLearner.hpp"
#include "ModelIO.hpp"
#include "StrongClassifier.hpp"
#include "applications/bootstrapping_lib/IntegralChannelsComputer.hpp"

#include "integral_channels_helpers.hpp"

#include "video_input/ImagesFromDirectory.hpp" // for the open_image helper method

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <boost/progress.hpp>
#include <boost/foreach.hpp>

#include <boost/gil/image_view.hpp>
#include <boost/gil/image.hpp>
#include <boost/gil/typedefs.hpp>
#include <boost/gil/extension/numeric/sampler.hpp>
#include <boost/gil/extension/numeric/resample.hpp>

#include <algorithm> // for std::sort
#include <ctime>

#include <iostream>
#include <fstream>
#include <sstream>

#include <omp.h>

namespace doppia {

// forward declarations
class GpuIntegralChannelsForPedestrians;
class IntegralChannelsForPedestrians;

}

namespace boosted_learning {

namespace gil = boost::gil;


LabeledData::LabeledData(bool silent_mode, int backgroundClassLabel):
    _silent_mode(silent_mode), _backgroundClassLabel(backgroundClassLabel), _numPosExamples(0), _numNegExamples(0)
{
    // nothing to do here
    return;
}

LabeledData::~LabeledData()
{
    //_intImages.clear();
    _integralImages.clear();
    _metaData.clear();
    return;
}

const doppia::IntegralChannelsForPedestrians::integral_channels_t &LabeledData::get_integral_image(const size_t Index) const
{
    return _integralImages[Index];
}

int LabeledData::get_class_label_for_sample(const size_t Index) const
{
    return _metaData[Index].imageClass;
}

const string &LabeledData::get_file_name(const size_t Index) const
{
    return _metaData[Index].filename;
}

int LabeledData::getX(const size_t Index) const
{
    return _metaData[Index].x;
}

int LabeledData::getY(const size_t Index) const
{
    return _metaData[Index].y;
}

const LabeledData::meta_datum_t &LabeledData::getMetaDatum(const size_t Index) const
{
    return _metaData[Index];
}

const LabeledData::meta_data_t &LabeledData::getMetaData() const
{
    return _metaData;
}

size_t LabeledData::get_num_examples() const
{
    return _integralImages.size();
}

size_t LabeledData::get_num_pos_examples()const
{
    return _numPosExamples;    //positive examples.
}

size_t LabeledData::getNumNegExamples() const
{
    return _numNegExamples;    //negative examples.
}

/// erases the currently stored integralImages and metaData
void LabeledData::clear()
{
    _numPosExamples = 0;
    _numNegExamples = 0;

    // _integralImages.clear() does not ensure to de-allocate memory
    // swap will force to free the memory
    // best known way for C++03
    // see also http://stackoverflow.com/questions/3054567/right-way-to-deallocate-an-stdvector-object

    //_integralImages.clear();
    IntegralImages().swap(_integralImages);
    //_metaData.clear();
    meta_data_t().swap(_metaData);
    return;
}


void LabeledData::add(const ImageData &data, const integral_channels_t &integralImage)
{
    if (data.imageClass == -1)
    {
        _numNegExamples++;
    }
    else
    {
        _numPosExamples++;
    }

    _metaData.push_back(data);
    _integralImages.push_back(integralImage);
    return;
}


//***************************************************************************************************
#if 1
gil::rgb8_image_t rescale(const gil::rgb8_image_t &image, float scale)
{
    int w = image.dimensions().x;
    int h = image.dimensions().y;

    w = int(w * scale + 0.5);
    h = int(h * scale + 0.5);
    gil::rgb8_image_t out(w, h);
    gil::resize_view(gil::const_view(image), gil::view(out), gil::bilinear_sampler());
    return out;
}
#endif


void getPositiveSamples(const std::vector<std::string>  &filenamesPositives,
                        const int width, const int height,
                        const int offsetX, const int offsetY,
                        LabeledData::meta_data_t &metaData,
                        LabeledData::IntegralImages &integralImages,
                        boost::progress_display &progress_indicator,
                        integral_channels_computer_t &integralChannelsComputer)
{

    // retrieve positives
    integralImages.resize(filenamesPositives.size());
    metaData.resize(filenamesPositives.size());
    for (size_t filenameIndex = 0; filenameIndex < filenamesPositives.size(); ++filenameIndex)
    {
        gil::rgb8_image_t image;
        gil::rgb8c_view_t image_view = doppia::open_image(filenamesPositives[filenameIndex].c_str(), image);

        integralChannelsComputer.set_image(image_view);
        integralChannelsComputer.compute();

        integral_channels_t &integralChannel = integralImages[filenameIndex];
        get_integral_channels(integralChannelsComputer.get_integral_channels(),
                              offsetX, offsetY, width, height,
                              integralChannelsComputer.get_shrinking_factor(),
                              integralChannel);
        ImageData d;
        d.scale =1.;
        d.filename = filenamesPositives[filenameIndex];
        d.scale = 1.;
        d.imageClass = 1;//classes[k];
        d.x = offsetX;
        d.y = offsetY ;
        metaData[filenameIndex] = d;

        ++progress_indicator;
    } // end of "for each filename"

    return;
}


// FIXME samplesPerImage is redudant with maxNegativeSamplesToAdd + filenamesBackground.size()
void getRandomNegativeSamples(const int samplesPerImage,
                              const size_t maxNegativeSamplesToAdd,
                              const std::vector<std::string>  &filenamesBackground,
                              const int width, const int height,
                              const int offsetX, const int offsetY,
                              LabeledData::meta_data_t &metaData,
                              LabeledData::IntegralImages &integralImages,
                              size_t &numNegExamples,
                              boost::progress_display &progress_indicator,
                              integral_channels_computer_t &integralChannelsComputer)

{
    size_t numNegativesSamplesAdded = 0;
    int numSkippedImages = 0;
    for (size_t filenameIndex = 0; filenameIndex < filenamesBackground.size(); filenameIndex +=1)
    {
        gil::rgb8c_view_t imageView;
        gil::rgb8_image_t image;
        imageView = doppia::open_image(filenamesBackground[filenameIndex].c_str(), image);

        // FIXME no idea what the +1 does
        if((imageView.width() < (width+1 + 2*offsetX)) or (imageView.height() < (height+1 + 2*offsetY)))
        {
            // if input image is too small, we skip it
            //printf("Skipping negative sample %s, because it is too small\n", filenamesBackground[filenameIndex].c_str());
            numSkippedImages += 1;
            continue;
        }

        integralChannelsComputer.set_image(imageView);
        integralChannelsComputer.compute();

        const size_t startingIndex = integralImages.size();
        const size_t newSize = startingIndex +
                std::min<size_t>(samplesPerImage, (maxNegativeSamplesToAdd - numNegativesSamplesAdded));
        integralImages.resize(newSize);
        metaData.resize(newSize);
        for (size_t integralChannelsIndex = startingIndex;
             integralChannelsIndex < integralImages.size();
             integralChannelsIndex += 1)
        {
            const int
                    xx = offsetX + rand() % (imageView.width() - width+1 - 2*offsetX),
                    yy = offsetY + rand() % (imageView.height() - height+1 - 2*offsetY);

            integral_channels_t &integralChannel = integralImages[integralChannelsIndex];
            get_integral_channels(integralChannelsComputer.get_integral_channels(),
                                  xx, yy, width, height,
                                  integralChannelsComputer.get_shrinking_factor(),
                                  integralChannel);
            ImageData d;
            d.scale = 1.;
            d.filename = filenamesBackground[filenameIndex];
            d.scale = 1.;
            d.imageClass = -1;//_backgroundClassLabel;
            d.x = xx;
            d.y = yy;
            metaData[integralChannelsIndex] = d;

            numNegativesSamplesAdded += 1;
            numNegExamples += 1;
            ++progress_indicator;
        }

        if(numNegativesSamplesAdded >= maxNegativeSamplesToAdd)
        {
            break;
        }
    } // end of "for each background image"


    const float skipped_fraction = static_cast<float>(numSkippedImages) / filenamesBackground.size();
    if (numSkippedImages > 0)
    {
        printf("Skipped %i images (out of %zi, %.3f%%) because they where too small\n",
               numSkippedImages, filenamesBackground.size(), skipped_fraction*100);
    }

    if (skipped_fraction > 0.25)
    {
        throw std::runtime_error("Too many negatives images where skipped. Dataset needs to be fixed");
    }

    return;
}


void LabeledData::bootstrap(const std::string classifierName,
                            const point_t modelWindow, const int offsetX, const int offsetY,
                            const int maxFalsePositives, const int maxFalsePositivesPerImage,
                            const std::vector<std::string> & negativeImagesPaths,
                            const boost::program_options::variables_map &options)
{


printf("Searching for hard negatives given the current model (%s), please wait...\n",
classifierName.c_str());

const float
min_scale = get_option_value<float>(options, "bootstrap_train.min_scale"),
max_scale = get_option_value<float>(options, "bootstrap_train.max_scale");

const int
num_scales = get_option_value<int>(options, "bootstrap_train.num_scales");

const float
min_ratio = get_option_value<float>(options, "bootstrap_train.min_ratio"),
max_ratio = get_option_value<float>(options, "bootstrap_train.max_ratio");

const int num_ratios = get_option_value<int>(options, "bootstrap_train.num_ratios");

const bool use_less_memory = get_option_value<bool>(options, "bootstrap_train.frugal_memory_usage");


    // will append false positives to the _integralImages
    const size_t initialIntegralImagesSize = _integralImages.size();
    std::vector<integral_channels_t> &falsePositives = _integralImages;
    std::vector<ImageData> &falsePositivesData = _metaData;
    bootstrapping::bootstrap(boost::filesystem::path(classifierName), negativeImagesPaths,
                             maxFalsePositives, maxFalsePositivesPerImage,
                             min_scale, max_scale, num_scales,
                             min_ratio, max_ratio, num_ratios,
                             use_less_memory,
                             falsePositivesData, falsePositives,
options);

    const size_t numFoundFalsePositives = falsePositives.size() - initialIntegralImagesSize;
    _numNegExamples += numFoundFalsePositives;

    const bool save_integral_images = false;
    if(save_integral_images)
    {
        boost::filesystem::path storage_path = "/tmp/";
        boost::format filename_pattern("false_positive_%i.png");

        const int max_images_to_save = 100;

        int false_positive_counter = 0;
        BOOST_FOREACH(const integral_channels_t &integral_channels, falsePositives)
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

    if(numFoundFalsePositives < static_cast<size_t>(maxFalsePositives))
    {

        const size_t numRandomNegativesToAdd = maxFalsePositives - numFoundFalsePositives;
        const int samplesPerImage = std::max<int>(1,
                                                  numRandomNegativesToAdd / negativeImagesPaths.size());

        std::vector<std::string> negativeImagesPathsSubSet;
        if(numRandomNegativesToAdd < negativeImagesPaths.size())
        {
            assert(samplesPerImage == 1);
            negativeImagesPathsSubSet.assign(negativeImagesPaths.begin(),
                                             negativeImagesPaths.end() - numRandomNegativesToAdd);
        }
        else
        {
            negativeImagesPathsSubSet.assign(negativeImagesPaths.begin(), negativeImagesPaths.end());
        }

        boost::progress_display progress(numRandomNegativesToAdd);

        integral_channels_computer_t integralChannelsComputer;

        getRandomNegativeSamples(samplesPerImage,
                                 numRandomNegativesToAdd,
                                 negativeImagesPathsSubSet,
                                 modelWindow.x(), modelWindow.y(), offsetX, offsetY,
                                 falsePositivesData, falsePositives,
                                 _numNegExamples,
                                 progress, integralChannelsComputer);

    }




    //verification DEBUG

    //int initialSize = _integralImages.size();
    //int noIncorrect = 0;
    //int noCorrect =0;
    //ModelIO modelReader;
    //modelReader.initRead(classifierName);
    //modelReader.print();
    //StrongClassifier classifier = modelReader.read();
    //for (int i = initialSize; i< initialSize+ falsePositives.size(); ++i){
    //////
    //    std::cout << "classifying image: " << _metaData[i].filename << std::endl;
    //////
    //
    //    integral_channels_const_view_t view = get_integral_channels_view(_integralImages[i],
    // 0,0, 64, 128,4); // 64x128 is not guaranteed anymore... should use model_size
    //    if (classifier.classify(view) ==-1){
    //        noIncorrect++;
    //        //std::cout <<     _metaData[i].filename << " class: " <<  _metaData[i].imageClass << " (x,y)= ("  <<  _metaData[i].x << "," <<  _metaData[i].y << "), scale: " << _metaData[i].scale << std::endl;
    //       // if ((_metaData[i].x != 0) != (_metaData[i].y !=0 ) ){
    //            std::cout <<     _metaData[i].filename << " class: " <<  _metaData[i].imageClass << " (x,y)= ("  <<  _metaData[i].x << "," <<  _metaData[i].y << "), scale: " << _metaData[i].scale << std::endl;
    //
    //    }
    ////
    //    else noCorrect++;
    //}
    //std::cout << "No correctly as false positive: " << noCorrect << " No of Incorrect false positives: " << noIncorrect << std::endl;
    return;
}

//int LabeledData::bootstrap(int winw, int winh, int maxNr,const StrongClassifier & learner,
// const bootstrapping::integral_channels_computer_t& integralImage, std::vector<integral_channels_t> & out, std::vector<imageData> & imgData){
//    out.clear();
//
//    //slide window and classify subimages
//
//    int w  = integralImage.get_input_size().x;
//    int h  = integralImage.get_input_size().y;
//    //DEBUG
//    std::cout << " w = " << w << " h = " << h << std::endl;
//    int counter = 0;
//
//
//    for (int i = 0; i< w - winw; i=i+4){
//        for (int j = 0 ; j < h- winh; j=j+4){
//            if (int(out.size()) > maxNr){
//                break;
//            }
//            integral_channels_const_view_t view =  integralImage.get_integral_channels_view(i,j, winw, winh);
//            if (learner.classify(view) ==1){
//                imageData d;
//                d.x = w;
//                d.y = j;
//                imgData.push_back(d);
//                integral_channels_t tmp;
//                integralImage.get_integral_channels(i,j,winw,winh,tmp);
//                out.push_back(tmp);
//            }
//            counter++;
//
//        }
//    }
//    return counter;
//}
//void LabeledData::bootstrap(const StrongClassifier & classifier, int maxNr,std::vector<std::string> & filenames_background)
//{
//
//    boost::array<float,5> scales = {1, 1.2,1.44, 0.83, 0.69};
//
//    int winChecked = 0;
//    int found = 0;
//    const int maxPerImage = 200;
//    #pragma omp parallel for default (none) shared(std::cout, winChecked, found, filenames_background, classifier, scales, maxNr)
//    for (size_t k = 0; k < filenames_background.size(); ++k){
//        if(found > maxNr)
//            continue;
//        gil::rgb8c_view_t image_view;
//        gil::rgb8_image_t image;
//        image_view = open_image(filenames_background[k].c_str(), image);
//        for (size_t sc = 0; sc < scales.size(); ++ sc){
//            if(found > maxNr)
//                continue;
//            image =rescale(image, scales[sc]);
//            image_view = gil::view(image);
//            integral_channels_computer_t integralImage(filenames_background[k]);
//
//            integralImage.set_image(image_view);
//            integralImage.compute();
//            std::vector<integral_channels_t> outPerImg;
//            std::vector<imageData> imageDataPerImg;
//            int wndchecked= bootstrap(this->_width, this->_height, maxPerImage, classifier, integralImage, outPerImg, imageDataPerImg);
//            int exFound = int(outPerImg.size());
//            #pragma omp critical
//            {
//                for (int j = 0; j< exFound; ++j){
//                    imageDataPerImg[j].filename = filenames_background[k];
//                    imageDataPerImg[j].imageClass = -1;//_backgroundClassLabel;
//                    imageDataPerImg[j].scale= scales[sc];//_backgroundClassLabel;
//                    _metaData.push_back(imageDataPerImg[j]);
//                }
//                std::cout << "example : " << filenames_background[k] << " scale: " << scales[sc] << " hardNegatives: " << exFound << std::endl;
//                _numNegExamples = _numNegExamples+ exFound;
//                _numExamples +=exFound;
//                _integralImages.insert(_integralImages.end(), outPerImg.begin(), outPerImg.end());
//                found +=exFound;
//                winChecked += wndchecked;
//
//                std::cout << "windows checkes: " << winChecked << " #sampled ones: " << found << std::endl;
//            }
//        }
//    }
//    //if (count++ % 200 == 0)
//    //    std::cout<< ".."<< float(count)/filenames_background.size() * 100.0<< "%\n"<<std::flush;
//    std::cout << "windows checkes: " << winChecked << " #sampled ones: " << found << std::endl;
//
//}



//***************************************************************************************************
void LabeledData::createIntegralImages(const std::vector<std::string> &filenamesPositives,
                                       const std::vector<std::string> &filenamesBackground,
                                       const point_t modelWindow,
                                       const int offsetX, const int offsetY,
                                       const boost::program_options::variables_map &options,
                                       const string bootStrapLearnerFile)
{

#if defined(DEBUG)
    srand(1);
#else
    srand(time(NULL));
#endif

    //=============================================================
    // eliminate any possible data  remaining there.
    _integralImages.clear();
    _numPosExamples = 0;
    _numNegExamples = 0;

    _metaData.clear();
    //=============================================================

    if (!_silent_mode)
    {
        //printf("Generating integral images for %s, please be patient...\n", imagesSetFile.c_str());
        printf("Generating integral images please be patient...\n");
    }

    _numPosExamples = filenamesPositives.size();

    const int width  = modelWindow.x(), height = modelWindow.y();

    const int trainNumNegativesSamples = get_option_value<int>(options, "train.num_negative_samples");
    int samplesPerImage = 1;

    if (not filenamesBackground.empty())
    {
        samplesPerImage = std::max<int>(1, trainNumNegativesSamples / filenamesBackground.size());
    }

    //cout << "Samples drawn per background image: " << samplesPerImage << endl;

    {
        //cout << "Computing features ..." << endl;

        // we pre-allocate the memory, to avoid memory re-allocations are run-time
        _integralImages.reserve(filenamesPositives.size() + trainNumNegativesSamples);
        _metaData.reserve(filenamesPositives.size() + trainNumNegativesSamples);

        // the integral channels computer is already fully multi-threaded,
        // so there is no particular benefit in processing the input images in parallel
        integral_channels_computer_t integralChannelsComputer;

        {
            boost::progress_display integral_images_progress(filenamesPositives.size() + trainNumNegativesSamples);

            getPositiveSamples(filenamesPositives,
                               width, height, offsetX, offsetY,
                               _metaData, _integralImages,
                               integral_images_progress, integralChannelsComputer);

            // now negatives --
            _numNegExamples = 0;
            getRandomNegativeSamples(samplesPerImage,
                                     trainNumNegativesSamples,
                                     filenamesBackground,
                                     width, height, offsetX, offsetY,
                                     _metaData, _integralImages,
                                     _numNegExamples, integral_images_progress, integralChannelsComputer);

        } // end of integral_images_progress life

    } // end of normal features loading

    if (bootStrapLearnerFile.empty() == false)
    {
        // use bootstrapping
        std::cout << "Reading bootStrapLearnerFile: " << bootStrapLearnerFile << std::endl;

        const int numBootstrappingSamples =
                get_option_value<int>(options, "bootstrap_train.num_bootstrapping_samples");
        const std::vector<int> maxNumSamplesPerImage =
                get_option_value<std::vector<int> >(options, "bootstrap_train.max_num_samples_per_image");

        // we assume a name like 2011_10_09_61602_trained_model.proto.bin.bootstrap2 and no more than 9 stages
        const int boostrapping_stage = boost::lexical_cast<int>(*(bootStrapLearnerFile.end() - 1)) + 1;

        std::cout << "Restarting from boostrapping_stage: " << boostrapping_stage << std::endl;

        if((boostrapping_stage < 0) or (boostrapping_stage >= static_cast<int>(maxNumSamplesPerImage.size())) )
        {
            throw std::runtime_error("Could not deduce the maxNumSamplesPerImage given "
                                     "bootStrapLearnerFile and bootstrapTrain.max_num_samples_per_image");
        }

        if(boostrapping_stage > 0)
        {
            bootstrap(bootStrapLearnerFile,
                      modelWindow, offsetX, offsetY,
                      numBootstrappingSamples*boostrapping_stage,
                      maxNumSamplesPerImage[boostrapping_stage],
                      filenamesBackground,
                      options);
        }
    }

    if ((_numPosExamples + _numNegExamples) != get_num_examples())
    {
        throw runtime_error("LabeledData::createIntegralImages (pos + neg examples) != (total number of examples), "
                            "something went terribly wrong");
    }

    if (!_silent_mode)
    {
        std::cout << "Examples loaded\t= " << get_num_examples() << std::endl;
        std::cout << "\tNegative samples\t= " << _numNegExamples << std::endl;
        std::cout << "\tPositive samples\t= " << _numPosExamples << std::endl;
    }

    return;
}

} // end of namespace boosted_learning
