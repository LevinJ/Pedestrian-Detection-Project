
#include "Feature.hpp"

#include <boost/random.hpp>

#include <stdexcept>
#include <iostream>
#include <ctime>
#include <fstream>

namespace boosted_learning {

void getFeatureConfigurationsFromFile(Features &featuresConfigurations, std::string featureConfigurationsFilename);

void getRandomFeaturesVerticallyMirrored(const point_t &modelWindow,
                                         const int numOfFeatures, const int numChannels,
                                         Features &featuresConfigurations,
                                         const boost::program_options::variables_map &options);

void getRandomFeaturesGradientMirrored(const point_t &modelWindow,
                                       const int numOfFeatures, const int numChannels,
                                       Features &featuresConfigurations,
                                       const boost::program_options::variables_map &options);

void getChannelInvariantFeatures(const point_t &modelWindow,
                                 const int numOfFeatures, const int numChannels,
                                 Features &featuresConfigurations,
                                 const boost::program_options::variables_map &options);

void getHogLikeMultiScale(const point_t &modelWindow,
                          const int numChannels,
                          Features &featuresConfigurations);

void getHogLikeFeatures(const int cellSize,
                        const point_t &modelWindow,
                        const int numChannels,
                        Features &featuresConfigurations);

void getRandomFeaturesGradientMirroredCopy(const point_t &modelWindow,
                                           const int numOfFeatures, const int numChannels,
                                           Features &featuresConfigurations,
                                           const boost::program_options::variables_map &options);

void getAllFeatures(const point_t &modelWindow,
                    const int numChannels,
                    Features &featuresConfigurations);

Feature mirrorFeatureVertically(const int modelWidth, const Feature &feature);

void writeFeaturesToFile(std::string filename , Features &featuresConfigurations);



/// The constructor that set everything to 0.
Feature::Feature() : x(0), y(0), width(0), height(0), channel(0)
{
    // nothing to do here
    return;
}


/// The constructor that ask for the sizes.
Feature::Feature(const int x_, const int y_, const int width_, const int height_, const int channel_)
    : x(x_), y(y_), width(width_), height(height_), channel(channel_)
{
    // nothing to do here
    return;
}

bool Feature::operator==(const Feature &other) const
{
    return (x == other.x) and (y == other.y) and
            (width == other.width) and (height == other.height) and (channel == other.channel);
}


void Feature::printConfigurationString(std::ostream &os) const
{
    os << x << "\t" << y << "\t" << width << "\t" << height << "\t" << channel << "\t";
    return;
}

void Feature::readConfigurationString(std::istream &is)
{
    if (!(is >> x >> y >> width >> height >> channel))
    {
        throw std::runtime_error("Reading configuration failed");
    }
    return;
}


void writeFeaturesToFile(std::string filename , Features &featuresConfigurations)
{
    std::ofstream myfile;
    myfile.open (filename.c_str());

    for (size_t i =0; i< featuresConfigurations.size(); ++i)
    {
        const Feature &f = featuresConfigurations[i];
        myfile << f.x << " " << f.y << " " << f.width << " " << f.height << " " << f.channel << "\n";
    }

    myfile.close();
    return;
}


void getFeaturesConfigurations(const point_t &modelWindow,
                               const int numOfFeatures,
                               const int numChannels,
                               Features &featuresConfigurations,
                               const std::string &configType,
                               const boost::program_options::variables_map &options)
{

    if (configType == "random")
    {
        computeRandomFeaturesConfigurations(modelWindow, numOfFeatures, numChannels, featuresConfigurations, options);
    }
    else if (configType == "file")
    {
        getFeatureConfigurationsFromFile(featuresConfigurations, get_option_value<std::string>(options, "train.features_configurations_file"));

    }
    else if (configType == "random-squares")
    {
        computeRandomFeaturesConfigurations(modelWindow, numOfFeatures, numChannels, featuresConfigurations, options, true);
    }
    else if ( configType == "HOG-like")
    {
        // 8 pixels, means size 2 because we assume shrinking factor 4
        getHogLikeFeatures(2, modelWindow, numChannels, featuresConfigurations);
    }
    else if ( configType == "HOG-like-1,2,4,8,16")
    {
        // 8 pixels, means size 2 because we assume shrinking factor 4
        getHogLikeFeatures(1, modelWindow, numChannels, featuresConfigurations);
        getHogLikeFeatures(2, modelWindow, numChannels, featuresConfigurations);
        //getHogLikeFeatures(3, modelWindow, numChannels, featuresConfigurations);
        getHogLikeFeatures(4, modelWindow, numChannels, featuresConfigurations);
        //getHogLikeFeatures(5, modelWindow, numChannels, featuresConfigurations);
        //getHogLikeFeatures(5, modelWindow, numChannels, featuresConfigurations);
    }
    else if (configType == "HOG-multiScale")
    {
        getHogLikeMultiScale(modelWindow, numChannels, featuresConfigurations);
    }
    else if (configType == "HOG-multiScale-gray")
    {
        getHogLikeMultiScaleGray(modelWindow, featuresConfigurations);
    }
    else if (configType == "rect-ratio2")
    {
        getRectRatioTwo(modelWindow, numChannels, featuresConfigurations);
    }
    else if (configType == "rect2+HOG-multiScale")
    {
        getRectRatioTwo(modelWindow, numChannels, featuresConfigurations);
        getHogLikeMultiScale(modelWindow, numChannels, featuresConfigurations);
    }
    else if (configType == "channelInvariant")
    {
        getChannelInvariantFeatures(modelWindow, numOfFeatures, numChannels, featuresConfigurations, options);
    }
    else if (configType == "gradientMirrored")
    {
        getRandomFeaturesGradientMirrored(modelWindow, numOfFeatures, numChannels, featuresConfigurations, options);
    }
    else if (configType == "verticalSymmetry")
    {
        getRandomFeaturesVerticallyMirrored(modelWindow, numOfFeatures, numChannels, featuresConfigurations, options);
    }
    else if (configType == "gradientMirroredCopy")
    {
        getRandomFeaturesGradientMirroredCopy(modelWindow, numOfFeatures, numChannels, featuresConfigurations, options);
    }
    else if (configType == "allRectangles")
    {
        getAllFeatures(modelWindow, numChannels, featuresConfigurations);
    }
    else
    {
        throw std::runtime_error("no known feature configuration type given!");
    }

    return;
}


void getRandomFeaturesVerticallyMirrored(const point_t &modelWindow,
                                         const int numOfFeatures, const int numChannels,
                                         Features &featuresConfigurations,
                                         const boost::program_options::variables_map &options)
{

    //if (_verbose > 2)
    if(false)
    {
        std::cout << "Computing the features configurations" << std::endl;
    }


    const int
            shrinking_factor = bootstrapping::integral_channels_computer_t::get_shrinking_factor(),
            modelWidth = modelWindow.x() / shrinking_factor,
            modelHeight = modelWindow.y() / shrinking_factor;

    //const float maxFeatureSizeRatio = get_option_value<float>(options, "train.max_feature_size_ratio");
    //maxWidth = static_cast<int>(maxFeatureSizeRatio * modelWidth),
    //maxHeight = static_cast<int>(maxFeatureSizeRatio * modelHeight);

    const int
            minWidth = std::max(1, get_option_value<int>(options, "train.min_feature_width") / shrinking_factor),
            minHeight = std::max(1, get_option_value<int>(options, "train.min_feature_height") / shrinking_factor);

    int
            maxWidth = get_option_value<int>(options, "train.max_feature_width"),
            maxHeight = get_option_value<int>(options, "train.max_feature_height");

    if(maxWidth < 0)
    {
        maxWidth = modelWidth;
    }
    else
    {
        maxWidth = std::max(1, maxWidth/shrinking_factor);
    }

    if(maxHeight < 0)
    {
        maxHeight = modelHeight;
    }
    else
    {
        maxHeight = std::max(1, maxHeight/shrinking_factor);
    }

    if((minWidth >= maxWidth) or (minHeight >= maxHeight))
    {
        throw std::invalid_argument("min width/height should be smaller than max width/height (check your configuration file)");
    }

    static int call_counter=0;
    boost::uint32_t random_seed = std::time(NULL);
    const int input_seed = get_option_value<boost::uint32_t>(options, "train.feature_pool_random_seed");
    if(input_seed > 0)
    {
        random_seed = input_seed+call_counter;
        call_counter+=1;
        printf("computeRandomFeaturesConfigurations is using user provided seed == %i\n", random_seed);
    }
    else
    {
        printf("computeRandomFeaturesConfigurations is using random_seed == %i\n", random_seed);
    }

    boost::mt19937 random_generator(random_seed);

    typedef boost::variate_generator<boost::mt19937&, boost::uniform_int<> > uniform_generator_t;

    // the distribution boundaries are inclusive
    boost::uniform_int<>
            x_distribution(0, (modelWidth ) - minWidth),
            y_distribution(0, (modelHeight) - minHeight),
            channel_distribution(0, numChannels - 1),
            width_distribution(minWidth, maxWidth - 1),
            height_distribution(minHeight, maxHeight - 1);

    if((x_distribution.max() <= 0) or
            (y_distribution.max() <= 0) or
            (width_distribution.max() <= 0) or
            (height_distribution.max() <= 0))
    {
        printf("shrinked model (width, height) == (%i, %i)\n", modelWidth, modelHeight);
        printf("min feature size (after shrinking) (width, height) == (%i, %i)\n", minWidth, minHeight);
        throw invalid_argument("It seems that minFeatWidth or minFeatHeight is bigger than the model size after shrinking");
    }

    uniform_generator_t
            x_generator(random_generator, x_distribution),
            y_generator(random_generator, y_distribution),
            channel_generator(random_generator, channel_distribution),
            width_generator(random_generator, width_distribution),
            height_generator(random_generator, height_distribution);

    size_t total_numOfFeatures = featuresConfigurations.size() + numOfFeatures;
    featuresConfigurations.reserve(total_numOfFeatures);

    int rejectionsInARow = 0, repetitionsCounter = 0;
    const int maxRejectionsInARow = 1000; // how many continuous rejection do we accept ?

    while(featuresConfigurations.size() < total_numOfFeatures)
    {
        const int
                x = x_generator(),
                y = y_generator(),
                c = channel_generator(),
                w = width_generator(),
                h = height_generator();
        //std::cout << x << " ";
        if(((x + w) <= modelWidth) and ((y + h) <= modelHeight))
        {
            Feature featureConfiguration(x, y, w, h, c);


            // we check if the feature already exists in the set or not
            const bool featureAlreadyInSet =
                    std::find(featuresConfigurations.begin(), featuresConfigurations.end(),
                              featureConfiguration) != featuresConfigurations.end();

            if(featureAlreadyInSet)
            {
                rejectionsInARow += 1;
                repetitionsCounter += 1;
                if(rejectionsInARow > maxRejectionsInARow)
                {
                    printf("once featuresPool reached size %zi, failed to find a new feature after %i attempts\n",
                           featuresConfigurations.size(), maxRejectionsInARow);
                    throw std::runtime_error("Failed to generate the requested features pool, is featuresPoolSize too big?");
                }
                continue;
            }
            else
            {

                rejectionsInARow = 0;
                featuresConfigurations.push_back(featureConfiguration);

                Feature newFeat = mirrorFeatureVertically(modelWidth, featureConfiguration);
                featuresConfigurations.push_back(newFeat);
                //                if (not(mirrorFeatureVertically(modelWidth,newFeat) == featureConfiguration)){
                //                    throw std::runtime_error("oh jeh, thats wrong");

                //                }else{
                //                    featureConfiguration.printConfigurationString(std::cout);
                //                    std::cout << std::endl;
                //                    mirrorFeatureVertically(modelWidth,newFeat).printConfigurationString(std::cout);
                //                    std::cout << std::endl;
                //                    std::cout << std::endl;
                //                }

            }
        } // end of "if the random feature has proper size"
    } // end of "while not enough features computed"

    if(true)
    {
        printf("When sampling %i features, randomly found (and rejected) %i repetitions\n",
               numOfFeatures, repetitionsCounter);
    }

    return;
} // end getRandomFeaturesVerticallyMirrored


void getRandomFeaturesGradientMirrored(const point_t &modelWindow,
                                       const int numOfFeatures, const int numChannels,
                                       Features &featuresConfigurations,
                                       const boost::program_options::variables_map &options)
{

    //if (_verbose > 2)
    if(false)
    {
        std::cout << "Computing the features configurations" << std::endl;
    }

    const int
            shrinking_factor = bootstrapping::integral_channels_computer_t::get_shrinking_factor(),
            modelWidth = modelWindow.x() / shrinking_factor,
            modelHeight = modelWindow.y() / shrinking_factor;

    //const float maxFeatureSizeRatio = get_option_value<float>(options, "train.max_feature_size_ratio");
    //maxWidth = static_cast<int>(maxFeatureSizeRatio * modelWidth),
    //maxHeight = static_cast<int>(maxFeatureSizeRatio * modelHeight)

    const int
            minWidth = std::max(1, get_option_value<int>(options, "train.min_feature_width") / shrinking_factor),
            minHeight = std::max(1, get_option_value<int>(options, "train.min_feature_height") / shrinking_factor);

    int
            maxWidth = get_option_value<int>(options, "train.max_feature_width"),
            maxHeight = get_option_value<int>(options, "train.max_feature_height");

    if(maxWidth < 0)
    {
        maxWidth = modelWidth;
    }
    else
    {
        maxWidth = std::max(1, maxWidth/shrinking_factor);
    }

    if(maxHeight < 0)
    {
        maxHeight = modelHeight;
    }
    else
    {
        maxHeight = std::max(1, maxHeight/shrinking_factor);
    }

    if((minWidth >= maxWidth) or (minHeight >= maxHeight))
    {
        throw std::invalid_argument("min width/height should be smaller than max width/height (check your configuration file)");
    }

    static int call_counter=0;
    boost::uint32_t random_seed = std::time(NULL);
    const int input_seed = get_option_value<boost::uint32_t>(options, "train.feature_pool_random_seed");
    if(input_seed > 0)
    {
        random_seed = input_seed+call_counter;
        call_counter+=1;
        printf("computeRandomFeaturesConfigurations is using user provided seed == %i\n", random_seed);
    }
    else
    {
        printf("computeRandomFeaturesConfigurations is using random_seed == %i\n", random_seed);
    }

    boost::mt19937 random_generator(random_seed);

    typedef boost::variate_generator<boost::mt19937&, boost::uniform_int<> > uniform_generator_t;

    // the distribution boundaries are inclusive
    boost::uniform_int<>
            x_distribution(0, (modelWidth) - minWidth),
            y_distribution(0, (modelHeight) - minHeight),
            channel_distribution(0, numChannels - 1),
            width_distribution(minWidth, maxWidth - 1),
            height_distribution(minHeight, maxHeight - 1);

    if((x_distribution.max() <= 0) or
            (y_distribution.max() <= 0) or
            (width_distribution.max() <= 0) or
            (height_distribution.max() <= 0))
    {
        printf("shrinked model (width, height) == (%i, %i)\n", modelWidth, modelHeight);
        printf("min feature size (after shrinking) (width, height) == (%i, %i)\n", minWidth, minHeight);
        throw invalid_argument("It seems that minFeatWidth or minFeatHeight is bigger than the model size after shrinking");
    }

    uniform_generator_t
            x_generator(random_generator, x_distribution),
            y_generator(random_generator, y_distribution),
            channel_generator(random_generator, channel_distribution),
            width_generator(random_generator, width_distribution),
            height_generator(random_generator, height_distribution);

    size_t total_numOfFeatures = featuresConfigurations.size() + numOfFeatures;
    featuresConfigurations.reserve(total_numOfFeatures);

    int rejectionsInARow = 0, repetitionsCounter = 0;
    const int maxRejectionsInARow = 1000; // how many continuous rejection do we accept ?

    while(featuresConfigurations.size() < total_numOfFeatures)
    {
        const int
                x = x_generator(),
                y = y_generator(),
                c = channel_generator(),
                w = width_generator(),
                h = height_generator();
        //std::cout << x << " ";
        if(((x + w) <= modelWidth) and ((y + h) <= modelHeight))
        {
            Feature featureConfiguration(x, y, w, h, c);


            // we check if the feature already exists in the set or not
            const bool featureAlreadyInSet =
                    std::find(featuresConfigurations.begin(), featuresConfigurations.end(),
                              featureConfiguration) != featuresConfigurations.end();

            if(featureAlreadyInSet)
            {
                rejectionsInARow += 1;
                repetitionsCounter += 1;
                if(rejectionsInARow > maxRejectionsInARow)
                {
                    printf("once featuresPool reached size %zi, failed to find a new feature after %i attempts\n",
                           featuresConfigurations.size(), maxRejectionsInARow);
                    throw std::runtime_error("Failed to generate the requested features pool, is featuresPoolSize too big?");
                }
                continue;
            }
            else
            {

                rejectionsInARow = 0;
                featuresConfigurations.push_back(featureConfiguration);
                if (c == 1 ){
                    Feature newFeat = mirrorFeatureVertically(modelWidth,featureConfiguration);
                    newFeat.channel = 5;
                    featuresConfigurations.push_back(newFeat);
                }else if (c == 2){
                    Feature newFeat = mirrorFeatureVertically(modelWidth,featureConfiguration);
                    newFeat.channel = 4;
                    featuresConfigurations.push_back(newFeat);

                }else if (c == 4){
                    Feature newFeat = mirrorFeatureVertically(modelWidth,featureConfiguration);
                    newFeat.channel = 2;
                    featuresConfigurations.push_back(newFeat);
                }
                else if (c == 5){
                    Feature newFeat = mirrorFeatureVertically(modelWidth,featureConfiguration);
                    newFeat.channel = 1;
                    featuresConfigurations.push_back(newFeat);
                }


            }
        } // end of "if the random feature has proper size"
    } // end of "while not enough features computed"

    if(true)
    {
        printf("When sampling %i features, randomly found (and rejected) %i repetitions\n",
               numOfFeatures, repetitionsCounter);
    }

    return;
}
// end getRandomFeaturesGradientMirrored


void getRandomFeaturesGradientMirroredCopy(const point_t &modelWindow,
                                           const int numOfFeatures, const int numChannels,
                                           Features &featuresConfigurations,
                                           const boost::program_options::variables_map &options)
{

    //if (_verbose > 2)
    if(false)
    {
        std::cout << "Computing the features configurations" << std::endl;
    }


    const int
            shrinking_factor = bootstrapping::integral_channels_computer_t::get_shrinking_factor(),
            modelWidth = modelWindow.x() / shrinking_factor,
            modelHeight = modelWindow.y() / shrinking_factor;

    //const float maxFeatureSizeRatio = get_option_value<float>(options, "train.max_feature_size_ratio");
    //maxWidth = static_cast<int>(maxFeatureSizeRatio * modelWidth),
    //maxHeight = static_cast<int>(maxFeatureSizeRatio * modelHeight)

    const int
            minWidth = std::max(1, get_option_value<int>(options, "train.min_feature_width") / shrinking_factor),
            minHeight = std::max(1, get_option_value<int>(options, "train.min_feature_height") / shrinking_factor);

    int
            maxWidth = get_option_value<int>(options, "train.max_feature_width"),
            maxHeight = get_option_value<int>(options, "train.max_feature_height");

    if(maxWidth < 0)
    {
        maxWidth = modelWidth;
    }
    else
    {
        maxWidth = std::max(1, maxWidth/shrinking_factor);
    }

    if(maxHeight < 0)
    {
        maxHeight = modelHeight;
    }
    else
    {
        maxHeight = std::max(1, maxHeight/shrinking_factor);
    }

    if((minWidth >= maxWidth) or (minHeight >= maxHeight))
    {
        throw std::invalid_argument("min width/height should be smaller than max width/height (check your configuration file)");
    }

    static int call_counter=0;
    boost::uint32_t random_seed = std::time(NULL);
    const int input_seed = get_option_value<boost::uint32_t>(options, "train.feature_pool_random_seed");
    if(input_seed > 0)
    {
        random_seed = input_seed+call_counter;
        call_counter+=1;
        printf("computeRandomFeaturesConfigurations is using user provided seed == %i\n", random_seed);
    }
    else
    {
        printf("computeRandomFeaturesConfigurations is using random_seed == %i\n", random_seed);
    }

    boost::mt19937 random_generator(random_seed);

    typedef boost::variate_generator<boost::mt19937&, boost::uniform_int<> > uniform_generator_t;

    // the distribution boundaries are inclusive
    boost::uniform_int<>
            x_distribution(0, (modelWidth ) - minWidth),
            y_distribution(0, (modelHeight ) - minHeight),

            width_distribution(minWidth, maxWidth - 1),
            height_distribution(minHeight, maxHeight - 1);

    if((x_distribution.max() <= 0) or
            (y_distribution.max() <= 0) or
            (width_distribution.max() <= 0) or
            (height_distribution.max() <= 0))
    {
        printf("shrinked model (width, height) == (%i, %i)\n", modelWidth, modelHeight);
        printf("min feature size (after shrinking) (width, height) == (%i, %i)\n", minWidth, minHeight);
        throw invalid_argument("It seems that minFeatWidth or minFeatHeight is bigger than the model size after shrinking");
    }

    uniform_generator_t
            x_generator(random_generator, x_distribution),
            y_generator(random_generator, y_distribution),

            width_generator(random_generator, width_distribution),
            height_generator(random_generator, height_distribution);

    size_t total_numOfFeatures = featuresConfigurations.size() + numOfFeatures;
    featuresConfigurations.reserve(total_numOfFeatures);

    int rejectionsInARow = 0, repetitionsCounter = 0;
    const int maxRejectionsInARow = 1000; // how many continuous rejection do we accept ?

    while(featuresConfigurations.size() < total_numOfFeatures)
    {
        const int
                x = x_generator(),
                y = y_generator(),
                w = width_generator(),
                h = height_generator();
        //std::cout << x << " ";
        if(((x + w) <= modelWidth) and ((y + h) <= modelHeight))
        {

            Feature featureConfiguration(x, y, w, h, 0);


            // we check if the feature already exists in the set or not
            const bool featureAlreadyInSet =
                    std::find(featuresConfigurations.begin(), featuresConfigurations.end(),
                              featureConfiguration) != featuresConfigurations.end();

            if(featureAlreadyInSet)
            {
                rejectionsInARow += 1;
                repetitionsCounter += 1;
                if(rejectionsInARow > maxRejectionsInARow)
                {
                    printf("once featuresPool reached size %zi, failed to find a new feature after %i attempts\n",
                           featuresConfigurations.size(), maxRejectionsInARow);
                    throw std::runtime_error("Failed to generate the requested features pool, is featuresPoolSize too big?");
                }
                continue;
            }
            else
            {

                rejectionsInARow = 0;
                Feature featureConfigurationm = mirrorFeatureVertically(modelWidth,featureConfiguration);

                for (int c =0; c< numChannels; ++ c){
                    featuresConfigurations.push_back(Feature(featureConfiguration.x, featureConfiguration.y,featureConfiguration.width, featureConfiguration.height,c));

                    featuresConfigurations.push_back(Feature(featureConfigurationm.x, featureConfigurationm.y,featureConfigurationm.width, featureConfigurationm.height,c));

                }


            }
        } // end of "if the random feature has proper size"
    } // end of "while not enough features computed"

    if(true)
    {
        printf("When sampling %i features, randomly found (and rejected) %i repetitions\n",
               numOfFeatures, repetitionsCounter);
    }

    return;
}
// end getRandomFeaturesGradientMirroredCopy



Feature mirrorFeatureVertically(const int modelWidth, const Feature &feature)
{
    int x1 = feature.x;
    int x2 = x1 + feature.width;

    int x1n = modelWidth - x2;
    Feature ret(x1n, feature.y, feature.width,feature.height,feature.channel);
    return ret;

}


void getChannelInvariantFeatures(const point_t &modelWindow,
                                 const int numOfFeatures, const int numChannels,
                                 Features &featuresConfigurations,
                                 const boost::program_options::variables_map &options)
{
    const int
            shrinking_factor = bootstrapping::integral_channels_computer_t::get_shrinking_factor(),
            modelWidth = modelWindow.x() / shrinking_factor,
            modelHeight = modelWindow.y() / shrinking_factor;

    const int
            minWidth = std::max(1, get_option_value<int>(options, "train.min_feature_width") / shrinking_factor),
            minHeight = std::max(1, get_option_value<int>(options, "train.min_feature_height") / shrinking_factor);

    int
            maxWidth = get_option_value<int>(options, "train.max_feature_width"),
            maxHeight = get_option_value<int>(options, "train.max_feature_height");

    if(maxWidth < 0)
    {
        maxWidth = modelWidth;
    }
    else
    {
        maxWidth = std::max(1, maxWidth/shrinking_factor);
    }

    if(maxHeight < 0)
    {
        maxHeight = modelHeight;
    }
    else
    {
        maxHeight = std::max(1, maxHeight/shrinking_factor);
    }

    if((minWidth >= maxWidth) or (minHeight >= maxHeight))
    {
        throw std::invalid_argument("min width/height should be smaller than max width/height (check your configuration file)");
    }


    static int call_counter=0;
    boost::uint32_t random_seed = std::time(NULL);
    const int input_seed = get_option_value<boost::uint32_t>(options, "train.feature_pool_random_seed");
    if(input_seed > 0)
    {
        random_seed = input_seed+call_counter;
        call_counter+=1;
        printf("computeRandomFeaturesConfigurations is using user provided seed == %i\n", random_seed);
    }
    else
    {
        printf("computeRandomFeaturesConfigurations is using random_seed == %i\n", random_seed);
    }

    boost::mt19937 random_generator(random_seed);

    typedef boost::variate_generator<boost::mt19937&, boost::uniform_int<> > uniform_generator_t;

    // the distribution boundaries are inclusive
    boost::uniform_int<>
            x_distribution(0, (modelWidth) - minWidth),
            y_distribution(0, (modelHeight) - minHeight),
            width_distribution(minWidth, maxWidth - 1),
            height_distribution(minHeight, maxHeight - 1);

    if((x_distribution.max() <= 0) or
            (y_distribution.max() <= 0) or
            (width_distribution.max() <= 0) or
            (height_distribution.max() <= 0))
    {
        printf("shrinked model (width, height) == (%i, %i)\n", modelWidth, modelHeight);
        printf("min feature size (after shrinking) (width, height) == (%i, %i)\n", minWidth, minHeight);
        throw invalid_argument("It seems that minFeatWidth or minFeatHeight is bigger than the model size after shrinking");
    }

    uniform_generator_t
            x_generator(random_generator, x_distribution),
            y_generator(random_generator, y_distribution),
            width_generator(random_generator, width_distribution),
            height_generator(random_generator, height_distribution);

    size_t total_numOfFeatures = featuresConfigurations.size() + numOfFeatures;
    featuresConfigurations.reserve(total_numOfFeatures);

    int rejectionsInARow = 0, repetitionsCounter = 0;
    const int maxRejectionsInARow = 1000; // how many continuous rejection do we accept ?

    while(featuresConfigurations.size() < total_numOfFeatures)
    {
        const int
                x = x_generator(),
                y = y_generator(),
                w = width_generator(),
                h = height_generator();
        //std::cout << x << " ";
        if(((x + w) < modelWidth) and ((y + h) < modelHeight))
        {
            for (int c = 0; c< numChannels; ++c){
                Feature featureConfiguration(x, y, w, h, c);

                // we check if the feature already exists in the set or not
                const bool featureAlreadyInSet =
                        std::find(featuresConfigurations.begin(), featuresConfigurations.end(),
                                  featureConfiguration) != featuresConfigurations.end();

                if(featureAlreadyInSet)
                {
                    rejectionsInARow += 1;
                    repetitionsCounter += 1;
                    if(rejectionsInARow > maxRejectionsInARow)
                    {
                        printf("once featuresPool reached size %zi, failed to find a new feature after %i attempts\n",
                               featuresConfigurations.size(), maxRejectionsInARow);
                        throw std::runtime_error("Failed to generate the requested features pool, is featuresPoolSize too big?");
                    }
                    break;
                }
                else
                {
                    rejectionsInARow = 0;
                    featuresConfigurations.push_back(featureConfiguration);
                }
            }
        } // end of "if the random feature has proper size"
    } // end of "while not enough features computed"

    if(true)
    {
        printf("When sampling %i features, randomly found (and rejected) %i repetitions\n",
               numOfFeatures, repetitionsCounter);
    }

    return;

}


void getHogLikeMultiScaleGray(const point_t &modelWindow, Features &featuresConfigurations)
{

    const int
            shrinking_factor = bootstrapping::integral_channels_computer_t::get_shrinking_factor(),
            modelWidth = modelWindow.x() / shrinking_factor,
            modelHeight = modelWindow.y() / shrinking_factor;

    for (int cellSize =1; cellSize <= min(modelWidth, modelHeight); ++cellSize)
    {
        getHogLikeFeaturesGray(cellSize, modelWindow, featuresConfigurations);
    }

    return;
}


void getHogLikeMultiScale(const point_t &modelWindow,
                          const int numChannels,
                          Features &featuresConfigurations)
{
    const int
            shrinking_factor = bootstrapping::integral_channels_computer_t::get_shrinking_factor(),
            modelWidth = modelWindow.x() / shrinking_factor,
            modelHeight = modelWindow.y() / shrinking_factor;

    for (int cellSize =1; cellSize <= min(modelWidth, modelHeight); ++ cellSize)
    {
        getHogLikeFeatures(cellSize, modelWindow, numChannels, featuresConfigurations);
    }

    return;
}


void getHogLikeFeaturesGray(const int cellSize,
                            const point_t &modelWindow,
                            Features &featuresConfigurations)
{
    const int
            shrinking_factor = bootstrapping::integral_channels_computer_t::get_shrinking_factor(),
            modelWidth = modelWindow.x() / shrinking_factor,
            modelHeight = modelWindow.y() / shrinking_factor,
            firstGrayChannel = 0,
            lastGrayChannel = 7;

    for (int c = firstGrayChannel; c <= lastGrayChannel; c +=1)
    {
        for (int col = 0; col <= (modelWidth - cellSize); col += 1)
        {
            for (int row = 0; row<= (modelHeight -cellSize); row += 1)
            {
                Feature featureConfiguration(col, row, cellSize, cellSize, c);
                featuresConfigurations.push_back(featureConfiguration);

            } // end of "for each row"
        } // end of "for each column"
    } // end of "for each channel"

    return;
}

void getHogLikeFeatures(const int cellSize, const point_t &modelWindow,
                        const int numChannels,
                        Features &featuresConfigurations)
{

    const int
            shrinking_factor = bootstrapping::integral_channels_computer_t::get_shrinking_factor(),
            modelWidth = modelWindow.x() / shrinking_factor,
            modelHeight = modelWindow.y() / shrinking_factor;

    for (int c = 0; c < numChannels; c += 1)
    {
        for (int col = 0; col <= (modelWidth - cellSize); col += 1)
        {
            for (int row = 0; row <= (modelHeight -cellSize); row += 1)
            {
                const Feature featureConfiguration(col, row, cellSize, cellSize, c);
                featuresConfigurations.push_back(featureConfiguration);
            } // end of "for each row"
        } // end of "for each column"
    } // end of "for each channel"

    return;
}

void getFeatureConfigurationsFromFile(Features &featuresConfigurations, const std::string filename)
{
    int repetitionsCounter = 0;
    std::string line;

    std::ifstream myfile (filename.c_str());
    if (!myfile) throw 42;
    while (std::getline(myfile, line))
    {
        std::stringstream ss(line);
        int x, y, w, h, c;
        ss >> x >> y >> w >> h >> c;
        //std::cerr << "line: " << line << "\n";
        //std::cerr << x << " " << c << "\n";
        Feature featureConfiguration(x, y, w, h, c);
        const bool featureAlreadyInSet =
                std::find(featuresConfigurations.begin(), featuresConfigurations.end(),
                          featureConfiguration) != featuresConfigurations.end();
        if(featureAlreadyInSet)
        {
            repetitionsCounter += 1;
        }else{
            featuresConfigurations.push_back( featureConfiguration);
        }
    }
    if(true)
    {
        printf("Number of features that have already been in the Pool are %i\n",
               repetitionsCounter);
    }




}


void computeRandomFeaturesConfigurations(const point_t &modelWindow,
                                         const int numOfFeatures, const int numChannels,
                                         Features &featuresConfigurations,
                                         const boost::program_options::variables_map &options,
                                         bool randomSquares)
{

    //if (_verbose > 2)
    if(false)
    {
        std::cout << "Computing the features configurations" << std::endl;
    }


    //const float maxFeatureSizeRatio = get_option_value<float>(options, "train.max_feature_size_ratio");

    const int
            shrinking_factor = bootstrapping::integral_channels_computer_t::get_shrinking_factor(),
            modelWidth = modelWindow.x() / shrinking_factor,
            modelHeight = modelWindow.y() / shrinking_factor;

    const int
            minWidth = std::max(1, get_option_value<int>(options, "train.min_feature_width") / shrinking_factor),
            minHeight = std::max(1, get_option_value<int>(options, "train.min_feature_height") / shrinking_factor);
    //maxWidth = static_cast<int>(maxFeatureSizeRatio * modelWidth),
    //maxHeight = static_cast<int>(maxFeatureSizeRatio * modelHeight);

    int
            maxWidth = get_option_value<int>(options, "train.max_feature_width"),
            maxHeight = get_option_value<int>(options, "train.max_feature_height");

    if(maxWidth < 0)
    {
        maxWidth = modelWidth;
    }
    else
    {
        maxWidth = std::max(1, maxWidth/shrinking_factor);
    }

    if(maxHeight < 0)
    {
        maxHeight = modelHeight;
    }
    else
    {
        maxHeight = std::max(1, maxHeight/shrinking_factor);
    }

    if((minWidth >= maxWidth) or (minHeight >= maxHeight))
    {
        throw std::invalid_argument("min width/height should be smaller than max width/height (check your configuration file)");
    }

    static int call_counter = 0;
    boost::uint32_t random_seed = std::time(NULL);
    const int input_seed = get_option_value<uint32_t>(options, "train.feature_pool_random_seed");

    if(input_seed > 0)
    {
        random_seed = input_seed + call_counter;
        call_counter += 1;
        printf("computeRandomFeaturesConfigurations is using user provided seed == %i\n", random_seed);
    }
    else
    {
        printf("computeRandomFeaturesConfigurations is using random_seed == %i\n", random_seed);
    }

    boost::mt19937 random_generator(random_seed);

    typedef boost::variate_generator<boost::mt19937&, boost::uniform_int<> > uniform_generator_t;

    // the distribution boundaries are inclusive
    boost::uniform_int<>
            x_distribution(0, (modelWidth) - minWidth),
            y_distribution(0, (modelHeight) - minHeight),
            channel_distribution(0, numChannels - 1),
            width_distribution(minWidth, maxWidth - 1),
            height_distribution(minHeight, maxHeight - 1);

    if((x_distribution.max() <= 0) or
            (y_distribution.max() <= 0) or
            (width_distribution.max() <= 0) or
            (height_distribution.max() <= 0))
    {
        printf("shrinked model (width, height) == (%i, %i)\n", modelWidth, modelHeight);
        printf("min feature size (after shrinking) (width, height) == (%i, %i)\n", minWidth, minHeight);
        throw invalid_argument("It seems that minFeatWidth or minFeatHeight is bigger than the model size after shrinking");
    }

    uniform_generator_t
            x_generator(random_generator, x_distribution),
            y_generator(random_generator, y_distribution),
            channel_generator(random_generator, channel_distribution),
            width_generator(random_generator, width_distribution),
            height_generator(random_generator, height_distribution);

    size_t total_numOfFeatures = featuresConfigurations.size() + numOfFeatures;
    featuresConfigurations.reserve(total_numOfFeatures);

    int rejectionsInARow = 0, repetitionsCounter = 0;
    const int maxRejectionsInARow = 1000; // how many continuous rejection do we accept ?

    while(featuresConfigurations.size() < total_numOfFeatures)
    {
        const int
                x = x_generator(),
                y = y_generator(),
                c = channel_generator(),
                w = width_generator();
        int h = height_generator();
        if (randomSquares){

            h = w;
        }
        //std::cout << x << " ";
        if(((x + w) <= modelWidth) and ((y + h) <= modelHeight))
        {
            Feature featureConfiguration(x, y, w, h, c);

            // we check if the feature already exists in the set or not
            const bool featureAlreadyInSet =
                    std::find(featuresConfigurations.begin(), featuresConfigurations.end(),
                              featureConfiguration) != featuresConfigurations.end();

            if(featureAlreadyInSet)
            {
                rejectionsInARow += 1;
                repetitionsCounter += 1;
                if(rejectionsInARow > maxRejectionsInARow)
                {
                    printf("once featuresPool reached size %zi, failed to find a new feature after %i attempts\n",
                           featuresConfigurations.size(), maxRejectionsInARow);
                    throw std::runtime_error("Failed to generate the requested features pool, is featuresPoolSize too big?");
                }
                continue;
            }
            else
            {
                rejectionsInARow = 0;
                featuresConfigurations.push_back(featureConfiguration);
            }
        } // end of "if the random feature has proper size"
    } // end of "while not enough features computed"

    if(true)
    {
        printf("When sampling %i features, randomly found (and rejected) %i repetitions\n",
               numOfFeatures, repetitionsCounter);
    }

    return;
}


void getRectRatioTwo(const point_t &modelWindow,
                     const int numChannels,
                     Features &featuresConfigurations)
{
    const int
            shrinking_factor = bootstrapping::integral_channels_computer_t::get_shrinking_factor(),
            modelWidth = modelWindow.x() / shrinking_factor,
            modelHeight = modelWindow.y() / shrinking_factor;

    for (int channel = 0; channel < numChannels; channel +=1)
    {
        for (int col = 0; col < modelWidth; col += 1)
        {
            for (int row = 0; row < modelHeight; row += 1)
            {
                for (int featureWidth = 1; (col + featureWidth) <= modelWidth; featureWidth += 1)
                {
                    const int featureHeight = featureWidth *2;
                    if ((row+featureHeight) <= modelHeight)
                    {
                        Feature featureConfiguration(col, row, featureWidth, featureHeight, channel);
                        featuresConfigurations.push_back(featureConfiguration);
                    }
                } // end of "for each feature width"
                for (int featureHeight = 1; (row + featureHeight) <= modelHeight; featureHeight += 1)
                {
                    const int featureWidth = featureHeight *2;
                    if ((col+featureWidth) <= modelWidth)
                    {
                        Feature featureConfiguration(col, row, featureWidth, featureHeight, channel);
                        featuresConfigurations.push_back(featureConfiguration);
                    }
                } // end of "for each feature height"

            } // end of "for each row"
        } // end of "for each column"
    } // end of "for each channel"

    return;
}


void getAllFeatures(const point_t &modelWindow,
                    const int numChannels,
                    Features &featuresConfigurations)
{
    const int
            shrinking_factor = bootstrapping::integral_channels_computer_t::get_shrinking_factor(),
            modelWidth = modelWindow.x() / shrinking_factor,
            modelHeight = modelWindow.y() / shrinking_factor;

    for (int channel = 0; channel < numChannels; channel +=1)
    {
        for (int col = 0; col < modelWidth; col += 1)
        {
            for (int row = 0; row < modelHeight; row += 1)
            {
                for (int featureWidth = 1; (col + featureWidth) <= modelWidth; featureWidth += 1)
                {
                    for (int featureHeight = 1; (row + featureHeight) <= modelHeight; featureHeight += 1)
                    {
                        Feature featureConfiguration(col, row, featureWidth, featureHeight, channel);
                        featuresConfigurations.push_back(featureConfiguration);
                    } // end of "for each feature height"

                } // end of "for each feature width"

            } // end of "for each row"
        } // end of "for each column"
    } // end of "for each channel"

    return;
}


} // end of namespace boosted_learning
