#ifndef __FEAT_CONFIG_H
#define __FEAT_CONFIG_H

#include "ImageData.hpp"

#include "applications/bootstrapping_lib/bootstrapping_lib.hpp"

#include <helpers/get_option_value.hpp>
#include "helpers/geometry.hpp"

#include <boost/multi_array.hpp>
#include <boost/shared_ptr.hpp>
#include <string>
#include <map>
#include <cassert>
#include <vector>
#include <boost/cstdint.hpp>
#include <iosfwd>

namespace boosted_learning {

using namespace std;

/// Simple structure of a feature
class Feature
{
public:
    /// The constructor that set everything to 0.
    Feature();

    /// The constructor that ask for the sizes.
    Feature(const int x, const int y, const int width, const int height, const int channel);

    int x; ///< x position.
    int y; ///< y position.
    int width; ///< width or phi1
    int height; ///< height or phi2
    int channel;

    bool operator==(const Feature &other) const;

    void printConfigurationString(std::ostream &os) const;
    void readConfigurationString(std::istream &is);

    int getResponse(const bootstrapping::integral_channels_t &integralImage) const;
};

typedef doppia::geometry::point_xy<int> point_t;
typedef doppia::geometry::box<point_t> rectangle_t;

typedef std::vector<Feature> Features;
typedef boost::shared_ptr<Features> FeaturesSharedPointer;
typedef boost::shared_ptr<const Features> ConstFeaturesSharedPointer;
//typedef int16_t bintype;
//typedef int bintype;
//typedef size_t bintype;
typedef boost::uint16_t bintype;

//typedef std::vector<int> FeaturesResponses;
typedef boost::multi_array<int, 2> FeaturesResponses;
typedef boost::multi_array<bintype, 2> FeaturesBinResponses;
typedef boost::shared_ptr<FeaturesResponses> FeaturesResponsesSharedPointer;
typedef boost::shared_ptr<const FeaturesResponses> ConstFeaturesResponsesSharedPointer;

typedef std::vector<int> MinOrMaxFeaturesResponses;
typedef boost::shared_ptr<MinOrMaxFeaturesResponses> MinOrMaxFeaturesResponsesSharedPointer;
typedef boost::shared_ptr<const MinOrMaxFeaturesResponses> ConstMinOrMaxFeaturesResponsesSharedPointer;


/// inline for speed reasons
inline int Feature::getResponse(const bootstrapping::integral_channels_t &integralImage) const
{
    const int
            a = integralImage[channel][y][x],
            b = integralImage[channel][y+0][x+width],
            c = integralImage[channel][y+height][x+width],
            d = integralImage[channel][y+height][x+0];
    return a + c - b - d;
}


/// this function gets most of its parameters from the Parameters::getParameter singleton
void getFeaturesConfigurations(const point_t &modelWindow,
                               const int numOfFeatures, const int numChannels,
                               Features &featuresConfigurations,
                               const std::string &configType,
                               const boost::program_options::variables_map &option);

void computeRandomFeaturesConfigurations(const point_t &modelWindow,
                                         const int numOfFeatures, const int numChannels,
                                         Features &featuresConfigurations,
                                         const boost::program_options::variables_map &options,
                                         bool randomSquares = false);

void getHogLikeFeatures(const int cellSize,
                        const point_t &modelWindow,
                        const int numChannels,
                        Features & featuresConfigurations);

void getRectRatioTwo(const point_t &modelWindow,
                     const int numChannels,
                     Features & featuresConfigurations);

void getHogLikeFeaturesGray(const int cellSize,
                            const point_t &modelWindow,
                            Features & featuresConfigurations);

void getHogLikeMultiScaleGray(const point_t &modelWindow, Features & featuresConfigurations);



} // end of namespace boosted_learning

#endif // __FEAT_CONFIG_H
