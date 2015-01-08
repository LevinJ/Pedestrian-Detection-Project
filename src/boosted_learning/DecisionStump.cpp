#include "DecisionStump.hpp"

namespace boosted_learning
{

DecisionStump::DecisionStump():
_feature(Feature(-1,-1,-1,-1,-1)), _feature_index(-1), _threshold(0), _alpha(1)     
{
    return;
}


DecisionStump::DecisionStump(const Feature &feature, const size_t feature_index, const int threshold, const int alpha):
_feature(feature), _feature_index(feature_index), _threshold(threshold), _alpha(alpha)
{
    return;
}


void DecisionStump::setAlpha(int alpha)
{
    _alpha = alpha;
    return;
}


int DecisionStump::getAlpha() const
{
    return _alpha;
}


int DecisionStump::get_feature_response(const integral_channels_t &integral_image) const
{
    const int
            a = integral_image[_feature.channel][_feature.y                ][_feature.x               ],
            b = integral_image[_feature.channel][_feature.y                ][_feature.x+_feature.width],
            c = integral_image[_feature.channel][_feature.y+_feature.height][_feature.x+_feature.width],
            d = integral_image[_feature.channel][_feature.y+_feature.height][_feature.x               ];
    return a + c - b - d;
}


int DecisionStump::apply_stump(const int feature_response) const
{
    return (int)(feature_response < _threshold);
}


void DecisionStump::print(std::string prefix) const
{
    std::cout << prefix.c_str() << ".feature.channel_index " << _feature.channel << std::endl;
    std::cout << prefix.c_str() << ".feature.box) " << _feature.width * _feature.height << std::endl;
    std::cout << prefix.c_str() <<  ".feature.box min_x " <<  _feature.x  << std::endl;
    std::cout << prefix.c_str() << ".feature.box min_y " <<  _feature.y << std::endl;
    std::cout << prefix.c_str() << ".feature.box max_x " <<  _feature.x +  _feature.width << std::endl;
    std::cout << prefix.c_str() << ".feature.box max_y " <<  _feature.y +  _feature.height   << std::endl;
    std::cout << prefix.c_str() << ".feature_threshold " << _threshold << std::endl;
    std::cout << prefix.c_str() << ".larger_than_threshold " << (_alpha==1) << std::endl;

    return;
}

}
