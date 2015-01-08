#ifndef BOOSTED_LEARNING_IMAGEDATA_HPP
#define BOOSTED_LEARNING_IMAGEDATA_HPP

#include <string>
#include <boost/cstdint.hpp>

namespace boosted_learning {

class ImageData
{
public:
    typedef boost::int8_t objectClassLabel_t;
    std::string filename;
    objectClassLabel_t imageClass;
    int x;
    int y;
    float scale;
};


} // namespace boosted_learning

#endif // BOOSTED_LEARNING_IMAGEDATA_HPP
