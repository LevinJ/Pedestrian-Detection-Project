#ifndef OPENCVLinesDetector_HPP
#define OPENCVLinesDetector_HPP

#include "AbstractLinesDetector.hpp"

namespace doppia {

class OpenCvLinesDetector: public AbstractLinesDetector
{
public:
    OpenCvLinesDetector();
    OpenCvLinesDetector(const float input_image_threshold,
                         const float direction_resolution, const float origin_resolution,
                         const int detection_threshold);
    ~OpenCvLinesDetector();

    void operator()(const source_view_t &src, lines_t &lines);

protected:
    /// 0 to 255, input_image pixels below this value are considered as 0
    const float input_image_threshold;
    const float direction_resolution;
    const float origin_resolution;
    const int detection_threshold;
};

} // namespace doppia

#endif // OPENCVLinesDetector_HPP
