#ifndef ABSTRACTOBJECTSDETECTOR_HPP
#define ABSTRACTOBJECTSDETECTOR_HPP

#include "Detection2d.hpp"
#include "DetectorSearchRange.hpp"
#include "ScaleData.hpp"
#include "AbstractModelWindowToObjectWindowConverter.hpp"

#include "stereo_matching/stixels/Stixel.hpp"
#include "helpers/geometry.hpp"

#include <boost/program_options.hpp>
#include <boost/gil/typedefs.hpp>
#include <boost/gil/image_view.hpp>
#include <boost/scoped_ptr.hpp>

#include <vector>

namespace doppia {

class AbstractObjectsDetector
{

public:

    typedef ScaleData::detection_window_size_t detection_window_size_t;

    typedef Detection2d detection_t;
    typedef std::vector<detection_t> detections_t;

    typedef std::vector<int> ground_plane_corridor_t;

public:

    static boost::program_options::options_description get_args_options();

    AbstractObjectsDetector(const boost::program_options::variables_map &options);
    virtual ~AbstractObjectsDetector();

    virtual void set_image(const boost::gil::rgb8c_view_t &input_image,
						   const std::string &image_file_path = std::string()) = 0;
    virtual void set_stixels(const stixels_t &stixels);
    virtual void set_ground_plane_corridor(const ground_plane_corridor_t &corridor);
    virtual void compute() = 0;

    /// overwrites the values computed during set_image
    //virtual void set_search_range(const detector_search_ranges_t &range);

    virtual const detections_t &get_detections();

    /// little backdoor for experimentation only, do not use this method
    const detections_t &get_raw_detections() const;

    /// little backdoor for experimentation only, do not use this method
    void set_raw_detections(const detections_t &detections);

protected:
    detections_t detections;

    /// the scale is the scale of the detection window
    /// for instance scale 0.5 means a detection window of
    /// half the nominal size applied to the input image
    const float min_detection_window_scale, max_detection_window_scale;
    const int num_scales;

    /// ratio defined as width/height
    const float min_detection_window_ratio, max_detection_window_ratio;
    const int num_ratios;

    const float x_stride, y_stride;
    const bool resize_detection_windows;


    boost::scoped_ptr<AbstractModelWindowToObjectWindowConverter> model_window_to_object_window_converter_p;

    /// which are the scales that should be explored ?
    detector_search_ranges_data_t search_ranges_data;

    /// helper method used to set the search_ranges
    void compute_search_ranges_meta_data(detector_search_ranges_data_t &search_ranges_data) const;

    /// obtain the 'final' search range including scale, occlusion, detector size and shrinking factor information
    virtual DetectorSearchRange compute_scaled_search_range(const size_t scale_index) const = 0;

    /// helper class for testing
    friend class DetectorsComparisonTestApplication;
};


} // end of namespace doppia

#endif // ABSTRACTOBJECTSDETECTOR_HPP
