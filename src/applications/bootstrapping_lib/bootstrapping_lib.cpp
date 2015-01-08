#include "bootstrapping_lib.hpp"

#include "objects_detection/ObjectsDetectorFactory.hpp"
#include "objects_detection/non_maximal_suppression/GreedyNonMaximalSuppression.hpp"


#if defined(USE_GPU)
#include "cudatemplates/error.hpp"
#include "objects_detection/GpuIntegralChannelsDetector.hpp"
#else
#include "objects_detection/IntegralChannelsDetector.hpp"
#endif

#include "objects_detection/SoftCascadeOverIntegralChannelsModel.hpp"

#include "video_input/ImagesFromDirectory.hpp" // for the open_image helper method
//#include "objects_detection/integral_channels/IntegralChannelsComputerFactory.hpp"
#include "video_input/preprocessing/AddBorderFunctor.hpp"

#include "helpers/add_to_parsed_options.hpp"
#include "helpers/get_option_value.hpp"

#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/progress.hpp>
#include <boost/tuple/tuple.hpp>

#include <algorithm>
#include <limits>
#include <cstdio>

namespace {

using namespace std;
using boost::filesystem::path;

class ImagesFromList
{
public:

    typedef boost::gil::rgb8_image_t input_image_t;
    typedef boost::gil::rgb8c_view_t input_image_view_t;
    typedef input_image_view_t::point_t dimensions_t;

    ImagesFromList(const vector<std::string> &files_list);
    ~ImagesFromList();

    bool next_frame();
    const input_image_view_t &get_image() const;
    const string &get_image_name() const;
    const path get_image_path() const;

protected:

    const vector<std::string> files_list;
    vector<std::string>::const_iterator the_files_iterator;

    input_image_t input_image;
    input_image_view_t input_image_view;
    string input_image_name;
    path input_image_path;
};



ImagesFromList::ImagesFromList(const vector<std::string> &files_list_)
    : files_list(files_list_)
{
    the_files_iterator = files_list.begin();
    return;
}


ImagesFromList::~ImagesFromList()
{
    // nothing to do here
    return;
}


bool ImagesFromList::next_frame()
{
    if(the_files_iterator == files_list.end())
    {
        return false;
    }

    // set the image name
    input_image_path = path(*the_files_iterator);
#if BOOST_VERSION <= 104400
    input_image_name = input_image_path.filename();
#else
    input_image_name = input_image_path.filename().string();
#endif

    // read the image, set the image view
    input_image_view = doppia::open_image(input_image_path.string(), input_image);

    //log_debug() << "read file " << image_path << std::endl;

    // move iterator to next image
    ++the_files_iterator;
    return true;
}


const ImagesFromList::input_image_view_t &ImagesFromList::get_image() const
{
    return input_image_view;
}


const string &ImagesFromList::get_image_name() const
{
    return input_image_name;
}


const path ImagesFromList::get_image_path() const
{
    return input_image_path;
}


} // end of anonymous namespace

namespace bootstrapping
{

using namespace std;
using namespace doppia;
using namespace boost;
using boost::filesystem::path;

typedef doppia::IntegralChannelsDetector::detections_t detections_t;
typedef doppia::IntegralChannelsDetector::detection_t detection_t;
typedef detection_t::rectangle_t box_t;


void get_integral_channels_subimage(
        const box_t &box,
        const integral_channels_t &integral_channels,
        integral_channels_t &out )
{
    const size_t
            h = box.max_corner().y() - box.min_corner().y(),
            w = box.max_corner().x() - box.min_corner().x();

    const size_t num_channels = integral_channels.shape()[0];
    out.resize(boost::extents[num_channels][h+1][w+1]);
    for(size_t ch=0; ch<num_channels; ch+=1)
    {
        for(int y=box.min_corner().y(); y< box.max_corner().y() + 1; y+=1)
        {
            //std::copy(&(integral_channels[ch][yy][x]), &(integral_channels[ch][yy][x+w+1]), out[ch][yy-y].begin());
            for(int x=box.min_corner().x(); x < box.max_corner().x() + 1; x += 1)
            {
                const int
                        y_index = y - box.min_corner().y(),
                        x_index = x - box.min_corner().x();

                out[ch][y_index][x_index] = integral_channels[ch][y][x];
            } // end of "for each column"
        }// end of "for each row"

    } // end of "for each channel"

    return;
}


#if defined(USE_GPU)
typedef GpuIntegralChannelsDetector FalsePositivesDataCollectorBaseClass;
#else
typedef IntegralChannelsDetector FalsePositivesDataCollectorBaseClass;
#endif


class FalsePositivesDataCollector: public FalsePositivesDataCollectorBaseClass
{

public:
    FalsePositivesDataCollector(
            const size_t max_false_positives_,
            const size_t max_false_positives_per_image_,
            append_result_functor_t &append_result_functor_,
            const boost::program_options::variables_map &options,
            boost::shared_ptr<SoftCascadeOverIntegralChannelsModel> cascade_model_p,
            boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
            const float score_threshold,
            const int additional_border);
    ~FalsePositivesDataCollector();

    void compute();

    path current_image_path;

    size_t num_false_positives_found() const;

protected:
    const size_t max_false_positives, max_false_positives_per_image;

protected:
    append_result_functor_t &append_result_functor;

protected:
    size_t added_false_positives, added_false_positives_on_current_image;
    void collect_false_positive_data(const detection_t &detection);
};


FalsePositivesDataCollector::FalsePositivesDataCollector(
        const size_t max_false_positives_,
        const size_t max_false_positives_per_image_,
        append_result_functor_t &append_result_functor_,
        const boost::program_options::variables_map &options,
        boost::shared_ptr<SoftCascadeOverIntegralChannelsModel> cascade_model_p,
        boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p,
        const float score_threshold,
        const int additional_border)
    : BaseIntegralChannelsDetector(options,
                                   cascade_model_p,
                                   non_maximal_suppression_p, score_threshold, additional_border),
      #if defined(USE_GPU)
      GpuIntegralChannelsDetector(options, cascade_model_p, non_maximal_suppression_p, score_threshold, additional_border),
      #else
      IntegralChannelsDetector(options, cascade_model_p, non_maximal_suppression_p, score_threshold, additional_border),
      #endif
      max_false_positives(max_false_positives_),
      max_false_positives_per_image(max_false_positives_per_image_),
      append_result_functor(append_result_functor_),
      added_false_positives(0),
      added_false_positives_on_current_image(0)
{
    // nothing to do here
    return;
}


FalsePositivesDataCollector::~FalsePositivesDataCollector()
{
    // nothing to do here
    return;
}


/// these tuples store the detection, the non rescaled detection, and the DetectorSearchRange index
typedef boost::tuples::tuple<detection_t, detection_t, size_t> detection_with_search_range_t;
typedef std::vector<detection_with_search_range_t> detections_with_search_range_t;


bool has_higher_score(const detection_with_search_range_t &a, const detection_with_search_range_t &b)
{
    return a.get<0>().score > b.get<0>().score;
}


bool has_higher_search_range_index(const detection_with_search_range_t &a, const detection_with_search_range_t &b)
{
    return a.get<2>() > b.get<2>();
}


/// this code is based on GreedyNonMaximalSuppression::compute
/// just choosing the false positives with strongest score ruins the learning process,
/// instead we select a subset of the detections, such as there is no strong overlap
/// between the final false positives
void select_subset_of_detections(detections_with_search_range_t &detections)
{

    const float minimal_overlap_threshold(0.5);
    // for bootstrapping this parameters is not very sensitive
    // 1/2 seems a reasonable overlap value

    // copy all the candidates
    typedef std::list<detection_with_search_range_t> candidate_detections_t;
    candidate_detections_t candidate_detections;

    // key point, instead of sorting, we simply shuffle
    // random_shuffle only works on std::vector, so we copy, shuffle and then move to a list
    // candidate_detections = candidate_detections_t(detections.begin(), detections.end());
    //candidate_detections.sort(has_higher_score);
    detections_with_search_range_t detections_shuffled(detections.begin(), detections.end()); // copy
    std::random_shuffle(detections_shuffled.begin(), detections_shuffled.end()); // shuffle
    candidate_detections = candidate_detections_t(detections.begin(), detections.end()); // copy to list

    detections_with_search_range_t &maximal_detections = detections; // we will store the result in the input vector
    maximal_detections.clear();
    maximal_detections.reserve(42); // we do not expect more than 42 pedestrians per scene

    candidate_detections_t::iterator detections_it = candidate_detections.begin();
    for(; detections_it != candidate_detections.end(); ++detections_it)
    {
        const detection_with_search_range_t &detection = *detections_it;

        // this detection passed the test
        maximal_detections.push_back(detection);

        candidate_detections_t::iterator lower_score_detection_it = detections_it;
        ++lower_score_detection_it; // = detections_it + 1
        for(; lower_score_detection_it != candidate_detections.end(); )
        {
            const float overlap = compute_overlap(detection.get<0>(), lower_score_detection_it->get<0>());

            if(overlap > minimal_overlap_threshold)
            {
                // this detection seems to overlap too much, we should remove it
                lower_score_detection_it = candidate_detections.erase(lower_score_detection_it);
            }
            else
            {
                // we keep this detection in the candidates list
                ++lower_score_detection_it;
            }

        } // end of "for each lower score detection"


    } // end of "for each candidate detection"

    return;
}


void FalsePositivesDataCollector::compute()
{
    added_false_positives_on_current_image = 0;

    // some debugging variables
    const bool save_score_image = false;
    //const bool save_score_image = true;
    static bool first_call = true;

    assert(integral_channels_computer_p);

    detections_with_search_range_t detections_with_search_range;
    detections_with_search_range.reserve(1000); // rough estimate

    // search for all false positives --
    {
        // for each range search
        //BOOST_FOREACH(const DetectorSearchRange &search_range, this->search_ranges_data)
        for(size_t scale_index = 0; scale_index < search_ranges_data.size(); scale_index += 1)
        {
            // we clear the detections at each range,
            // we store them anyway inside detections_with_search_range
            detections.clear();
            non_rescaled_detections.clear();

#if defined(USE_GPU)
            try
            {
                compute_detections_at_specific_scale_v0(scale_index, save_score_image, first_call);
                // FIXME v1 does not work properly. NEEDS to be checked again (major changes in v1)
                //compute_detections_at_specific_scale_v1(scale_index, first_call);
            }
            catch(::Cuda::Error &e)
            {
                printf("compute_detections_at_specific_scale_v0 failed at %zi, skipping detections at this scale.\n%s\n",
                       scale_index, e.what());
                detections.clear();
                non_rescaled_detections.clear();
            }

#else
            compute_detections_at_specific_scale(scale_index, save_score_image, first_call);
#endif

            if(detections.size() != non_rescaled_detections.size())
            {
                throw std::runtime_error("Something went terribly wrong with the non_rescaled_detections collection");
            }

            for(size_t index = 0; index < detections.size(); index += 1)
            {
                const detection_t
                        &detection = detections[index],
                        &non_rescaled_detection = non_rescaled_detections[index];

                detections_with_search_range.push_back(
                            boost::tuples::make_tuple(detection, non_rescaled_detection, scale_index));
            } // end of "for each detection"

        } // end of "for each search range"

    } // end of "search for all false positives"


    // we apply some kind of non maximal suppresion
    select_subset_of_detections(detections_with_search_range);

    // after non maximal suppresion, the best detections are at the begining of the list
    detections_with_search_range.resize(
                std::min<size_t>(max_false_positives_per_image,
                                 detections_with_search_range.size()));

    // we sort to search range indices, to avoid recomputing the features when not needed
    std::sort(detections_with_search_range.begin(), detections_with_search_range.end(),
              has_higher_search_range_index);

    size_t last_search_range_index = 0;
    if(detections_with_search_range.empty() == false)
    {
        // we need to initialize last_search_range_index such as
        // last_search_range_index != = detections_with_search_range[0]
        last_search_range_index = detections_with_search_range[0].get<2>() + 1;
    }

    // we now recompute the integral channels and retrieve the false positives data
    BOOST_FOREACH(const detection_with_search_range_t &detection_with_search_range,
                  detections_with_search_range)
    {
        const detection_t &non_rescaled_detection = detection_with_search_range.get<1>();
        const size_t search_range_index = detection_with_search_range.get<2>();

        if(last_search_range_index != search_range_index)
        {
            // (we ignore the gpu_integral_channels_t result value)
            //doppia::objects_detection::gpu_integral_channels_t &integral_channels =
            // FIXME this method is not yet implemented in the CPU code (only works on GPU code)
            resize_input_and_compute_integral_channels(search_range_index);
        }
        else
        {
            // no need to recompute the integral channels
        }

        last_search_range_index = search_range_index;
        collect_false_positive_data(non_rescaled_detection);

        if((added_false_positives >= max_false_positives)
                or ((max_false_positives_per_image > 0) and
                    added_false_positives_on_current_image >= max_false_positives_per_image) )
        {
            break;
        }
    }

    first_call = false;
    return;
}


void FalsePositivesDataCollector::collect_false_positive_data(const detection_t &detection)
{
    const integral_channels_t &integral_channels = integral_channels_computer_p->get_integral_channels();

    // set the meta data --
    meta_datum_t meta_datum;

    meta_datum.filename = current_image_path.string();
    meta_datum.imageClass  = -1; // magic number from Markus Mathias code, represents background class label
    meta_datum.x = detection.bounding_box.min_corner().x();
    meta_datum.y = detection.bounding_box.min_corner().y();

    const float image_scale = this->current_image_scale; // == 1 / detection_scale
    meta_datum.scale = image_scale; // this is the image scale

    // set the integral channels data --
    {
        integral_channels_t integral_image;

        // we added inside doppia::collect_the_detections an #ifdef to ensure that the detections are on the correct
        // scale right away
        const box_t &box = detection.bounding_box;

#if not defined(NDEBUG)
        const int expected_window_width= scale_one_detection_window_size.x() / integral_channels_computer_p->get_shrinking_factor();
        if((box.max_corner().x() - box.min_corner().x()) != expected_window_width)
        {
            printf("box.max_corner().x() - box.min_corner().x() == %i\n",
                   box.max_corner().x() - box.min_corner().x());
            throw std::runtime_error("wrong box size");
        }
#endif

        const bool save_last_integral_channels = false; // for debugging only
        if(save_last_integral_channels)
        {
            printf("Collecting data at coordinates %i,%i\n", box.min_corner().x(), box.min_corner().y());

            doppia::save_integral_channels_to_file(integral_channels, "/tmp/integral_channels.png");
            printf("Created /tmp/integral_channels.png so you can debug  FalsePositivesDataCollector::collect_false_positive_data\n");
            //throw std::runtime_error("Created /tmp/integral_channels.png so you can debug  FalsePositivesDataCollector::collect_false_positive_data");
        }

        get_integral_channels_subimage(box, integral_channels, integral_image);
        append_result_functor(meta_datum, integral_image); // store the false positive result
        added_false_positives += 1;
        added_false_positives_on_current_image += 1;
    }


    return;
}


size_t FalsePositivesDataCollector::num_false_positives_found() const
{
    return added_false_positives;
}

// ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

PushBackFunctor::PushBackFunctor(std::vector<meta_datum_t> &meta_data_,
                                 std::vector<integral_channels_t> &integral_images_)
    : meta_data(meta_data_),
      integral_images(integral_images_)
{
    // nothing to do here
    return;
}


PushBackFunctor::~PushBackFunctor()
{
    // nothing to do here
    return;
}


void PushBackFunctor::operator()(const meta_datum_t &meta_datum, const integral_channels_t &integral_image)
{
    meta_data.push_back(meta_datum);
    integral_images.push_back(integral_image);
    return;
}

// ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

/// Given the path to a classifier and set of negative images,
/// and a maximum number of false positive to find, will search for false positives
/// (assuming pedestrians from the INRIA dataset) and fill in the meta_data and integral_images structures.
void bootstrap(const path &classifier_model_filepath,
               const std::vector<std::string> &negative_image_paths_to_explore,
               const size_t max_false_positives,
               const size_t max_false_positives_per_image,
               const float min_scale, const float max_scale, const int num_scales,
               const float min_ratio, const float max_ratio, const int num_ratios,
               const bool use_less_memory,
               std::vector<meta_datum_t> &meta_data,
               std::vector<integral_channels_t> &integral_images,
const boost::program_options::variables_map &options)
{
    append_result_functor_t the_functor = PushBackFunctor(meta_data, integral_images);
    bootstrap(classifier_model_filepath, negative_image_paths_to_explore,
              max_false_positives, max_false_positives_per_image,
              min_scale, max_scale, num_scales,
              min_ratio, max_ratio, num_ratios,
              use_less_memory,
              the_functor,
options);
    return;
}


/// Given the path to a classifier and set of negative images,
/// and a maximum number of false positive to find, will search for false positives
/// (assuming pedestrians from the INRIA dataset) and fill in the meta_data and integral_images structures.
void bootstrap(const path &classifier_model_filepath,
               const std::vector<std::string> &negative_image_paths_to_explore,
               const size_t max_false_positives,
               const size_t max_false_positives_per_image,
               const float min_scale, const float max_scale, const int num_scales,
               const float min_ratio, const float max_ratio, const int num_ratios,
               const bool use_less_memory,
               append_result_functor_t &functor,
const boost::program_options::variables_map &options)
{
    // open the classifier_model_file --
    if(filesystem::exists(classifier_model_filepath) == false)
    {
        printf("Could not find the indicated file %s\n",
               classifier_model_filepath.string().c_str());
        throw std::invalid_argument("Indicated model file does not exist");
    }

    shared_ptr<doppia_protobuf::DetectorModel> detector_model_data_p;
    read_protobuf_model(classifier_model_filepath.string(), detector_model_data_p);
    if(detector_model_data_p == false)
    {
        throw std::runtime_error("Failed to read a model compatible with "
                                 "the IntegralChannelsDetector/FastestPedestrianDetectorInTheWest");
    }


    shared_ptr<SoftCascadeOverIntegralChannelsModel>
            cascade_model_p(new SoftCascadeOverIntegralChannelsModel(*detector_model_data_p));

    float score_threshold = 0;
    if(cascade_model_p->has_soft_cascade())
    {
        score_threshold = cascade_model_p->get_last_cascade_threshold();
    }

    printf("Using score_threshold == %.5f\n", score_threshold);


    shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p;

    program_options::options_description desc;
    //desc.add(AbstractObjectsDetector::get_args_options());
    //desc.add(IntegralChannelsDetector::get_args_options());
    desc.add(ObjectsDetectorFactory::get_args_options()); // may add more than we need, but should be ok
    //desc.add(IntegralChannelsComputerFactory::get_args_options());

    const bool print_options = false;
    if(print_options)
    { // just for debugging
        std::cout << desc << std::endl;
    }

    program_options::parsed_options the_parsed_options(&desc);

    // set the INRIA pedestrians dataset options

    // strides smaller than 1 ensures that will use 1 pixel at all scales
    add_to_parsed_options(the_parsed_options, "objects_detector.x_stride", 1e-6);
    add_to_parsed_options(the_parsed_options, "objects_detector.y_stride", 1e-6);
    add_to_parsed_options(the_parsed_options, "objects_detector.min_scale", min_scale);
    add_to_parsed_options(the_parsed_options, "objects_detector.max_scale", max_scale);
    add_to_parsed_options(the_parsed_options, "objects_detector.num_scales", num_scales);
    add_to_parsed_options(the_parsed_options, "objects_detector.min_ratio", min_ratio);
    add_to_parsed_options(the_parsed_options, "objects_detector.max_ratio", max_ratio);
    add_to_parsed_options(the_parsed_options, "objects_detector.num_ratios", num_ratios);

    //add_to_parsed_options(the_parsed_options, "channels.num_hog_angle_bins", get_option_value<int>(options, "channels.num_hog_angle_bins"));

#if defined(USE_GPU)
    // enable/disable lower memory usage option (traded off for slower speed too)
    add_to_parsed_options(the_parsed_options, "objects_detector.gpu.frugal_memory_usage", use_less_memory);
#endif

    //add_to_parsed_options(the_parsed_options, "additional_border", 80);
    const int additional_border = 80;

    // by setting this to false, we ensure that the  detection windows fit model_window (and not the object_window as usual)
    add_to_parsed_options(the_parsed_options, "objects_detector.resize_detections", false);

    program_options::variables_map the_program_options;
    program_options::store(the_parsed_options, the_program_options);
    program_options::notify(the_program_options);

    FalsePositivesDataCollector chnftrs_detector(
                max_false_positives, max_false_positives_per_image,
                functor,
                the_program_options, cascade_model_p, non_maximal_suppression_p,
                score_threshold,
                additional_border);


    // create the images iterator --
    ImagesFromList images_source(negative_image_paths_to_explore);

    bool video_input_is_available = false;
    video_input_is_available = images_source.next_frame();

    AddBorderFunctor add_border(additional_border);

    size_t false_positives_found = 0;
    int images_visited = 0;

    {
        boost::progress_display progress_bar(max_false_positives);
        // for each input image
        while(video_input_is_available and (false_positives_found < max_false_positives))
        {
            ImagesFromList::input_image_view_t input_view;
            input_view = images_source.get_image();
            input_view = add_border(input_view);

            try
            {
                // FIXME harcoded values
                const size_t
                        expected_channels_size = input_view.size()*10,
                        max_texture_size = 134217728; // 2**27 for CUDA capability 2.x
                if(expected_channels_size > max_texture_size)
                {
                    throw std::invalid_argument("The image is monstruously big!");
                }

                // find the false positives
                chnftrs_detector.set_image(input_view);
                chnftrs_detector.current_image_path = images_source.get_image_path();
                chnftrs_detector.compute();
            }
            catch(std::exception &e)
            {
                printf("Processing of image %s \033[1;31mfailed\033[0m (size %zix%zi). Skipping it. Error was:\n%s\n",
                       images_source.get_image_path().string().c_str(),
                       input_view.width(), input_view.height(),
                       e.what());
            }

            // update the progress_bar
            progress_bar += chnftrs_detector.num_false_positives_found() - false_positives_found;

            false_positives_found = chnftrs_detector.num_false_positives_found();
            images_visited += 1;
            video_input_is_available = images_source.next_frame();
        } // end of "for each input image"

    } // finish the progress_bar

    printf("bootstrapping::bootstrap visited %i images to collect %zi false positives\n",
           images_visited, false_positives_found);
    return;
}


} // end of namespace bootstrapping


