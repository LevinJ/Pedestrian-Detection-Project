#include "ObjectsDetectionGui.hpp"

#include "drawing/gil/draw_the_detections.hpp"
#include "drawing/gil/draw_the_tracks.hpp"

#include "ObjectsDetectionApplication.hpp"


#include "video_input/AbstractVideoInput.hpp"
#include "video_input/MetricStereoCamera.hpp"
#include "video_input/MetricCamera.hpp"
#include "video_input/ImagesFromDirectory.hpp"
#include "video_input/VideoFromFiles.hpp"
#include "video_input/preprocessing/CpuPreprocessor.hpp"

#include "applications/stixel_world/StixelWorldGui.hpp"

#include "stereo_matching/stixels/StixelWorldEstimator.hpp"
#include "stereo_matching/stixels/FastStixelWorldEstimator.hpp"
#include "stereo_matching/ground_plane/GroundPlaneEstimator.hpp"
#include "stereo_matching/ground_plane/FastGroundPlaneEstimator.hpp"
#include "stereo_matching/stixels/StixelsEstimator.hpp"
#include "stereo_matching/stixels/FastStixelsEstimator.hpp"
#include "stereo_matching/stixels/StixelsEstimatorWithHeightEstimation.hpp"
#include "stereo_matching/stixels/FastStixelsEstimatorWithHeightEstimation.hpp"

#include "objects_detection/AbstractObjectsDetector.hpp"

#if defined(USE_GPU)
#include "objects_detection/GpuVeryFastIntegralChannelsDetector.hpp"
#endif

#include "objects_tracking/AbstractObjectsTracker.hpp"
#include "objects_tracking/DummyObjectsTracker.hpp"

#include "helpers/data/DataSequence.hpp"
#include "objects_detection/detections.pb.h"

#include "helpers/Log.hpp"
#include "helpers/get_option_value.hpp"
#include "helpers/for_each.hpp"
#include "helpers/xyz_indices.hpp"

#include <boost/format.hpp>
#include <boost/foreach.hpp>
#include <boost/bind.hpp>

#include <boost/gil/extension/io/png_io.hpp>
#include <boost/cstdint.hpp>

#include <opencv2/core/core.hpp>
#include "boost/gil/extension/opencv/ipl_image_wrapper.hpp"

#include "drawing/gil/line.hpp"
#include "drawing/gil/colors.hpp"
#include "drawing/gil/draw_ground_line.hpp"
#include "drawing/gil/draw_horizon_line.hpp"
#include "drawing/gil/draw_matrix.hpp"
#include "drawing/gil/hsv_to_rgb.hpp"

#include <SDL/SDL.h>

#include "cudatemplates/copy.hpp"
#include "cudatemplates/hostmemoryheap.hpp"

#include <limits>

namespace
{

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "ObjectsDetectionGui");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "ObjectsDetectionGui");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "ObjectsDetectionGui");
}

} // end of anonymous namespace


namespace doppia
{

using namespace boost::gil;


program_options::options_description ObjectsDetectionGui::get_args_options()
{
    program_options::options_description desc("ObjectsDetectionGui options");

    // specific options --
    desc.add_options()

            //("gui.colorize_disparity",
            // program_options::value<bool>()->default_value(true),
            // "colorize the disparity map or draw it using grayscale")

            ("gui.ground_truth_path",
             program_options::value<string>()->default_value(std::string()),
             "path to protobuf data sequence containing the ground truth")

            ;

    // add base options --
    BaseSdlGui::add_args_options(desc);

    desc.add( StixelWorldGui::get_args_options(false) );
    return desc;
}


ObjectsDetectionGui::ObjectsDetectionGui(ObjectsDetectionApplication &application_,
                                         const program_options::variables_map &options)
    : StixelWorldGui(application_, application_.video_input_p,
                     application_.stixel_world_estimator_p, shared_ptr<AbstractStixelMotionEstimator>(),
                     options, false),
      application(application_)
{

    //if(application.video_input_p.get() == NULL
    //        or application.stixel_world_estimator_p.get() == NULL)
    //{
    //    throw std::runtime_error("ObjectsDetectionGui constructor expects that video_input and stixel_world_estimator_p are already initialized");
    //}

    // retrieve program options --
    //max_disparity = get_option_value<int>(options, "max_disparity");
    max_disparity = 128;

    max_detection_score = 0.001; // detection score is usually 1~3

    // create the application window --
    {
        AbstractVideoInput::input_image_view_t input_view;

        if(application.video_input_p)
        {
            input_view = application.video_input_p->get_left_image();
        }
        else if(application.directory_input_p)
        {
            input_view = application.directory_input_p->get_image();
        }
        else
        {
            throw std::runtime_error("ObjectsDetectionGui constructor expects video_input or directory_input to be already initialized");
        }

        const int input_width = input_view.width();
        const int input_height = input_view.height();

        BaseSdlGui::init_gui(application.get_application_title(), input_width, input_height);
    }


    const string ground_truth_path = get_option_value<string>(options, "gui.ground_truth_path");
    if(not ground_truth_path.empty())
    {
        // we read the datasequence
        ground_truth_data_sequence_p.reset(new DetectionsDataSequence(ground_truth_path));
    }

    // populate the views map --
    views_map[SDLK_1] = view_t(boost::bind(&ObjectsDetectionGui::draw_video_input, this), "draw_video_input");
    views_map[SDLK_2] = view_t(boost::bind(&ObjectsDetectionGui::draw_detections, this), "draw_detections");
    views_map[SDLK_3] = view_t(boost::bind(&ObjectsDetectionGui::draw_tracks, this), "draw_tracks");
    views_map[SDLK_4] = view_t(boost::bind(&ObjectsDetectionGui::draw_stixel_world, this), "draw_stixel_world");
    views_map[SDLK_5] = view_t(boost::bind(&ObjectsDetectionGui::draw_ground_plane_estimation, this), "draw_ground_plane_estimation");
    views_map[SDLK_6] = view_t(boost::bind(&ObjectsDetectionGui::draw_stixels_estimation, this), "draw_stixels_estimation");
    views_map[SDLK_7] = view_t(boost::bind(&ObjectsDetectionGui::draw_stixels_height_estimation, this), "draw_stixels_height_estimation");
    views_map[SDLK_8] = view_t(boost::bind(&ObjectsDetectionGui::draw_disparity_map, this), "draw_disparity_map");
    //views_map[SDLK_9] = view_t(boost::bind(&ObjectsDetectionGui::draw_optical_flow, this), "draw_optical_flow");

#if defined(USE_GPU)
    views_map[SDLK_9] = view_t(boost::bind(&ObjectsDetectionGui::draw_gpu_stixel_world, this), "draw_gpu_stixel_world");
#endif

    // draw the first image --
    draw_video_input();

    // set the initial view mode --
    current_view = views_map[SDLK_1]; // video input

    if(application_.objects_detector_p)
    {
        current_view = views_map[SDLK_2]; // draw detections
    }

    if(application_.stixel_world_estimator_p)
    {
        current_view = views_map[SDLK_4]; // draw stixels world
    }

    if(application_.objects_tracker_p)
    {
        current_view = views_map[SDLK_3]; // draw tracks
    }

    return;
}


ObjectsDetectionGui::~ObjectsDetectionGui()
{
    // nothing to do here
    return;
}


void map_to_rectified_image(
        const CpuPreprocessor &preprocessor,
        const ObjectsDetectionGui::detection_t::rectangle_t &box,
        ObjectsDetectionGui::detection_t::rectangle_t &rectified_box)
{
    const int left_camera_index = 1; // FIXME there is a terrible bug somewhere below !!!

    const point2<int>
            top_left = point2<int>(box.min_corner().x(), box.min_corner().y()),
            bottom_right = point2<int>(box.max_corner().x(), box.max_corner().y()),
            top_right = point2<int>(bottom_right.x, top_left.y),
            bottom_left = point2<int>(top_left.x, bottom_right.y);

    const point2<float>
            rectified_top_left = preprocessor.compute_warping(top_left, left_camera_index),
            rectified_bottom_right = preprocessor.compute_warping(bottom_right, left_camera_index),
            rectified_top_right = preprocessor.compute_warping(top_right, left_camera_index),
            rectified_bottom_left = preprocessor.compute_warping(bottom_left, left_camera_index);

    // we compensate a little bit the rectification distortion by averaging
    // the height of the corners,
    // the horizontal dimension is not modified
    const float
            rectified_left_x = (rectified_top_left.x + rectified_bottom_left.x) / 2.0,
            rectified_right_x = (rectified_top_right.x + rectified_bottom_right.x) / 2.0,
            rectified_top_y = (rectified_top_left.y + rectified_top_right.y) / 2.0,
            rectified_bottom_y = (rectified_bottom_left.y + rectified_bottom_right.y) / 2.0;

    rectified_box.min_corner().x(rectified_left_x);
    rectified_box.min_corner().y(rectified_top_y);

    rectified_box.max_corner().x(rectified_right_x);
    rectified_box.max_corner().y(rectified_bottom_y);


    return;
}


void normalized_box_width(ObjectsDetectionGui::detection_t::rectangle_t &box)
{
    const float
            height = box.max_corner().y() - box.min_corner().y(),
            width = height * 0.4, // See P. Dollar 2011 evaluation paper
            center_x = (box.max_corner().x() + box.min_corner().x())/2.0,
            center_y = (box.max_corner().y() + box.min_corner().y())/2.0;

    box.min_corner().x(center_x - width/2.0);
    box.min_corner().y(center_y - height/2.0);

    box.max_corner().x(center_x + width/2.0);
    box.max_corner().y(center_y + height/2.0);
    return;
}


void map_to_rectified_image(
        const CpuPreprocessor &preprocessor,
        const ::doppia_protobuf::Detections &detections,
        ObjectsDetectionGui::detections_t &rectified_detections)
{
    typedef ObjectsDetectionGui::detection_t detection_t;
    typedef detection_t::rectangle_t rectangle_t;

    rectified_detections.clear();
    for(int i=0; i < detections.detections_size(); i+=1)
    {
        const ::doppia_protobuf::Detection &detection = detections.detections(i);

        detection_t rectified_detection;

        rectified_detection.score = 0;
        //rectified_detection.object_class = detection.object_class();
        rectified_detection.object_class = Detection2d::Pedestrian; // FIXME hardcoded value

        const doppia_protobuf::Box &detection_box = detection.bounding_box();

        const int image_width = 640, image_height = 480; // FIXME HARDCODED PARAMETERS
        detection_t::rectangle_t box;
        box.min_corner().x(std::max(0, detection_box.min_corner().x()));
        box.min_corner().y(std::max(0, detection_box.min_corner().y()));
        box.max_corner().x(std::min(image_width - 1, detection_box.max_corner().x()));
        box.max_corner().y(std::min(image_height - 1, detection_box.max_corner().y()));

        map_to_rectified_image(preprocessor, box, rectified_detection.bounding_box);

        normalized_box_width(rectified_detection.bounding_box);

        rectified_detections.push_back(rectified_detection);
    } // end of "for each protocol buffer detection"

    return;
}

bool ObjectsDetectionGui::process_inputs()
{
    const bool end_of_game = StixelWorldGui::process_inputs();
    // process_inputs is called for every frame
    // we also use it to read input data

    if(ground_truth_data_sequence_p)
    {
        DetectionsDataSequence::data_type ground_truth_detections_pb;
        // FIXME are we ok frame-wise or we are one frame off ?
        ground_truth_data_sequence_p->read(ground_truth_detections_pb);


        VideoFromFiles *video_from_files_p = dynamic_cast<VideoFromFiles *>(application.video_input_p.get());
        if(video_from_files_p and video_from_files_p->get_preprocessor())
        {
            const CpuPreprocessor *preprocessor_p =  dynamic_cast<CpuPreprocessor *>(video_from_files_p->get_preprocessor().get());

            if(preprocessor_p)
            {
                map_to_rectified_image(*preprocessor_p, ground_truth_detections_pb, ground_truth_detections);
            }
        }
    }

    return end_of_game;
}


void ObjectsDetectionGui::resize_if_necessary()
{
    if(application.video_input_p)
    {
        // video_input_p should have a stable input size
        return;
    }

    if(application.directory_input_p == false)
    {
        throw std::runtime_error("ObjectsDetectionGui::resize_if_necessary could not find an image input");
    }

    const ImagesFromDirectory::input_image_view_t::point_t &input_dimensions = \
            application.directory_input_p->get_image().dimensions();

    if((screen_left_view.width() < input_dimensions.x) or
            (screen_left_view.height() < input_dimensions.y))
    {

        const int
                new_width = std::max(screen_left_view.width(), input_dimensions.x),
                new_height = std::max(screen_left_view.height(), input_dimensions.y);

        resize_gui(new_width, new_height);
    }

    return;
}

// FIXME should this be part of the BaseSdlGui ? (and video_input_p ?)
void ObjectsDetectionGui::draw_video_input()
{

    resize_if_necessary();

    if(application.video_input_p)
    {
        const AbstractVideoInput::input_image_view_t &left_input_view = application.video_input_p->get_left_image();
        const AbstractVideoInput::input_image_view_t &right_input_view = application.video_input_p->get_right_image();

        copy_and_convert_pixels(left_input_view, screen_left_view);
        copy_and_convert_pixels(right_input_view, screen_right_view);
    }
    else if(application.directory_input_p)
    {
        const AbstractVideoInput::input_image_view_t &left_input_view = application.directory_input_p->get_image();
        // screen view may be bigger than input view

        // fill in black
        boost::gil::fill_pixels(this->screen_image_view, rgb8_colors::black);

        // copy on the upper left corner
        boost::gil::rgb8_view_t screen_left_subview =
                boost::gil::subimage_view(screen_left_view, 0, 0, left_input_view.width(), left_input_view.height());

        copy_and_convert_pixels(left_input_view, screen_left_subview);
    }

    return;
}


void ObjectsDetectionGui::draw_detections()
{
    this->draw_video_input();

    draw_the_detections(application.objects_detector_p->get_detections(),
                        ground_truth_detections,
                        max_detection_score, application.additional_border,
                        screen_left_view);
    return;
}



void draw_u_disparity_cost_threshold(AbstractStixelWorldEstimator *stixel_world_estimator_p,
                                     const boost::gil::rgb8_view_t &view)
{

    FastStixelWorldEstimator *the_fast_stixel_world_estimator_p = \
            dynamic_cast< FastStixelWorldEstimator *>(stixel_world_estimator_p);

    if(the_fast_stixel_world_estimator_p != NULL)
    {

        const StixelsEstimator *stixels_estimator_p = \
                dynamic_cast<StixelsEstimator *>(the_fast_stixel_world_estimator_p->stixels_estimator_p.get());

        // draw left screen ---
        if(stixels_estimator_p)
        {

            // cost on top -
            const StixelsEstimator::u_disparity_cost_t &u_disparity_cost =
                    stixels_estimator_p->get_u_disparity_cost();
            StixelsEstimator::u_disparity_cost_t filtered_u_disparity_cost = u_disparity_cost;


            const std::vector<int> &ground_plane_corridor = the_fast_stixel_world_estimator_p->get_ground_plane_corridor();
            const std::vector<int> &v_given_disparity = stixels_estimator_p->get_v_given_disparity();

            //const float cost_threshold = 25;
            const float cost_threshold = 35;
            for(int u=0; u < u_disparity_cost.cols(); u+=1)
            {
                for(int disparity=0; disparity < u_disparity_cost.rows(); disparity+=1)
                {
                    float filtered_cost = 0;
                    const int v = v_given_disparity[disparity];
                    const int top_v = ground_plane_corridor[v];

                    if(top_v > 0)
                    {
                        const int
                                //delta_d = 2,
                                delta_d = 5,
                                //delta_d = 10, // 10 is better than 2, 20 is like 10
                                //delta_d = 50, // 50 is like 10
                                min_d = std::max<int>(0, disparity - delta_d),
                                max_d = std::min<int>(disparity + delta_d, u_disparity_cost.cols() - 1);

                        const int
                                object_height = v - top_v,
                                object_width = object_height*0.4,
                                min_u = std::max<int>(0, u - object_width/2),
                                max_u = std::min<int>(u + object_width/2, u_disparity_cost.cols());

                        for(int uu=min_u; uu < max_u; uu+=1)
                        {
                            const bool search_around = true;
                            if(search_around)
                            {
                                float minimum_local_cost = std::numeric_limits<float>::max();
                                for(int d=min_d; d <= max_d; d+=1)
                                {
                                    const float t_cost = u_disparity_cost(d, u);
                                    minimum_local_cost = std::min(minimum_local_cost, t_cost);
                                }
                                filtered_cost += minimum_local_cost;
                            }
                            else
                            {
                                filtered_cost += u_disparity_cost(disparity, u);
                            }

                        } // end of "for each column in the detection window"

                        filtered_cost /= (max_u - min_u); // we average the cost

                        filtered_u_disparity_cost(disparity, u) = filtered_cost;
                    }
                    else
                    {
                        filtered_cost = u_disparity_cost(disparity, u);

                    }//  end of "if valid object"

                    if( filtered_cost > cost_threshold )
                    {
                        filtered_u_disparity_cost(disparity, u) = 0;
                    }
                    else
                    {
                        view(u,v) = rgb8_colors::red;
                    }
                } // end of "for each column"
            } // end of "for each row"


            boost::gil::rgb8_view_t left_top_sub_view =
                    boost::gil::subimage_view(view,
                                              0, 0,
                                              u_disparity_cost.cols(), u_disparity_cost.rows());

            //draw_matrix(u_disparity_cost, left_top_sub_view);
            draw_matrix(filtered_u_disparity_cost, left_top_sub_view);


        } // end of draw left screen -

    }
    else
    { // the_stixel_world_estimator_p == NULL
        // simply freeze the screen
    }

    return;
}


void ObjectsDetectionGui::draw_tracks()
{

    { // copy input images as background
        //this->draw_video_input();
        const AbstractVideoInput::input_image_view_t
                &left_input_view = application.video_input_p->get_left_image();

        //copy_and_convert_pixels(left_input_view, screen_left_view);
        StixelWorldGui::draw_stixel_world(); // will draw left and right screen views
        //StixelWorldGui::draw_stixels_estimation();

        // we overwrite the right image
        copy_and_convert_pixels(left_input_view, screen_right_view);
    }


    if(application.objects_tracker_p)
    {
        const DummyObjectsTracker *dummy_objects_tracker_p = \
                dynamic_cast<const DummyObjectsTracker *>(application.objects_tracker_p.get());


        //AbstractVideoInput &video_input = *(application.video_input_p);

        if(dummy_objects_tracker_p != NULL)
        {
            const DummyObjectsTracker::tracks_t &tracks = dummy_objects_tracker_p->get_tracks();
            draw_the_tracks(tracks,
                            max_detection_score, application.additional_border,
                            track_id_to_hue,
                            screen_left_view);

            // draw ground truth
            for(size_t i=0; i < ground_truth_detections.size(); i+=1)
            {
                gil::rgb8c_pixel_t color = rgb8_colors::white;
                const detection_t::rectangle_t &box = ground_truth_detections[i].bounding_box;
                draw_rectangle(screen_left_view, color, box, 1);
            }

        }
        else
        {
            draw_the_detections(application.objects_tracker_p->get_current_detections(),
                                ground_truth_detections,
                                max_detection_score, application.additional_border,
                                screen_left_view);
        }

        // on the right screen we show the raw detections
        draw_the_detections(application.objects_tracker_p->get_current_detections(),
                            ground_truth_detections,
                            max_detection_score, application.additional_border,
                            screen_right_view);
    }
    else
    {
        // we show nothing
    }


    // draw the threshold regions
    const bool do_draw_u_disparity_cost_threshold = false;
    if(do_draw_u_disparity_cost_threshold)
    {
        draw_u_disparity_cost_threshold(
                    stixel_world_estimator_p.get(),
                    screen_left_view);
    } // end of "if draw_u_disparity_cost_threshold"

    return;
}


void ObjectsDetectionGui::draw_stixel_world()
{
    StixelWorldGui::draw_stixel_world();

    // draw the detections on top of the stixel world
    draw_the_detections(application.objects_detector_p->get_detections(),
                        ground_truth_detections,
                        max_detection_score, application.additional_border,
                        screen_left_view);
    return;
}

#if defined(USE_GPU)
void ObjectsDetectionGui::draw_gpu_stixel_world()
{
    StixelWorldGui::draw_stixel_world();

    // draw the detections on top of the stixel world
    draw_the_detections(application.objects_detector_p->get_detections(),
                        ground_truth_detections,
                        max_detection_score, application.additional_border,
                        screen_left_view);

    GpuVeryFastIntegralChannelsDetector* gpu_detector_p = \
            dynamic_cast<GpuVeryFastIntegralChannelsDetector*>(application.objects_detector_p.get());

    if(gpu_detector_p)
    {
        Cuda::HostMemoryHeap1D<doppia::objects_detection::gpu_stixel_t> cpu_stixels(gpu_detector_p->gpu_stixels.getNumElements());

        Cuda::copy(cpu_stixels, gpu_detector_p->gpu_stixels);

        printf("cpu_stixels.size %zi\n", cpu_stixels.getNumElements());

        boost::gil::rgb8_view_t &view = screen_left_view;

        const int
                // shrinking_factor is 1, 2 or 4
                shrinking_factor = IntegralChannelsForPedestrians::get_shrinking_factor();

        for(size_t gpu_stixel_index=0; gpu_stixel_index < cpu_stixels.getNumElements(); gpu_stixel_index +=1)
        {

            doppia::objects_detection::gpu_stixel_t &stixel = cpu_stixels[gpu_stixel_index];
            const int
                    u = gpu_stixel_index * shrinking_factor,
                    min_y = stixel.min_y * shrinking_factor,
                    max_y = stixel.max_y * shrinking_factor;

            if(min_y > max_y)
            {
                printf("gpu_stixel_index == %zi, min_y %i, max_y %i\n",
                       gpu_stixel_index, min_y, max_y);
                throw std::runtime_error("ObjectsDetectionGui::draw_gpu_stixel_world says oooups");
            }

            for(int i=0; i < shrinking_factor; i+=1)
            {
                view(u+i, min_y) = rgb8_colors::green;
                view(u+i, max_y) = rgb8_colors::dark_green;
            }
        } // end of "for each gpu stixel"
    } // end of "if gpu very fast detector"


    return;
}
#endif

} // end of namespace doppia



