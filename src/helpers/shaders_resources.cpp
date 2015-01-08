#include "shaders_resources.hpp"


#include <boost/filesystem/path.hpp>

#include <cstdio>

// resources embedding trick based on
// http://burtonini.com/blog/computers/ld-blobs-2007-07-13-15-50

extern char _binary_klt_affine_tracker_cg_start[];
extern char _binary_klt_affine_tracker_cg_end[];

extern char _binary_klt_detector_build_histpyr_cg_start[];
extern char _binary_klt_detector_build_histpyr_cg_end[];

extern char _binary_klt_detector_nonmax_cg_start[];
extern char _binary_klt_detector_nonmax_cg_end[];


extern char _binary_klt_tracker_cg_start[];
extern char _binary_klt_tracker_cg_end[];

extern char _binary_pyramid_pass1h_cg_start[];
extern char _binary_pyramid_pass1h_cg_end[];


extern char _binary_undistort_parametric_cg_start[];
extern char _binary_undistort_parametric_cg_end[];

extern char _binary_klt_copy_patch_cg_start[];
extern char _binary_klt_copy_patch_cg_end[];

extern char _binary_klt_detector_discriminator_cg_start[];
extern char _binary_klt_detector_discriminator_cg_end[];

extern char _binary_klt_detector_pass1_cg_start[];
extern char _binary_klt_detector_pass1_cg_end[];

extern char _binary_klt_detector_pass2_cg_start[];
extern char _binary_klt_detector_pass2_cg_end[];


extern char _binary_klt_detector_traverse_histpyr_cg_start[];
extern char _binary_klt_detector_traverse_histpyr_cg_end[];

extern char _binary_klt_tracker_with_gain_cg_start[];
extern char _binary_klt_tracker_with_gain_cg_end[];

extern char _binary_pyramid_pass1v_cg_start[];
extern char _binary_pyramid_pass1v_cg_end[];

extern char _binary_pyramid_with_derivative_pass2_cg_start[];
extern char _binary_pyramid_with_derivative_pass2_cg_end[];


extern char _binary_pyramid_with_derivative_pass1v_cg_start[];
extern char _binary_pyramid_with_derivative_pass1v_cg_end[];

extern char _binary_pyramid_with_derivative_pass1h_cg_start[];
extern char _binary_pyramid_with_derivative_pass1h_cg_end[];

extern char _binary_pyramid_pass2_cg_start[];
extern char _binary_pyramid_pass2_cg_end[];


extern char _binary_v3d_klt_detector_discriminator_cg_start[];
extern char _binary_v3d_klt_detector_discriminator_cg_end[];

extern char _binary_v3d_klt_detector_pass1_cg_start[];
extern char _binary_v3d_klt_detector_pass1_cg_end[];

extern char _binary_v3d_klt_detector_pass2_cg_start[];
extern char _binary_v3d_klt_detector_pass2_cg_end[];

extern char _binary_v3d_pyramid_with_derivative_pass1v_cg_start[];
extern char _binary_v3d_pyramid_with_derivative_pass1v_cg_end[];

extern char _binary_v3d_pyramid_with_derivative_pass1h_cg_start[];
extern char _binary_v3d_pyramid_with_derivative_pass1h_cg_end[];

extern char _binary_v3d_pyramid_pass2_cg_start[];
extern char _binary_v3d_pyramid_pass2_cg_end[];


extern char _binary_tvl1_flow_relaxed_compute_uv_cg_start[];
extern char _binary_tvl1_flow_relaxed_compute_uv_cg_end[];

extern char _binary_tvl1_flow_relaxed_update_p_cg_start[];
extern char _binary_tvl1_flow_relaxed_update_p_cg_end[];

extern char _binary_tvl1_flow_relaxed_update_uv_cg_start[];
extern char _binary_tvl1_flow_relaxed_update_uv_cg_end[];

extern char _binary_flow_warp_image_cg_start[];
extern char _binary_flow_warp_image_cg_end[];



std::string retrieve_embedded_resource(const std::string filepath)
{
    const std::string filename = boost::filesystem::path(filepath).filename();

    const bool print_resource_search = true;

    if(print_resource_search)
    {
        printf("Searching for shared resource named: %s", filename.c_str());
    }

    std::string program_file_content;

    if(filename == "klt_affine_tracker.cg")
    {
        program_file_content.assign(_binary_klt_affine_tracker_cg_start, _binary_klt_affine_tracker_cg_end);
    }
    else if(filename == "klt_detector_build_histpyr.cg")
    {
        program_file_content.assign(_binary_klt_detector_build_histpyr_cg_start, _binary_klt_detector_build_histpyr_cg_end);
    }
    else if(filename == "klt_detector_nonmax.cg")
    {
        program_file_content.assign(_binary_klt_detector_nonmax_cg_start, _binary_klt_detector_nonmax_cg_end);
    }
    else if(filename == "klt_tracker.cg")
    {
        program_file_content.assign(_binary_klt_tracker_cg_start, _binary_klt_tracker_cg_end);
    }
    else if(filename == "pyramid_pass1h.cg")
    {
        program_file_content.assign(_binary_pyramid_pass1h_cg_start, _binary_pyramid_pass1h_cg_end);
    }
    else if(filename == "undistort_parametric.cg")
    {
        program_file_content.assign(_binary_undistort_parametric_cg_start, _binary_undistort_parametric_cg_end);
    }
    else if(filename == "klt_copy_patch.cg")
    {
        program_file_content.assign(_binary_klt_copy_patch_cg_start, _binary_klt_copy_patch_cg_end);
    }
    else if(filename == "klt_detector_traverse_histpyr.cg")
    {
        program_file_content.assign(_binary_klt_detector_traverse_histpyr_cg_start, _binary_klt_detector_traverse_histpyr_cg_end);
    }
    else if(filename == "klt_tracker_with_gain.cg")
    {
        program_file_content.assign(_binary_klt_tracker_with_gain_cg_start, _binary_klt_tracker_with_gain_cg_end);
    }
    else if(filename == "pyramid_pass1v.cg")
    {
        program_file_content.assign(_binary_pyramid_pass1v_cg_start, _binary_pyramid_pass1v_cg_end);
    }
    else if(filename == "pyramid_with_derivative_pass2.cg")
    {
        program_file_content.assign(_binary_pyramid_with_derivative_pass2_cg_start, _binary_pyramid_with_derivative_pass2_cg_end);
    }

    else if(filename == "klt_detector_discriminator.cg")
    {
        program_file_content.assign(_binary_klt_detector_discriminator_cg_start, _binary_klt_detector_discriminator_cg_end);
    }
    else if(filename == "klt_detector_pass1.cg")
    {
        program_file_content.assign(_binary_klt_detector_pass1_cg_start, _binary_klt_detector_pass1_cg_end);
    }
    else if(filename == "klt_detector_pass2.cg")
    {
        program_file_content.assign(_binary_klt_detector_pass2_cg_start, _binary_klt_detector_pass2_cg_end);
    }
    else if(filename == "pyramid_with_derivative_pass1h.cg")
    {
        program_file_content.assign(_binary_pyramid_with_derivative_pass1h_cg_start, _binary_pyramid_with_derivative_pass1h_cg_end);
    }
    else if(filename == "pyramid_with_derivative_pass1v.cg")
    {
        program_file_content.assign(_binary_pyramid_with_derivative_pass1v_cg_start, _binary_pyramid_with_derivative_pass1v_cg_end);
    }
    else if(filename == "pyramid_pass2.cg")
    {
        program_file_content.assign(_binary_pyramid_pass2_cg_start, _binary_pyramid_pass2_cg_end);
    }
    else if(filename == "v3d_klt_detector_discriminator.cg")
    {
        program_file_content.assign(_binary_v3d_klt_detector_discriminator_cg_start, _binary_v3d_klt_detector_discriminator_cg_end);
    }
    else if(filename == "v3d_klt_detector_pass1.cg")
    {
        program_file_content.assign(_binary_v3d_klt_detector_pass1_cg_start, _binary_v3d_klt_detector_pass1_cg_end);
    }
    else if(filename == "v3d_klt_detector_pass2.cg")
    {
        program_file_content.assign(_binary_v3d_klt_detector_pass2_cg_start, _binary_v3d_klt_detector_pass2_cg_end);
    }
    else if(filename == "v3d_pyramid_with_derivative_pass1h.cg")
    {
        program_file_content.assign(_binary_v3d_pyramid_with_derivative_pass1h_cg_start, _binary_v3d_pyramid_with_derivative_pass1h_cg_end);
    }
    else if(filename == "v3d_pyramid_with_derivative_pass1v.cg")
    {
        program_file_content.assign(_binary_v3d_pyramid_with_derivative_pass1v_cg_start, _binary_v3d_pyramid_with_derivative_pass1v_cg_end);
    }
    else if(filename == "v3d_pyramid_pass2.cg")
    {
        program_file_content.assign(_binary_v3d_pyramid_pass2_cg_start, _binary_v3d_pyramid_pass2_cg_end);
    }
    else if(filename == "tvl1_flow_relaxed_compute_uv.cg")
    {
        program_file_content.assign(_binary_tvl1_flow_relaxed_compute_uv_cg_start, _binary_tvl1_flow_relaxed_compute_uv_cg_end);
    }
    else if(filename == "tvl1_flow_relaxed_update_p.cg")
    {
        program_file_content.assign(_binary_tvl1_flow_relaxed_update_p_cg_start, _binary_tvl1_flow_relaxed_update_p_cg_end);
    }
    else if(filename == "tvl1_flow_relaxed_update_uv.cg")
    {
        program_file_content.assign(_binary_tvl1_flow_relaxed_update_uv_cg_start, _binary_tvl1_flow_relaxed_update_uv_cg_end);
    }
    else if(filename == "flow_warp_image.cg")
    {
        program_file_content.assign(_binary_flow_warp_image_cg_start, _binary_flow_warp_image_cg_end);
    }


    if(print_resource_search)
    {
        if(program_file_content.empty() == false)
        {
            printf(". Loaded.\n");
        } else {
            printf(". Fail.\n");
        }
    }

    return program_file_content;
}

