#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function


from stixels_evaluation import StixelsEvaluationApplication, Bunch, Recording

import collections

options = Bunch()

#options.ground_truth_path = "/users/visics/rbenenso/data/bertan_datasets/Zurich/bahnhof/annotations/bahnhof-annot.idl"
options.ground_truth_path = "/home/rodrigob/data/bahnhof/annotations/bahnhof-annot.idl"
#options.ground_truth_path = "/users/visics/rbenenso/data/bertan_datasets/Zurich/bahnhof/annotations/bahnhof-annot-local-filtered.idl"
#options.ground_truth_path = "/users/visics/rbenenso/data/bertan_datasets/Zurich/bahnhof/annotations/Annotations.idl"
 
#options.recordings = dict()

options.recordings = collections.OrderedDict()

base_recordings_path = "/users/visics/rbenenso/code/doppia/src/applications/stixel_world/"
rss2011_recordings_path = "/users/visics/rbenenso/data/2011_rss_paper_results/applications/stixel_world/"
iros2011_recordings_path = "/users/visics/rbenenso/data/2011_iros_paper_results/applications/stixel_world/"
iccv2011_workshop_recordings_path = "/users/visics/rbenenso/data/2011_iccv_workshop_paper_results/applications/stixel_world/"
eccv2012_workshop_recordings_path = "/home/rodrigob/code/doppia/src/applications/stixel_world/"

if False:
    
    options.recordings["fixed initial_results all_bboxes 1.8m 50px"] = \
        Recording(
            directory = base_recordings_path + "2011_01_05_67521_recordings_first_results",
            frontmost_bboxes_only = False)
    
    options.recordings["fixed all_bboxes"] = \
        Recording(
            directory = base_recordings_path + "2011_01_15_59709_recordings_fixed_height",
            frontmost_bboxes_only = False)
    
    options.recordings["fixed residual_image all_bboxes"] = \
        Recording(
            directory = base_recordings_path + "2011_01_15_61191_recordings_fixed_height_and_residual_image",
            frontmost_bboxes_only = False)
            
            
    options.recordings["fixed  no_postfiltering all_bboxes"] = \
        Recording(
            directory =  base_recordings_path + "2011_01_17_66016_recordings_stixel_world_to_use_with_fixed_height_with_no_post_filtering",
            frontmost_bboxes_only = False)
    
    options.recordings["fixed  no_postfiltering residual_ground with_1to1_ratio all_bboxes"] = \
        Recording(
            directory =  base_recordings_path + "2011_01_17_70193_recordings_stixel_world_to_use_with_fixed_height_with_no_post_filtering_and_residual_ground_WITH_BUG",
            frontmost_bboxes_only = False)
    
    options.recordings["fixed  no_postfiltering residual_ground with_1to1.5_ratio all_bboxes"] = \
        Recording(
            directory =  base_recordings_path + "2011_01_17_72941_recordings_stixel_world_to_use_with_fixed_height_with_no_post_filtering_and_residual_ground_with_1.5_ratio",
            frontmost_bboxes_only = False)
            
        
if False:
    options.recordings["two_steps partial_depth enforced_height all_bboxes"] = \
        Recording(
            directory = base_recordings_path + "2011_01_15_75456_recordings_stixel_world_to_use_with_two_steps_uses_partial_depth_map_and_enforced_heights",
            frontmost_bboxes_only = False)
            
    options.recordings["two_steps partial_depth without_enforced_height all_bboxes"] = \
        Recording(
            directory = base_recordings_path + "2011_01_15_75790_recordings_stixel_world_to_use_with_two_steps_uses_partial_depth_map_and_without_enforced_heights",
            frontmost_bboxes_only = False)
            
    options.recordings["two_steps full_depth enforced_height all_bboxes"] = \
        Recording(
            directory =  base_recordings_path + "2011_01_15_76098_recordings_stixel_world_to_use_with_two_steps_uses_full_depth_map_and_enforced_heights",
            frontmost_bboxes_only = False)
            
    options.recordings["two_steps no_postfiltering residual_ground without_enforced_height all_bboxes"] = \
        Recording(
            directory =  base_recordings_path + "2011_01_17_70426_recordings_stixel_world_to_use_with_two_steps_uses_partial_depth_map_without_enforced_heights_with_residual_ground",
            frontmost_bboxes_only = False)
    
    options.recordings["two_steps  no_postfiltering residual_ground enforced_pixel_height all_bboxes"] = \
        Recording(
            directory =  base_recordings_path + "2011_01_17_73880_recordings_stixel_world_to_use_with_two_steps_uses_partial_depth_map_with_pixels_enforced_heights_with_residual_ground_with_1.5_ratio",
            frontmost_bboxes_only = False)
            
    options.recordings["two_steps  no_postfiltering residual_ground enforced_pixel_height_12 ratio_1to1 all_bboxes"] = \
        Recording(
            directory =  base_recordings_path + "2011_01_17_85243_recordings_stixel_world_to_use_with_two_steps_uses_partial_depth_map_with_pixels_12_enforced_heights_with_residual_ground_with_1to1_ratio",
            frontmost_bboxes_only = False)

    options.recordings["two_steps  no_postfiltering residual_ground enforced_pixel_height_20 ratio_1to1 all_bboxes"] = \
        Recording(
            directory =  base_recordings_path + "2011_01_17_85378_recordings_stixel_world_to_use_with_two_steps_uses_partial_depth_map_with_pixels_20_enforced_heights_with_residual_ground_with_1to1_ratio",
            frontmost_bboxes_only = False)


     #  no_postfiltering residual_ground with_1to1_ratio all_bboxes
    options.recordings["fixed"] = \
        Recording(
            directory =  base_recordings_path + "2011_01_17_70193_recordings_stixel_world_to_use_with_fixed_height_with_no_post_filtering_and_residual_ground_WITH_BUG",
            frontmost_bboxes_only = False)
    
    # estimated partial depth
    # two_steps  no_postfiltering residual_ground enforced_pixel_height_20 ratio_1to1 all_bboxes
    options.recordings["estimated partial depth height60"] = \
        Recording(
            directory =  base_recordings_path + "2011_01_17_85378_recordings_stixel_world_to_use_with_two_steps_uses_partial_depth_map_with_pixels_20_enforced_heights_with_residual_ground_with_1to1_ratio",
            frontmost_bboxes_only = False)

    # estimated full depth
    # two_steps full_depth enforced_height all_bboxes
    options.recordings["estimated full depth"] = \
        Recording(
            directory =  base_recordings_path + "2011_01_15_76098_recordings_stixel_world_to_use_with_two_steps_uses_full_depth_map_and_enforced_heights",
            frontmost_bboxes_only = False)

    # two_steps full_depth enforced_height all_bboxes
    options.recordings["estimated partial depth height50 pixels12"] = \
        Recording(
            directory =  base_recordings_path + "2011_01_18_4371_recordings_stixel_world_to_use_with_two_steps_uses_partial_depth_map_with_pixels_12_enforced_heights_with_residual_ground_with_1.1_ratio_with_height50",
            frontmost_bboxes_only = False)

    options.recordings["estimated partial depth height50 pixels20"] = \
        Recording(
            directory =  base_recordings_path + "2011_01_18_7563_recordings_stixel_world_to_use_with_two_steps_uses_partial_depth_map_with_pixels_20_enforced_heights_with_residual_ground_with_1to1_ratio_with_height50",
            frontmost_bboxes_only = False)

    options.recordings["estimated full depth height50 pixels20"] = \
        Recording(
            directory =  base_recordings_path + "2011_01_18_7843_recordings_stixel_world_to_use_with_two_steps_uses_full_depth_map_with_pixels_20_enforced_heights_with_residual_ground_with_1to1_ratio",
            frontmost_bboxes_only = False)
 
    options.recordings["fixed height50"] = \
        Recording(
            directory =  base_recordings_path + "2011_01_18_16836_recordings_stixel_world_to_use_with_fixed_with_residual_ground_with_1to1_ratio_with_height50",
            frontmost_bboxes_only = False)
            

if False:
    # final plot of rss2011 submission
    options.recordings["rss2011 estimated height, partial depth"] = \
        Recording(
            directory =  rss2011_recordings_path + "2011_01_18_7563_recordings_stixel_world_to_use_with_two_steps_uses_partial_depth_map_with_pixels_20_enforced_heights_with_residual_ground_with_1to1_ratio_with_height50",
            frontmost_bboxes_only = False)

    options.recordings["rss2011 estimated height, full depth"] = \
        Recording(
            directory =  rss2011_recordings_path + "2011_01_18_7843_recordings_stixel_world_to_use_with_two_steps_uses_full_depth_map_with_pixels_20_enforced_heights_with_residual_ground_with_1to1_ratio",
            frontmost_bboxes_only = False)

    options.recordings["rss2011 fixed height"] = \
        Recording(
            directory =  rss2011_recordings_path + "2011_01_18_16836_recordings_stixel_world_to_use_with_fixed_with_residual_ground_with_1to1_ratio_with_height50",
            frontmost_bboxes_only = False)


if False:
    # a few baseline methods
    options.recordings["rss2011 fixed initial_results all_bboxes 1.8m 50px"] = \
        Recording(
            directory = rss2011_recordings_path + "2011_01_05_67521_recordings_first_results",
            frontmost_bboxes_only = False)
    
    
    options.recordings["baseline method"] = \
        Recording(
            directory =  iros2011_recordings_path + "2011_02_24_61080_recordings_baseline",
            frontmost_bboxes_only = False)

if False:
    # iros2011 experiments
    options.recordings["with_ground_cost_miroring_high_cost_weight"] = \
        Recording(
            directory =  iros2011_recordings_path + "2011_02_24_59621_recordings_with_ground_cost_miroring_high_cost_weight",
            frontmost_bboxes_only = False)

    options.recordings["without_ground_cost_miroring_high_cost_weight"] = \
        Recording(
            directory =  iros2011_recordings_path + "2011_02_24_60410_recordings_without_ground_cost_miroring_high_cost_weight",
            frontmost_bboxes_only = False)

    options.recordings["with_ground_cost_miroring_normal_cost_weight"] = \
        Recording(
            directory =  iros2011_recordings_path + "2011_02_24_64214_recordings_with_ground_cost_miroring_normal_cost_weight",
            frontmost_bboxes_only = False)

    options.recordings["without_ground_cost"] = \
        Recording(
            directory =  iros2011_recordings_path + "2011_02_25_35465_recordings_without_ground_cost",
            frontmost_bboxes_only = False)
            
    options.recordings["ground_cost_threshold_0.3"] = \
        Recording(
            directory =  iros2011_recordings_path + "2011_02_25_41072_recordings_with_ground_cost_threshold_0.3",
            frontmost_bboxes_only = False)

    options.recordings["u_disparity_boundary_diagonal_weight_0.0"] = \
        Recording(
            directory =  iros2011_recordings_path + "2011_02_25_50355_recordings_u_disparity_boundary_diagonal_weight_0.0",
            frontmost_bboxes_only = False)

if False:
    
    # more iros2011 experiments
    options.recordings["2011_03_09_no_enforcement"] = \
        Recording(
            directory =  iros2011_recordings_path + "2011_03_10_38201_recordings_no_enforcement",
            frontmost_bboxes_only = False)
            
if False:
    # plots of iros2011 submission

    #options.recordings["2011_03_09_fixed"] = \
    options.recordings["fixed height ours"] = \
        Recording(
            directory =  iros2011_recordings_path + "2011_03_09_70217_recordings_fixed",
            frontmost_bboxes_only = False)
    
    #options.recordings["2011_03_09_two_steps"] = \
    options.recordings["estimated height ours"] = \
        Recording(
            directory =  iros2011_recordings_path + "2011_03_09_72241_recordings_two_steps",
            frontmost_bboxes_only = False)

if True and False:
    # plots of iccv2011 workshop submission
    
    options.recordings["fixed height csbp"] = \
        Recording(
            directory =  iccv2011_workshop_recordings_path + "2011_07_07_62311_recordings_bahnhof_fixed_csbp",
            frontmost_bboxes_only = False)
            
    options.recordings["estimated height csbp"] = \
        Recording(
            directory =  iccv2011_workshop_recordings_path + "2011_07_07_54576_recordings_bahnhof_two_steps_csbp",
            frontmost_bboxes_only = False)

    options.recordings["fixed height simple sad"] = \
        Recording(
            directory =  iccv2011_workshop_recordings_path + "2011_07_07_60542_recordings_bahnhof_fixed_simple_sad_w_9",
            frontmost_bboxes_only = False)
            
    options.recordings["estimated height simple sad"] = \
        Recording(
            directory =  iccv2011_workshop_recordings_path + "2011_07_07_57673_recordings_bahnhof_two_steps_simple_sad_w_9",
            frontmost_bboxes_only = False)


if False:
      options.recordings["estimated height csbp no constraints"] = \
        Recording(
            directory =  iccv2011_workshop_recordings_path + "2011_07_07_68630_recordings_bahnhof_two_steps_csbp_no_constraints_no_filtering",
            frontmost_bboxes_only = False)

      options.recordings["estimated height csbp no filtering"] = \
        Recording(
            directory =  iccv2011_workshop_recordings_path + "2011_07_08_45509_recordings_bahnhof_two_steps_csbp_no_filtering",
            frontmost_bboxes_only = False)

      options.recordings["estimated height simple sad no filtering"] = \
        Recording(
            directory =  iccv2011_workshop_recordings_path + "2011_07_08_50800_recordings_bahnhof_two_steps_simple_wad_w_9_no_filtering",
            frontmost_bboxes_only = False)

if False:
      options.recordings["estimated height gpu_trees no filtering"] = \
        Recording(
            directory =  iccv2011_workshop_recordings_path + "2011_07_12_67071_recordings_bahnhof_two_steps_gpu_trees",
            frontmost_bboxes_only = False)

      options.recordings["estimated height opencv sad gray"] = \
        Recording(
            directory =  iccv2011_workshop_recordings_path + "2011_07_13_72596_recordings_bahnhof_two_steps_opencv_sad_gray",
            frontmost_bboxes_only = False)

if False:
    # ground filtering seems to provide only a very small improvement over no filtering        
    options.recordings["estimated height fast ground filtering"] = \
        Recording(
            directory =  base_recordings_path + "2011_07_20_52297_recordings_fast_stixels_with_ground_filtering",
            frontmost_bboxes_only = False)

    options.recordings["estimated height fast ground filtering height smoothing"] = \
        Recording(
            directory =  base_recordings_path + "2011_07_20_66376_recordings_fast_stixels_with_ground_filtering_with_height_cost_smoothing",
            frontmost_bboxes_only = False)

if False:
    options.recordings["estimated height 3 pixels wide"] = \
        Recording(
            directory =  base_recordings_path + "2011_07_19_66013_recordings_3_pixels_stixels",
            frontmost_bboxes_only = False)
    
    options.recordings["estimated height fast"] = \
        Recording(
            directory =  base_recordings_path + "2011_07_20_67563_recordings_fast_stixels",
            frontmost_bboxes_only = False)

    options.recordings["estimated height fast residual ground"] = \
        Recording(
            directory =  base_recordings_path + "2011_07_20_69135_recordings_fast_residual_ground",
            frontmost_bboxes_only = False)
    
    options.recordings["estimated height residual ground"] = \
        Recording(
            directory =  base_recordings_path + "2011_07_20_71630_recordings_residual_ground",
            frontmost_bboxes_only = False)
                    
    #options.recordings["estimated height fast residual ground cost volume margin"] = \
    #    Recording(
    #        directory =  base_recordings_path + "2011_07_21_48014_recordings_fast_ground_residual_cost_volume_margin",
    #        frontmost_bboxes_only = False)
    
    options.recordings["estimated height not fast residual ground cost volume margin"] = \
        Recording(
            directory =  base_recordings_path + "2011_07_21_49363_recordings_not_fast_ground_residual_cost_volume_margin",
            frontmost_bboxes_only = False)         
            
if False:
    # same results as in iros2011 submission 2011_03_09_72241_recordings_two_steps
    options.recordings["estimated height iros2011_03_09 baseline"] = \
        Recording(
            directory =   "/users/visics/rbenenso/code/doppia_iros2011/src/applications/stixel_world/" \
                        + "2011_07_21_54447_recordings_iros2011_03_09_baseline",
            frontmost_bboxes_only = False)         
    
                                                           
if False:
    # only for the histogram
    options.recordings["fixed"] = \
        Recording(
            directory =  base_recordings_path + "2011_01_17_70193_recordings_stixel_world_to_use_with_fixed_height_with_no_post_filtering_and_residual_ground_WITH_BUG",
            frontmost_bboxes_only = False)

    options.recordings["estimated"] = \
        Recording(
            directory =  base_recordings_path + "2011_01_17_85243_recordings_stixel_world_to_use_with_two_steps_uses_partial_depth_map_with_pixels_12_enforced_heights_with_residual_ground_with_1to1_ratio",
            frontmost_bboxes_only = False)    
            
    
    
#    2011_11_18_55646_recordings_vanilla_gpu_very_fast_over_bahnhof
    
    


if False:
    # plots of eccv2012 workshop submission,
    # varying stixel width
    
    options.recordings["stixel width 1"] = \
        Recording(
            directory = eccv2012_workshop_recordings_path + "2012_07_06_39132_recordings_support_1_width_1_row_steps_128",
            frontmost_bboxes_only = False)
            
    #options.recordings["stixel width 2"] = \
    #    Recording(
    #        directory = eccv2012_workshop_recordings_path + "2012_07_06_43330_recordings_support_1_width_2_row_steps_128",
    #        frontmost_bboxes_only = False)
            
    options.recordings["stixel width 3"] = \
        Recording(
            directory = eccv2012_workshop_recordings_path + "2012_07_06_40293_recordings_support_1_width_3_row_steps_128",
            frontmost_bboxes_only = False)

    #options.recordings["stixel width 4"] = \
    #    Recording(
    #        directory = eccv2012_workshop_recordings_path + "2012_07_06_42336_recordings_support_1_width_4_row_steps_128",
    #        frontmost_bboxes_only = False)

    options.recordings["stixel width 5"] = \
        Recording(
            directory = eccv2012_workshop_recordings_path + "2012_07_06_41631_recordings_support_1_width_5_row_steps_128",
            frontmost_bboxes_only = False)

    options.recordings["stixel width 7"] = \
        Recording(
            directory = eccv2012_workshop_recordings_path + "2012_07_06_42210_recordings_support_1_width_7_row_steps_128",
            frontmost_bboxes_only = False)

    options.recordings["stixel width 11"] = \
        Recording(
            directory = eccv2012_workshop_recordings_path + "2012_07_06_41779_recordings_support_1_width_11_row_steps_128",
            frontmost_bboxes_only = False)

    options.recordings["stixel width 21"] = \
        Recording(
            directory = eccv2012_workshop_recordings_path + "2012_07_06_41994_recordings_support_1_width_21_row_steps_128",
            frontmost_bboxes_only = False)            

    options.recordings["stixel v1"] = \
        Recording(
            directory = eccv2012_workshop_recordings_path + "2012_07_06_42473_recordings_stixels_v1_max_disparity_128",
            frontmost_bboxes_only = False)            

    #options.recordings["stixel v1 (d = 80)"] = \
    #    Recording(
    #        directory = eccv2012_workshop_recordings_path + "2012_07_06_42598_recordings_stixels_v1_max_disparity_80",
    #        frontmost_bboxes_only = False)            
            
            
            
if True:
    # plots of eccv2012 workshop submission,
    # varying number of row bands
            
    options.recordings["128 row bands"] = \
        Recording(
            directory = eccv2012_workshop_recordings_path + "2012_07_06_40293_recordings_support_1_width_3_row_steps_128",
            frontmost_bboxes_only = False)
            
    options.recordings["50 row bands"] = \
        Recording(
            directory = eccv2012_workshop_recordings_path + "2012_07_06_42764_recordings_support_1_width_3_row_steps_50",
            frontmost_bboxes_only = False)
    
    options.recordings["25 row bands"] = \
        Recording(
            directory = eccv2012_workshop_recordings_path + "2012_07_06_43073_recordings_support_1_width_3_row_steps_25",
            frontmost_bboxes_only = False)

    options.recordings["15 row bands"] = \
        Recording(
            directory = eccv2012_workshop_recordings_path + "2012_07_06_43185_recordings_support_1_width_3_row_steps_15",
            frontmost_bboxes_only = False)

    options.recordings["10 row bands"] = \
        Recording(
            directory = eccv2012_workshop_recordings_path + "2012_07_06_43824_recordings_support_1_width_3_row_steps_10",
            frontmost_bboxes_only = False)
                
    options.recordings["5 row bands"] = \
        Recording(
            directory = eccv2012_workshop_recordings_path + "2012_07_06_43431_recordings_support_1_width_3_row_steps_5",
            frontmost_bboxes_only = False)

if False:
    # plots of eccv2012 workshop submission,
    # varying stixel support width
                
    options.recordings["support width 1"] = \
        Recording(
            directory = eccv2012_workshop_recordings_path + "2012_07_06_43073_recordings_support_1_width_3_row_steps_25",
            frontmost_bboxes_only = False)

    options.recordings["support width 2"] = \
        Recording(
            directory = eccv2012_workshop_recordings_path + "2012_07_06_44175_recordings_support_2_width_3_row_steps_25",
            frontmost_bboxes_only = False)
            
    options.recordings["support width 3"] = \
        Recording(
            directory = eccv2012_workshop_recordings_path + "2012_07_06_44365_recordings_support_3_width_3_row_steps_25",
            frontmost_bboxes_only = False)

    options.recordings["support width 5"] = \
        Recording(
            directory = eccv2012_workshop_recordings_path + "2012_07_06_44444_recordings_support_5_width_3_row_steps_25",
            frontmost_bboxes_only = False)
        
    options.recordings["support width 10"] = \
        Recording(
            directory = eccv2012_workshop_recordings_path + "2012_07_06_44644_recordings_support_10_width_3_row_steps_25",
            frontmost_bboxes_only = False)
        

options.max_num_frames = -1
#options.max_num_frames = 5 # just for debugging
#options.max_num_frames = 900

options.minimum_box_height = 0


def render_paper_graphs():
    """
    Render the final graphs for RSS2011/ICCV2011 workshop/ECCV2012 workshop paper
    """
    
    application = StixelsEvaluationApplication()
    application.run(options)
    
    return


if __name__ == '__main__':
     # Import Psyco if available
    try:
        import psyco
        psyco.full()
    except ImportError:
        #print("(psyco not found)")
        pass
    else:
        print("(using psyco)")
        
    render_paper_graphs()
        
