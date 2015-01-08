
% This script will detect faces in a set of images
% The minimum face detection size is 36 pixels,
% the maximum size is the full image.

images_folder_path = 'doppia/data/sample_test_images/pascal_faces';
results_folder_path = '~/face_detection_results';

model_path = 'doppia/data/trained_models/face_detection/dpm_baseline.mat';


face_model = load(model_path);

% lower detection threshold generates more detections
% detection_threshold = -0.5; 
detection_threshold = 0; 

% 0.3 or 0.2 are adequate for face detection.
nms_threshold = 0.3;

image_names = dir(fullfile(images_folder_path, '*.png'));

for i=1:numel(image_names)
%for i=1:5
   
    image_name = image_names(i).name;
    image_path = fullfile(images_folder_path, image_name);
    image = imread(image_path);
    [ds, bs] = process_face(image, face_model.model,  ...
                            detection_threshold, nms_threshold);
    
    result_path = fullfile(results_folder_path, [image_name, '.result.png']);
    showsboxes_face(image, ds, result_path);
    disp(['Created ', result_path]);
end

disp('All images processed');
