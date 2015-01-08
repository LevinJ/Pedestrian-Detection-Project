function [areaBaseline, areaBaseline2] = plotBaseline59622(figure_h, range)
figure(figure_h); hold on; grid on
fnt={ 'FontSize',12 };
x(1) = 0.5
x(2) = 0.625;
x(3) = 0.6875;
x(4) = 0.75;
x(5) = 0.875;
x(6) = 1;
x = 1-x;

basefolder = '/users/visics/rbenenso/data/bertan_datasets/CalTechEvaluation/data-INRIA/res/Ours-wip/'
%half_height = [basefolder '/set01_2012_03_29_45922_recordings_2012_03_28_58135_trained_model_half_height/2012_03_29_45922_recordings_2012_03_28_58135_trained_model_half_heightOurs-wip.mat'];
half_height = [basefolder '/set01_2012_04_30_60543_recordings_70411_size_0.5_model_seed59166_noPushup/2012_04_30_60543_recordings_70411_size_0.5_model_seed59166_noPushupOurs-wip.mat'];
markersize=8;
load (half_height)
xs = xy(:,1);
ys = xy(:,2);
m = xs > range(1) & xs < range(2);
mm = 1-mean(ys(m));
y(1) = mm;
plot(x(1),mm,'ro', 'markersize', markersize, 'markerfacecolor', 'r','HandleVisibility','off');
%line([0.5,1],[halfmean, halfmean])

zero625height='/users/visics/rbenenso/data/bertan_datasets/CalTechEvaluation/data-INRIA/res/Ours-wip/set01_2012_04_13_49684_recordings_2012_04_12_80905_size_0.625_model_recursive_seed_0/2012_04_13_49684_recordings_2012_04_12_80905_size_0.625_model_recursive_seed_0Ours-wip.mat';
zero625height=[basefolder '/set01_2012_04_30_65553_recordings_2012_04_27_81409_size_0.625_model/2012_04_30_65553_recordings_2012_04_27_81409_size_0.625_modelOurs-wip.mat'];
load (zero625height);
xs = xy(:,1);
ys = xy(:,2);
m = xs > range(1) & xs < range(2);
mm = 1-mean(ys(m));
y(2) = mm;
plot(x(2),mm,'ro', 'markersize', markersize, 'markerfacecolor', 'r','HandleVisibility','off');
xlabel('occlusion',fnt{:});
ylabel('mean recall',fnt{:});

zero69height ='/users/visics/rbenenso/data/bertan_datasets/CalTechEvaluation/data-INRIA/res/Ours-wip/set01_2012_04_11_59478_recordings_2012_04_11_576_trained_model_75_object_height_octave_0/2012_04_11_59478_recordings_2012_04_11_576_trained_model_75_object_height_octave_0Ours-wip.mat';
zero69height= [basefolder '/set01_2012_05_01_68650_recordings_2012_04_28_5838_size_0.6875_model_seed59166/2012_05_01_68650_recordings_2012_04_28_5838_size_0.6875_model_seed59166Ours-wip.mat'];
load (zero69height)
xs = xy(:,1);
ys = xy(:,2);
m = xs > range(1) & xs < range(2);
mm = 1-mean(ys(m));
y(3) = mm;
plot(x(3),mm,'ro', 'markersize', markersize, 'markerfacecolor', 'r','HandleVisibility','off');

zero75height ='/users/visics/rbenenso/data/bertan_datasets/CalTechEvaluation/data-INRIA/res/Ours-wip/set01_2012_03_29_46266_recordings_2012_03_28_59943_trained_model_75_height_octave_0/2012_03_29_46266_recordings_2012_03_28_59943_trained_model_75_height_octave_0Ours-wip.mat';
zero75height = [basefolder '/set01_2012_05_01_75654_recordings_012_04_28_16647_size_0.75_model_seed59166/2012_05_01_75654_recordings_012_04_28_16647_size_0.75_model_seed59166Ours-wip.mat'];


load (zero75height)
xs = xy(:,1);
ys = xy(:,2);
m = xs > range(1) & xs < range(2);
mm = 1-mean(ys(m));
y(4) = mm;
plot(x(4),mm,'ro', 'markersize', markersize, 'markerfacecolor', 'r','HandleVisibility','off');


zero875height='/users/visics/rbenenso/data/bertan_datasets/CalTechEvaluation/data-INRIA/res/Ours-wip/set01_2012_04_13_49969_recordings_2012_04_12_80903_size_0.875_model_recursive_seed_0/2012_04_13_49969_recordings_2012_04_12_80903_size_0.875_model_recursive_seed_0Ours-wip.mat';
zero875height=[basefolder '/set01_2012_05_03_39648_recordings_2012_04_28_18932_size_0.875_model_seed59166/2012_05_03_39648_recordings_2012_04_28_18932_size_0.875_model_seed59166Ours-wip.mat'];
load (zero875height);
xs = xy(:,1);
ys = xy(:,2);
m = xs > range(1) & xs < range(2);
mm = 1-mean(ys(m));
y(5) = mm;
plot(x(5),mm,'ro', 'markersize', markersize, 'markerfacecolor', 'r','HandleVisibility','off');


full_height ='/users/visics/rbenenso/data/bertan_datasets/CalTechEvaluation/data-INRIA/res/Ours-wip/set01_2012_03_28_62839_recordings_2012_03_28_35668_NEWW_trained_model_full_height_CVPRini/2012_03_28_62839_recordings_2012_03_28_35668_NEWW_trained_model_full_height_CVPRiniOurs-wip.mat'
full_height =[basefolder '/set01_2012_05_03_40355_recordings_2012_04_28_7267_size_full_model_seed59166/2012_05_03_40355_recordings_2012_04_28_7267_size_full_model_seed59166Ours-wip.mat']
load (full_height)
xs = xy(:,1);
ys = xy(:,2);
m = xs > range(1) & xs < range(2);
mm = 1-mean(ys(m));
y(6) = mm;
plot(x(6),mm,'ro', 'markersize', markersize, 'markerfacecolor', 'r','HandleVisibility','off');
%line([0.5,1],[halfmean, halfmean])
plot(x,y, 'r--','lineWidth',3);


[a,b] = stairs([x(1) x(4) x(6)], [y(1) y(4) y(6)], 'r--');
plot(a,b,'c--','lineWidth',3);
%calcAreaBaseline
areaBaseline = getArea(x,y);
areaBaseline2 = (x(4) - x(1)) * y(1) + (x(6) - x(4))* y(4);



a = [x(1:2) x(4:end)];
b = [y(1:2) y(4:end)];
areaBaseline5 = getArea(a,b)

end