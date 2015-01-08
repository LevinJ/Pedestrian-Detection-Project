function [areaBaseline, areaBaseline2]=plotBaselineLeft(figure_h, range)
figure(figure_h); hold on; grid on
fnt={ 'FontSize',12 };

x(1) = 0.5
x(2) = 0.625;
x(3) = 0.75;
x(4) = 0.875;
x(5) = 1;
x = 1-x;

basefolder = '/users/visics/rbenenso/data/bertan_datasets/CalTechEvaluation/data-INRIA/res/Ours-wip/'
%half_height = [basefolder '/set01_2012_03_29_45922_recordings_2012_03_28_58135_trained_model_half_height/2012_03_29_45922_recordings_2012_03_28_58135_trained_model_half_heightOurs-wip.mat'];
half_height = [basefolder '/set01_2012_04_24_40863_recordings_2012_04_24_1445_crop_32/2012_04_24_40863_recordings_2012_04_24_1445_crop_32Ours-wip.mat'];
markersize=8;
load (half_height)
xs = xy(:,1);
ys = xy(:,2);
m = xs > range(1) & xs < range(2);
mm = 1-mean(ys(m));
y(1) = mm;
plot(x(1),mm,'ro', 'markersize', markersize, 'markerfacecolor', 'r','HandleVisibility','off');
%line([0.5,1],[halfmean, halfmean])



zero625height=[basefolder '/set01_2012_04_24_41114_recordings_2012_04_24_12534_crop_24/2012_04_24_41114_recordings_2012_04_24_12534_crop_24Ours-wip.mat'];
load (zero625height);
xs = xy(:,1);
ys = xy(:,2);
m = xs > range(1) & xs < range(2);
mm = 1-mean(ys(m));
y(2) = mm;
plot(x(2),mm,'ro', 'markersize', markersize, 'markerfacecolor', 'r','HandleVisibility','off');
xlabel('occlusion',fnt{:});
ylabel('mean recall',fnt{:});


zero75height = [basefolder '/set01_2012_04_24_41326_recordings_2012_04_24_1510_crop16/2012_04_24_41326_recordings_2012_04_24_1510_crop16Ours-wip.mat'];
load (zero75height)
xs = xy(:,1);
ys = xy(:,2);
m = xs > range(1) & xs < range(2);
mm = 1-mean(ys(m));
y(3) = mm;
plot(x(3),mm,'ro', 'markersize', markersize, 'markerfacecolor', 'r','HandleVisibility','off');





zero875height=[basefolder '/set01_2012_04_24_41565_recordings_2012_04_24_12437_crop_8/2012_04_24_41565_recordings_2012_04_24_12437_crop_8Ours-wip.mat'];
load (zero875height);
xs = xy(:,1);
ys = xy(:,2);
m = xs > range(1) & xs < range(2);
mm = 1-mean(ys(m));
y(4) = mm;
plot(x(4),mm,'ro', 'markersize', markersize, 'markerfacecolor', 'r','HandleVisibility','off');







full_height ='/users/visics/rbenenso/data/bertan_datasets/CalTechEvaluation/data-INRIA/res/Ours-wip/set01_2012_03_28_62839_recordings_2012_03_28_35668_NEWW_trained_model_full_height_CVPRini/2012_03_28_62839_recordings_2012_03_28_35668_NEWW_trained_model_full_height_CVPRiniOurs-wip.mat'
full_height =[basefolder '/set01_2012_04_17_58823_recordings_full_model/2012_04_17_58823_recordings_full_modelOurs-wip.mat']
load (full_height)
xs = xy(:,1);
ys = xy(:,2);
m = xs > range(1) & xs < range(2);
mm = 1-mean(ys(m));
y(5) = mm;
plot(x(5),mm,'ro', 'markersize', markersize, 'markerfacecolor', 'r','HandleVisibility','off');
plot(x,y, 'r--','lineWidth',3);
[a,b] = stairs([x(1) x(3) x(5)], [y(1) y(3) y(5)], 'r--');
plot(a,b,'c--','lineWidth',3);
%calcAreaBaseline
areaBaseline = getArea(x,y);
areaBaseline2 = (x(1) - x(3)) * y(1) + (x(3) - x(5))* y(3);


end