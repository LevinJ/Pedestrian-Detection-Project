close all;
clear
loadColors;
range = [0.003472, 1e-0];
%range = [1e-0-0.000001, 1e-0+0.000001];
%range = [1e-2, 1e-1];
%range = [1e-1-0.01, 1e-1+0.01];
%range =[1e-4, 1]

range = [1e-1, 1e-0];
show = false;
resultfolder='/users/visics/rbenenso/data/bertan_datasets/CalTechEvaluation/data-INRIA/res/Ours-wip/';

%%setup figure
figure_h = figure;hold on; grid on
axis([0,.5,0,1])
set(gca,'XTick',[0:1/16:0.5],'XTickLabel','0|2/32|4/32|6/32|8/32|10/32|12/32|14/32|16/32');
fnt={ 'FontSize',14 };
xlabel('occlusion level',fnt{:});
ylabel('mean recall',fnt{:});
set(gca,'XDir','Reverse')


% folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/cvprtmp/variance/nopushup/';
% getErrorBarCurves(folder, resultfolder, range, 'noPushup.mat')
% folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/cvprtmp/variance/pushup/';
% getErrorBarCurves(folder, resultfolder, range, 'Pushup.mat')
% return


%full baseline
folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/baseline/';
folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/cvprtmp/baseline';
folder ='/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/cvprtmp/baseline2';
show = true;
[crop, meany] = get_crop_and_mean(folder, resultfolder,range, show);
plot(crop,meany,'o','color',red, 'markersize', 5, 'markerfacecolor', red,'HandleVisibility','off');

[a,b] = stairs(crop, meany);
a = [a(1); a(1:end-1)];
b = [b(2:end); b(end)];
%baselineArea = plotStairs(figure_h, crop, meany,1, 3, 'k');
s = stairs(a,b,'Color', red,'lineWidth',3);

baselineArea = getArea(crop, meany);
%baselineArea = getArea(a, b);
baselineLegend='Brute-force 100%';


%plot fillup classifier
folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/franken_non_recursive';
[crop, meany] = get_crop_and_mean(folder, resultfolder,range, show);
[fillupArea, fillupHandle] = plotStairs(figure_h, crop, meany,baselineArea, 3, green);
fillupLegend=['Filled-up ' sprintf('%.0f',fillupArea) '%'];


%plot compound classifier
%folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/franken_recursive';
%folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/cvpr2013/bottom_svm';
folder ='/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/cvpr2013/franken_bmvc';
folder ='/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/cvprtmp/a';
folder ='/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/cvprtmp/bmvcdetWithHack';
folder ='/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/cvprtmp/e';
%folder ='/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/cvprtmp/c';
%folder ='/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/cvprtmp/biased';
%folder ='/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/cvprtmp/withSVM';


[crop, meany] = get_crop_and_mean(folder, resultfolder,range, show);
[compoundArea, compoundHandle] = plotStairs(figure_h, crop, meany,baselineArea, 3, orange);
compoundLegend=['Franken ' sprintf('%.0f',compoundArea) '%'];
    


%uistack(fillupHandle, 'bottom') 

%baseline 2
aa = [a(1) a(1) a(17) a(17) a(33)];
bb = [b(1) b(17) b(17) b(33) b(33)];
plot([a(1) a(17) a(33)] ,[b(1) b(17) b(33)], 'o','Color', grey,  'markersize', 8, 'markerfacecolor', grey,'HandleVisibility','off')
b2 = plot (aa, bb, 'Color', grey, 'LineStyle', '--','lineWidth',3);
baseline2Area = getArea(aa, bb);
baseline2Legend=['3 classifiers '  sprintf('%.0f',baseline2Area/baselineArea*100) '%'];



%plot biased 
folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/franken_art';
folder ='/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/cvprtmp/biased';
folder ='/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/cvprtmp/biased_reweight_up';

[crop, meany] = get_crop_and_mean(folder, resultfolder,range, show);
[biasedArea, biasedHandle]=plotStairs(figure_h, crop, meany,baselineArea, 3, blue)
biasedLegend=['Biased '  sprintf('%.0f',biasedArea) '%'];



%naive
show = true;
folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/full_art/';
folder ='/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/cvprtmp/naive';
[crop, meany] = get_crop_and_mean(folder, resultfolder,range, show);
[naiveArea, naiveHandle] = plotStairs(figure_h, crop, meany,baselineArea, 3, purple);
naiveLegend=['Naive '  sprintf('%.0f',naiveArea) '%'];

%baseline 4
aa = [a(1) a(1) a(7) a(7) a(17) a(17) a(25) a(25) a(33)];
bb = [b(1) b(7) b(7) b(17) b(17) b(25) b(25) b(33) b(33)];
%plot (aa, bb, 'k--','lineWidth',3);
baseline4Area = getArea(aa, bb);
baseline4Legend=['Basline 4 '  sprintf('%.0f',baseline4Area/baselineArea*100) '%']










%h_legend = legend(baselineLegend, compoundLegend,fillupLegend, baseline2Legend, biasedLegend , naiveLegend,'Location','SouthEast');
h_legend = legend([s compoundHandle fillupHandle biasedHandle b2 naiveHandle], baselineLegend, compoundLegend,fillupLegend, biasedLegend,baseline2Legend,   naiveLegend,'Location','SouthEast');
set(h_legend,'FontSize',15);
uistack(s, 'top') 
uistack(b2, 'bottom') 
