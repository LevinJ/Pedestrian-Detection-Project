%close all;
clear

green = [115 210 22]/255;
orange = [245 121 0]/255;
blue = [52 101 164]/255;
%purple = [117 80 123]/255;
purple = [173 127 168]/255;
red = [204 0 0]/255;
grey = [85 87 83]/255;
show = false;

%range = [1e-2, 1e-0];
%range = [1e-0-0.000001, 1e-0+0.000001];
range = [1e-1, 1e-0];
%range = [1e-2, 1e-1];
%range = [1e-1-0.01, 1e-1+0.01];
figure_h = figure;hold on; grid on
axis([0,.5,0,1])
set(gca,'YTick', [0:.1:1],'XTick',[0:1/16:0.5],'XTickLabel','0|1/16|2/16|3/16|4/16|5/16|6/16|7/16|8/16');
fnt={ 'FontSize',14 };
xlabel('occlusion level',fnt{:});
ylabel('mean recall',fnt{:});
 set(gca,'XDir','Reverse')
resultfolder='/users/visics/rbenenso/data/bertan_datasets/CalTechEvaluation/data-INRIA/res/Ours-wip/';

%full baseline
folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/left22222seed/baseline/';
folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/cvprtmp/baselineLeft';
[crop, meany] = get_crop_and_mean(folder, resultfolder,range, show);
plot(crop,meany,'o','color',red, 'markersize', 5, 'markerfacecolor', red,'HandleVisibility','off');

[a,b] = stairs(crop, meany);
a = [a(1); a(1:end-1)];
b = [b(2:end); b(end)];
%baselineArea = plotStairs(figure_h, crop, meany,1, 3, 'k');
s = stairs(a,b,'Color', red,'lineWidth',3);

baselineArea = getArea(crop, meany);
baselineLegend='Brute-force 100%';


%plot fillup
folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/left22222seed/fillup';
folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/cvprtmp/fillinLeft';
[crop, meany] = get_crop_and_mean(folder, resultfolder,range, show);
[fillupArea, fillupHandle] = plotStairs(figure_h, crop, meany,baselineArea, 3, green);
fillupLegend=['Filled-up ' sprintf('%.0f',fillupArea) '%'];

%plot compound
folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/left22222seed/franken_final';
folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/cvprtmp/frankenLeft';
[crop, meany] = get_crop_and_mean(folder, resultfolder,range, show);
[compoundArea, compoundHandle] = plotStairs(figure_h, crop, meany,baselineArea, 3, orange);
compoundLegend=['Franken ' sprintf('%.0f',compoundArea) '%'];





%baseline 2
aa = [a(1) a(1) a(9) a(9) a(17)];
bb = [b(1) b(9) b(9) b(17) b(17)];
plot([a(1) a(9) a(17)] ,[b(1) b(9) b(17)], 'o','Color', grey,  'markersize', 8, 'markerfacecolor', grey,'HandleVisibility','off')
b2 = plot (aa, bb, 'Color', grey, 'LineStyle', '--','lineWidth',3);
baseline2Area = getArea(aa, bb);
baseline2Legend=['3 classifiers '  sprintf('%.0f',baseline2Area/baselineArea*100) '%'];

%plot biased
folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/left22222seed/artificial_franken';
folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/cvprtmp/biasedLeft';
folder ='/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/cvprtmp/biased_reweight/';
[crop, meany] = get_crop_and_mean(folder, resultfolder,range, show);
[biasedArea, biasedHandle] =plotStairs(figure_h, crop, meany,baselineArea, 3, blue)
biasedLegend=['Biased '  sprintf('%.0f',biasedArea) '%'];



%naive
folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/left22222seed/artificial_normal/';
folder = '/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/cvprtmp/naiveLeft';
%folder ='/users/visics/mmathias/devel/doppia/src/applications/objects_detection/59166seed/cvprtmp/biased_reweight/';
[crop, meany] = get_crop_and_mean(folder, resultfolder,range, show);
[naiveArea , naiveHandle]= plotStairs(figure_h, crop, meany,baselineArea, 3, purple);
naiveLegend=['Naive '  sprintf('%.0f',naiveArea) '%'];


h_legend = legend([s compoundHandle fillupHandle biasedHandle b2 naiveHandle], baselineLegend, compoundLegend,fillupLegend, biasedLegend,baseline2Legend,   naiveLegend,'Location','SouthEast');


%h_legend = legend(baselineLegend,  compoundLegend,fillupLegend,baseline2Legend, biasedLegend , naiveLegend,'Location','SouthEast');
set(h_legend,'FontSize',15);
uistack(s, 'top') 
uistack(b2, 'bottom') 




