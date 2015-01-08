 %%%%%%%%%%%%%%%%%Which training set
    resultfolder = '/esat/kochab/mmathias/caltech_pedestrian/evaluation/code3.0.0/data-ETH/res/';
    basefolder ='/users/visics/mmathias/devel/doppia/src/applications/objects_detection/warp/ETH';
    lgd = []; handels = []; aps = []
    
    h = figure(227); hold on; grid;
    lims=[3e-3 1e0+4 .1 1];
    axis( lims ); lgdLoc='sw'; plot([1 1],[eps 1],'-k','HandleVisibility','off');
     text(0.3,0.85,'ETH', 'Color', colors{16}, 'FontSize',20, 'FontWeight', 'bold');
    set(gca,'XScale','log','YScale','log','YTick',[0 .05 .1:.1:1]);
    set(gca,'XMinorGrid','off','XMinorTic','off');
    set(gca,'YMinorGrid','off','YMinorTic','off');
    fnt={ 'FontSize',12 };
    xlabel('false positives per image',fnt{:});
    ylabel('miss rate',fnt{:});

    str='';
    if(~isempty(str)), title(['ROC for ' str],fnt{:}); end
    %colors = varycolor(numel(names));
    %set(gcf, 'DefaultAxesColorOrder', colors);

%     names = {'HOG-multiscale-moreNegatives2' ,'smallJitter', 'HOG-multiScale_hardNegatives',  'HOG-multiscale-crisp', 'moreNegatives', 'HOG-multiScale-INRIASplit'};
%     
%     plotNames = { 'PascalNegatives', 'Jitter', 'PedestrianNegatives', 'Inria', 'PascalNegatives All', 'Inria original crops'};
% names = {'HOG-multiscale-moreNegatives2' ,'smallJitter', 'HOG-multiScale_hardNegatives',  'HOG-multiscale-crisp'};
    
    %plotNames = { 'PascalNegatives', 'Jitter', 'PedestrianNegatives', 'Inria'};
   % plot_colors = { colors{2}, colors{1},colors{4}, colors{3},colors{5},colors{1},colors{8},colors{8}}
     names = {'smallJitter', 'HOG-multiScale_hardNegatives',  'HOG-multiscale-crisp'};
    
    plotNames = {  'Jitter', 'PedestrianNegatives', 'Inria'};
 plot_colors = { colors{1},colors{4}, colors{3},colors{5},colors{1},colors{8},colors{8}}
 

    for i=1:numel(names)
        [aps{end+1}, lgd{end+1}, handels(i)] =  addCurve(h, plot_colors{i}, names(i), plotNames(i), basefolder, resultfolder);
    end
    
    HOGmat = load('/users/visics/mmathias/devel/doppia/src/applications/objects_detection/warp/ETH/HOG.mat');
    handels(end+1) = plot(HOGmat.xy(:,1),HOGmat.xy(:,2), 'Color', 'k', 'linewidth', 3);
    ap = getCurveQuality(HOGmat.xy(:,1)',HOGmat.xy(:,2)')*100;
    lgd{end+1} = sprintf('HOG (%.2f %%)', ap);
    aps{end+1} = sprintf('%.2f',ap);
%     MLSmat = load('/users/visics/mmathias/devel/doppia/src/applications/objects_detection/warp/INRIA/MLS.mat');
%     handels(end+1) = plot(MLSmat.xy(:,1),MLSmat.xy(:,2), 'Color', colors{5}, 'linewidth', 3);
%     ap = getCurveQuality(MLSmat.xy(:,1)',MLSmat.xy(:,2)')*100;
%     lgd{end+1} = sprintf('MLS (%.2f %%)', ap);
%     aps{end+1} = sprintf('%.2f',ap);
    
    
    
    [dummy, order] = sort(aps);
    order = order(end:-1:1)
    
    h_legend = legend(handels(order), lgd(order), 'Location','SouthWest');
    set(h_legend,'FontSize',15);
    set(h_legend,'Interpreter','none');
                set(h, 'renderer', 'painters');
    