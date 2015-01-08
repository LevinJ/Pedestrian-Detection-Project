  %%%%%%%%%%%%%%%%% Which Feature Pool OCTAVE 1
          lgd = []; handels = []; aps = []
       basefolder ='/users/visics/mmathias/devel/doppia/src/applications/objects_detection/warp/INRIA/octave1';
       resultfolder = '/esat/kochab/mmathias/CalTechEvaluation_3.0.0/data-INRIA/res/Ours-wip';
    h = figure(253); hold on; grid;
    lims=[3e-3 1e0+0.3 .025 1];
    axis( lims ); lgdLoc='sw'; plot([1 1],[eps 1],'-k','HandleVisibility','off');
        text(0.14,0.75,'INRIA', 'Color', colors{16}, 'FontSize',20, 'FontWeight', 'bold');
        text(0.14,0.6,'Scale 2 models', 'Color', colors{16}, 'FontSize',18, 'FontWeight', 'bold');
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

   
    names =     { 'gradientMirroredCopy', 'random', 'random-squares'     ,'HOG-multiscale',        'HOG-like'     ,'bestof_gradientMirrorCopy' , 'bestof_random'  , 'bestof_random-squares'};
    plotNames = {'RandomSymmetric 30k',     'Random 30k','SquaresChnFtrs 30k','SquaresChnFtrs All', 'SquaresChnFtrs 8x8', 'RandomSymmetric++',   'Random++','SquaresChnFtrs++'};
   % names =     { 'gradientMirroredCopy', 'HOG-multiscale',        'HOG-like'     ,'bestof_gradientMirrorCopy'};
   % plotNames = {'RandomSymmetric30k'  ,'SquaresChnFtrs_All', 'SquaresChnFtrs-8x8', 'BestOf_randomSymmetric'};
    plot_colors = {colors{1}, colors{8}, colors{8},colors{2}, colors{4},colors{1},colors{8},colors{8}}


    for i=1:numel(names)
        [aps{end+1}, lgd{end+1}, handels(i)] =  addCurve(h, plot_colors{i}, names(i), plotNames(i), basefolder, resultfolder);
        if strcmp(names{i}, 'bestof_gradientMirrorCopy')
            set(handels(i), 'LineStyle', '--');
        end
    end
    HOGmat = load('/users/visics/mmathias/devel/doppia/src/applications/objects_detection/warp/INRIA/HOG.mat');
    handels(end+1) = plot(HOGmat.xy(:,1),HOGmat.xy(:,2), 'Color', 'k', 'linewidth', 3);
    ap = getCurveQuality(HOGmat.xy(:,1)',HOGmat.xy(:,2)')*100;
    lgd{end+1} = sprintf('HOG, scale 1 (%.2f %%)', ap);
    aps{end+1} = sprintf('%.2f',ap);
    
    
    
    
    [dummy, order] = sort(aps);
    order = order(end:-1:1)
    
    h_legend = legend(handels(order), lgd(order), 'Location','SouthWest');
    
    set(h_legend,'FontSize',10);
    set(h_legend,'Interpreter','none');
    %uistack(h_legend,'down', numel(names)-1);
                  set(h, 'renderer', 'painters');