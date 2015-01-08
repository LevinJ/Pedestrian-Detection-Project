function [ap, lgd, curve_handle] = addCurve(figure_handle, color, name, plotname , basefolder, resultfolder)
    figure(figure_handle);
    range =[1e-4, 1];
    folder = fullfile(basefolder, name);
    folder{1}
    [meanx, meany, vary, maxy, miny, averaged] = getMeanOverXruns(folder{1}, range, resultfolder);
    s = 1;
    j = 1000;
    ap = sprintf('%.2f', getCurveQuality(meanx, meany)*100);
    lgd = strcat(plotname,' (', ap, '%)');
    lgd = lgd{1};
    curve_handle = plot(meanx,meany, 'Color', color, 'linewidth', 3);
    if (averaged == true)
        %errorbar(meanx(s:j:end), meany(s:j:end), maxy(s:j:end),'x','Color', color, 'linewidth',3)  ;
        jbfill(meanx,maxy, miny, color, color, true, 0.1);
        meanAp = getCurveQuality(meanx, meany)*100;
        up = getCurveQuality(meanx, maxy)*100 - meanAp;
        down = meanAp - getCurveQuality(meanx, miny)*100;
        
        variance = sprintf('%2.1f', (up+down)/2);
        
        lgd = strcat(plotname,' (', ap, ' ±' , variance , '%)');
        lgd = lgd{1};
    end

end