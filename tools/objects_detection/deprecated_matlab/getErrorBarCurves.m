function getErrorBarCurves(folder, resultfolder, range, saveName)
    dirs=get_sorted_dirnames(folder);
    colors = varycolor(17);
    lims=[3e-3 1e1 .025 1];
    show=true;
    if show
        figure; hold on; grid;
         axis( lims ); lgdLoc='sw'; plot([1 1],[eps 1],'-k','HandleVisibility','off');
     set(gca,'XScale','log','YScale','log','YTick',[0 .05 .1:.1:1]);
    set(gca,'XMinorGrid','off','XMinorTic','off');
    set(gca,'YMinorGrid','off','YMinorTic','off');
    fnt={ 'FontSize',12 };
    set(gcf, 'DefaultAxesColorOrder', colors);
    str='';
    if(~isempty(str)), title(['ROC for ' str],fnt{:}); end
    
    xlabel('false positives per image',fnt{:});
    ylabel('miss rate',fnt{:});
    end

%     newx = range(1):0.001:range(2);
    newx = exp(log(range(1)):0.001:log(range(2)));
    meany = zeros(1,size(newx,2));
    s = zeros(1,size(newx,2));
    
    sampled_values = zeros(0, numel(newx));
    
    for i=1:numel(dirs) 
        name{i}=dirs{i};

        foldername = ['set01_' dirs{i}];

        matfilename =[dirs{i} 'Ours-wip.mat'];
        matfilename = fullfile(resultfolder, foldername, matfilename)
        if ~exist(matfilename, 'file')
            continue;
        end
        load(matfilename);
        xs = xy(:,1);
        ys = xy(:,2);
        
        sampled_values = [ sampled_values ; zeros(1, numel(newx)) ];
        
        for k=1:size(newx,2)
            value = newx(k);
%             [m,idx] = min(abs(xs-value) );
            nearest = max(xs(xs <= newx(k)));
            positions = xs == nearest;
%             positions = (xs == xs(idx));
            sampled_values(end, k) = min(ys(positions));
%             s(k) = s(k) + sum(positions);
%             meany(k) = meany(k)+ sum(ys(positions));

        end

      
        
        m = xs > range(1) & xs < range(2);
        %m =xs
        
%         if show
%             plot(xs(m), ys(m), 'Color', colors(i,:), 'linewidth',3 );

%             
%         end
        
    end
    
    meany = mean(sampled_values);
    vary = std(sampled_values);
    maxy = max(sampled_values)-meany;
    miny = meany-min(sampled_values);
    maxy = max(maxy, miny);
    
    
    
    
    
%     meany = meany./s;
%     
%    %variance
%     newx = range(1):0.001:range(2);
%     vary = zeros(1,size(newx,2));
%     s = zeros(1,size(newx,2));
%     
%     for i=1:numel(dirs) 
%         name{i}=dirs{i};
% 
%         foldername = ['set01_' dirs{i}];
% 
%         matfilename =[dirs{i} 'Ours-wip.mat'];
%         matfilename = fullfile(resultfolder, foldername, matfilename)
%         if ~exist(matfilename, 'file')
%             continue;
%         end
%         load(matfilename);
%         xs = xy(:,1);
%         ys = xy(:,2);
%         for k=1:size(newx,2)
%             value = newx(k)
%             [m,idx] = min(abs(xs-value) );
%             positions = (xs == xs(idx));
%             s(k) = s(k) + sum(positions);
%             vary(k) = sum((meany(k)- ys(positions)).^2);
% 
%         end
% 
%         
%     end
%     vary = vary./s
%     
   
   plot(newx,meany, 'Color', 'r', 'linewidth', 3)
   errorbar(newx(1:300:end), meany(1:300:end), maxy(1:300:end),'kx', 'linewidth',3)  
   errorbar(newx(1:300:end), meany(1:300:end), vary(1:300:end),'rx', 'linewidth',3)  
    save(saveName, 'newx', 'meany', 'vary', 'maxy');
    legend(name,'FontSize',15)
 


end