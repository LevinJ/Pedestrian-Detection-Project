function [meanx, meany, vary, maxy, miny, averaged] = getMeanOverXruns(folder, range, resultfolder)
    
    isETH = false;
    if (strfind(folder, 'ETH'))
        isETH = true;
    end
    dirs=get_sorted_dirnames(folder);
    
    newx = exp(log(range(1)):0.001:log(range(2)));
    newx = newx(newx>0.003472);
    meany = zeros(1,size(newx,2));
    s = zeros(1,size(newx,2));
    
    sampled_values = zeros(0, numel(newx));
    numdirs = numel(dirs);
    if (numdirs ==1)
        averaged = false;
        name{1}=dirs{1};

       
        foldername = ['set01_' dirs{1}];
        matfilename =[dirs{1} 'Ours-wip.mat'];
        if (isETH)
           foldername=['Ours-wip_' dirs{1}];
           matfilename =['Ours-wip.mat'];
        end

        
        matfilename = fullfile(resultfolder, foldername, matfilename);
        if ~exist(matfilename, 'file')
            return;
        end
        load(matfilename);
        xs = xy(:,1);
        ys = xy(:,2);
        meany = ys';
        meanx = xs';
        vary = ys';
        maxy = ys';
        miny =0;
        return
    end
        
    for i=1:numel(dirs) 
        averaged = true;
        name{i}=dirs{i};

        foldername = ['set01_' dirs{i}];

        matfilename =[dirs{i} 'Ours-wip.mat'];
        if (isETH)
           foldername=['Ours-wip_' dirs{i}];
           matfilename =['Ours-wip.mat'];
        end
        matfilename = fullfile(resultfolder, foldername, matfilename);
        if ~exist(matfilename, 'file')
            continue;
        end
        load(matfilename);
        xs = xy(:,1);
        ys = xy(:,2);
        
        sampled_values = [ sampled_values ; zeros(1, numel(newx)) ];
        
        for k=1:size(newx,2)
            value = newx(k);
            nearest = max(xs(xs <= newx(k)));
            positions = xs == nearest;
            sampled_values(end, k) = min(ys(positions));
        end

      
        
        m = xs > range(1) & xs < range(2);
        
    end
    
    meany = mean(sampled_values);
    meanx = newx;
    vary = std(sampled_values);
    maxy = max(sampled_values);%-meany;
    miny = min(sampled_values);
    %maxy = max(maxy, miny);
    %miny = min(miny
    
    
end