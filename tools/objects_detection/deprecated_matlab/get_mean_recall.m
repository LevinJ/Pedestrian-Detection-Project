function [recall, croppingfraction] = get_mean_recall(folder,xrange)
    files = get_sorted_filenames(folder);
    kk = 0;
    values = 0;
    croppingfraction = 0;
    if size(files,2) == 17
        for i=1:size(files,2)

                fn = [folder cell2mat(files(i))]
                f = load(fn);
                meanvalue = 0
                m = (f(:,2) < xrange(2)) & (f(:,2) > xrange(1));
                %plot(f(m,2), f(m,1), 'g','linewidth',3 );
                kk = kk+1
                values(kk) = mean(f(m,1));
                croppingfraction(kk) = ((i-1)*4)/128;
        end
    elseif size(files,2) == 1
        fn = [folder cell2mat(files(1))]
        f = load(fn);
        
        m = (f(:,2) < xrange(2)) & (f(:,2) > xrange(1));
                %plot(f(m,2), f(m,1), 'g','linewidth',3 );
        meanvalue = mean(f(m,1));
        
        for i=1:17
            kk = kk+1;
            values(kk) = meanvalue;    
            croppingfraction(kk) = ((i-1)*4)/128;
        end
            
    end
    recall = 1-values;
    croppingfraction = 1-croppingfraction;
    

end