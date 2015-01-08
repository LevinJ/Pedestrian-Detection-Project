function files = get_sorted_dirnames(folder)
    files=dir(folder);
    files=files(3:end);
    filesnew = {};
    for i=1:size(files,1)
        if (files(i).isdir())
            filesnew{end+1} = files(i).name;
        end
    end
    files = sort(filesnew);
    


end