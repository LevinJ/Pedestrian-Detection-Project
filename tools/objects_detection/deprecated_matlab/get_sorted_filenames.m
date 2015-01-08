function files = get_sorted_filenames(folder)
    files=dir(folder);
    filesnew = {};
    for i=1:size(files,1)
        if (~files(i).isdir())
            filesnew{end+1} = files(i).name;
        end
    end
    files = sort(filesnew);
    sort(files);


end