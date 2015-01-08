function A = getArea(x,y)
    A = 0;
    for i=1:size(x,2)-1
        binwidth = abs(x(i+1) - x(i)) ;
        binheight = y(i) + y(i+1);
        binArea = 0.5*binwidth*binheight;
        A = A + binArea;
    end
    

end