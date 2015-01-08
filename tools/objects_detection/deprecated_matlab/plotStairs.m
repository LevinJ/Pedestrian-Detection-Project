function [area, handle] = plotStairs(figure_h, crop, meany, baseline5Area, linewidth, color)

thisArea = getArea(crop, meany);
area=abs(thisArea/baseline5Area*100);
figure(figure_h);
[a,b] = stairs(crop, meany);
a = [a(1); a(1:end-1)];
b = [b(2:end); b(end)];
handle = stairs(a,b, 'Color', color ,'lineWidth',linewidth);

end