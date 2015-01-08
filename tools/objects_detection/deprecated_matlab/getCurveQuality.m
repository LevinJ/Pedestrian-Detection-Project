function r = getCurveQuality(xs, ys)
 ref=10.^(-2:.01:0); 
    m=length(ref); rs=zeros(1,m);
   
    xs1=[xs'; 1]; ys1=[ys'; ys(end)];
  for i=1:m, j=find(xs1>=ref(i)); rs(i)=ys1(j(1)); end
  r=exp(mean(log(rs)));

end