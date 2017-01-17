function R=compute(x,y,x0,y0,r)


[n,m]=size(x);
R=0;
for i=1:n
  temp=sqrt((x(i)-x0)^2+(y(i)-y0)^2)+r(i);
  if temp>R
    R=temp;
  end
end
