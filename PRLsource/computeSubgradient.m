function  grad=computeSubgradient(x,y,x0,y0,r)

grad=zeros(2,1);
n=length(x);
d=zeros(n,1);
for i=1:n
  d(i)=sqrt((x(i)-x0)^2+(y(i)-y0)^2)+r(i);
end
d_max=max(d);
for i=1:n
  if max(d_max-d(i))<1e-8
    grad(1)=grad(1)+x(i)-x0;
    grad(2)=grad(2)+y(i)-y0;
  end
end