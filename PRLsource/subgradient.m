function [R,x0,y0]=subgradient(x,y,r)

n=length(x);
x0=0;
y0=0;
for i=1:n
  x0=x0+x(i)*r(i);
  y0=y0+y(i)*r(i);
end
%x0=x0/sum(r);
%y0=y0/sum(r);
x0=0;
y0=0;
alpha=1;
counter=1;
while abs(alpha)>1e-5
  grad=computeSubgradient(x,y,x0,y0,r);
  alpha=linesearchSubgradient(grad,x,y,x0,y0,r);
  x0=x0+grad(1)*alpha;
  y0=y0+grad(2)*alpha;
  if norm(grad)<1e-7
      break
  end
  if counter>100
      break
  end
  counter=counter+1;
end
R=compute(x,y,x0,y0,r);
