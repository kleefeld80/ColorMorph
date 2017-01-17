function [R,x0,y0]=FindR(x,y,r)
%tic
%[x0,y0,R]=miniCircle(x,y,r);
[R,x0,y0]=subgradient(x,y,r);
%toc
