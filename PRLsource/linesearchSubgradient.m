function alpha=linesearchSubgradient(grad,x,y,x0,y0,r)

left=0;
right=1;

while (right-left)>1e-5
  alpha=(right+left)/2;
  R=compute(x,y,x0+alpha*grad(1),y0+alpha*grad(2),r);
  RR=compute(x,y,x0+1.01*alpha*grad(1),y0+1.01*alpha*grad(2),r);
  RL=compute(x,y,x0+0.99*alpha*grad(1),y0+0.99*alpha*grad(2),r);
  if RR>RL
    right=alpha;
  else
    if RR==RL
      break;
    else
      left=alpha;
    end
  end
end
