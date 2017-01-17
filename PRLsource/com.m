% Copyright by Andreas Kleefeld
% Last updated 12/11/2013
function com(filenamein,filenameout,choice,iter)
    img=im2double(imread(filenamein));
    [N,M,~]=size(img);
    fprintf('Filename:%s %d %d\n',filenamein,N,M);
    
    % get the RGB values
    R=img(:,:,1);
    G=img(:,:,2);
    B=img(:,:,3);
    
    Rneu=R;
    Gneu=G;
    Bneu=B;
    
    fprintf('Choice=%d, Iter=%d\n',choice,iter);
    if choice==1
        for i=1:iter
            Rneu=dilation(R);
            R=Rneu;
            Gneu=dilation(G);
            G=Gneu;
            Bneu=dilation(B);
            B=Bneu;
        end
    elseif choice==2
        for i=1:iter
            Rneu=erosion(R);
            R=Rneu;
            Gneu=erosion(G);
            G=Gneu;
            Bneu=erosion(B);
            B=Bneu;
        end
    elseif choice==3
        Rneu=opening(R); 
        Gneu=opening(G); 
        Bneu=opening(B); 
    elseif choice==4
        Rneu=closing(R);  
        Gneu=closing(G);  
        Bneu=closing(B);  
    elseif choice==5
        Rneu=wth(R); 
        Gneu=wth(G);
        Bneu=wth(B); 
    elseif choice==6
        Rneu=bth(R);  
        Gneu=bth(G);
        Bneu=bth(B);
    elseif choice==7
        Rneu=sdth(R);
        Gneu=sdth(G);
        Bneu=sdth(B);
    elseif choice==8
        Rneu=beucher(R);
        Gneu=beucher(G);
        Bneu=beucher(B);
    elseif choice==9
        Rneu=internalgradient(R);
        Gneu=internalgradient(G);
        Bneu=internalgradient(B);
    elseif choice==10
        Rneu=externalgradient(R);
        Gneu=externalgradient(G);
        Bneu=externalgradient(B);
    elseif choice==11
        Rneu=mlaplacian(R);
        Gneu=mlaplacian(G);
        Bneu=mlaplacian(B);
    elseif choice==12
        Rneu=shockfilter(R);
        Gneu=shockfilter(G);
        Bneu=shockfilter(B);
    end
    
    img2=zeros(N,M,3,'uint8');
    for i=1:N
        for j=1:M
            img2(i,j,1)=uint8(Rneu(i,j)*255.0);
            img2(i,j,2)=uint8(Gneu(i,j)*255.0);
            img2(i,j,3)=uint8(Bneu(i,j)*255.0);
        end
    end
    imagesc(img2);
    imwrite(img2,filenameout);
end

function out=dilation(in)
    global mask
    
    [N,M]=size(in);
    out=zeros(N,M);
    
    length=(size(mask)-1)/2;
    for i=1:N
        for j=1:M
            % get the points according to the mask
            inds=j-length:j+length;
            len=size(inds(inds>M),2);
            indj=[abs(inds(inds<=0))+1,inds(inds>=1 & inds<=M),M*ones(1,len)+1-(1:len)];
            inds=i-length:i+length;
            len=size(inds(inds>N),2);
            indi=[abs(inds(inds<=0))+1,inds(inds>=1 & inds<=N),N*ones(1,len)+1-(1:len)];
            window=in(indi,indj);
            
            out(i,j)=max(max(window));
        end
    end
end

function out=erosion(in)
    global mask
    
    [N,M]=size(in);
    out=zeros(N,M);
    
    length=(size(mask)-1)/2;
    for i=1:N
        for j=1:M
            % get the points according to the mask
            inds=j-length:j+length;
            len=size(inds(inds>M),2);
            indj=[abs(inds(inds<=0))+1,inds(inds>=1 & inds<=M),M*ones(1,len)+1-(1:len)];
            inds=i-length:i+length;
            len=size(inds(inds>N),2);
            indi=[abs(inds(inds<=0))+1,inds(inds>=1 & inds<=N),N*ones(1,len)+1-(1:len)];
            window=in(indi,indj);       
            out(i,j)=min(min(window));
        end
    end
end

function out=dilation2(in)
    [N,M]=size(in);
    out=zeros(N,M);
    
    % Mask B
    indexH=1:M;
    indexHm=indexH-1;
    indexHm(1)=1;
    indexHp=indexH+1;
    indexHp(M)=M;   
    indexV=1:N;
    indexVm=indexV-1;
    indexVm(1)=1;
    indexVp=indexV+1;
    indexVp(N)=N;
    
    for i=1:N
        for j=1:M
            north=in(i         ,indexHm(j));
            south=in(i         ,indexHp(j));
            west =in(indexVm(i),j         );
            east =in(indexVp(i),j         );
            cent =in(i         ,j         );
            
            out(i,j)=max([north,south,west,east,cent]);
        end
    end
end

function out=erosion2(in)
    [N,M]=size(in);
    out=zeros(N,M);
    
    % Mask B
    indexH=1:M;
    indexHm=indexH-1;
    indexHm(1)=1;
    indexHp=indexH+1;
    indexHp(M)=M;   
    indexV=1:N;
    indexVm=indexV-1;
    indexVm(1)=1;
    indexVp=indexV+1;
    indexVp(N)=N;
    
    for i=1:N
        for j=1:M
            north=in(i         ,indexHm(j));
            south=in(i         ,indexHp(j));
            west =in(indexVm(i),j         );
            east =in(indexVp(i),j         );
            cent =in(i         ,j         );
            
            out(i,j)=min([north,south,west,east,cent]);
        end
    end
end

function out=shockfilter(in)
    out2=mlaplacian(in);
    d=dilation(in);
    e=erosion(in);
    out=out2;
    [N,M,~]=size(out);
    for i=1:N
        for j=1:M
            if trace(vec2mat(out2(i,j,:)))<=0
                out(i,j,:)=d(i,j,:);
            else
                out(i,j,:)=e(i,j,:);
            end
        end
    end
end

function out=mlaplacian(in)
    out=externalgradient(in)-internalgradient(in);
end

function out=internalgradient(in)
    out=in-erosion(in);
end

function out=externalgradient(in)
    out=dilation(in)-in;
end

function out=beucher(in)
    out=dilation(in)-erosion(in);
end

function out=sdth(in)
    out=closing(in)-opening(in);
end

function out=bth(in)
    out=closing(in)-in;
end

function out=wth(in)
    out=in-opening(in);
end

function out=opening(in)
    out=dilation(erosion(in));
end

function out=closing(in)
    out=erosion(dilation(in));
end

function mat=vec2mat(vec)
    a=vec(1);
    b=vec(2);
    c=vec(3);
    mat=[a b; b c];
end

function d = im2double(img, typestr) 
%IM2DOUBLE Convert image to double precision. 
%   IM2DOUBLE takes an image as input, and returns an image of 
%   class double.  If the input image is of class double, the 
%   output image is identical to it.  If the input image is of 
%   class uint8, im2double returns the equivalent image of class 
%   double, rescaling or offsetting the data as necessary. 
% 
%   I2 = IM2DOUBLE(I1) converts the intensity image I1 to double 
%   precision, rescaling the data if necessary. 
% 
%   RGB2 = IM2DOUBLE(RGB1) converts the truecolor image RGB1 to 
%   double precision, rescaling the data if necessary. 
% 
%   BW2 = IM2DOUBLE(BW1) converts the binary image BW1 to double 
%   precision. 
% 
%   X2 = IM2DOUBLE(X1,'indexed') converts the indexed image X1 to 
%   double precision, offsetting the data if necessary. 
%  
%   See also DOUBLE, IM2UINT8, UINT8. 
 
%   Chris Griffin 6-9-97 
%   Copyright 1993-1998 The MathWorks, Inc.  All Rights Reserved. 
%   $Revision: 1.5 $  $Date: 1997/11/24 15:35:13 $ 
 
if isa(img, 'double') 
   d = img;  
elseif isa(img, 'uint8') 
   if nargin==1 
      if islogical(img)        % uint8 binary image 
         d = double(img); 
      else                   % uint8 intensity image 
         d = double(img)/255; 
      end 
   elseif nargin==2 
      if ~ischar(typestr) || (typestr(1) ~= 'i') 
         error('Invalid input arguments'); 
      else  
         d = double(img)+1; 
      end 
   else 
      error('Invalid input arguments.'); 
   end 
else 
   error('Unsupported input class.'); 
end 
end