% Copyright by Andreas Kleefeld
% Last updated 12/11/2013
function Einsteindiff()
    img=im2double(imread('bilder/ex4dilation11x11x1.png'));
    R=img(:,:,1);
    G=img(:,:,2);
    B=img(:,:,3);
    
    img2=im2double(imread('bilder/ex4dilation3x3x6.png'));
    R2=img2(:,:,1);
    G2=img2(:,:,2);
    B2=img2(:,:,3);
    
    % create the tensorfield
    % read the size of the image (N and M)
    [N,M,~]=size(img);
    field=zeros(N,M,3);
    field2=zeros(N,M,3);
    field3=zeros(N,M,3);
    
    % rgb2hclm
    % hclm2point
    for i=1:N
        for j=1:M
            Rd=R(i,j);
            Gd=G(i,j);
            Bd=B(i,j);
            [Hd,chroma,Lm]=rgb2hclm(Rd,Gd,Bd);
            field(i,j,:)=hclm2point(Hd,chroma,Lm);
            Rd2=R2(i,j);
            Gd2=G2(i,j);
            Bd2=B2(i,j);
            [Hd2,chroma2,Lm2]=rgb2hclm(Rd2,Gd2,Bd2);
            field3(i,j,:)=hclm2point(Hd2,chroma2,Lm2);
        end
    end
    
    field2=EinsteinAdditionMod(field,field3,1,-1);
    
    img2=zeros(N,M,3,'uint8');
    % point2hclm
    % hclm2rgb
    count=0;
    for i=1:N
        for j=1:M
            [Hd,chroma,Lm]=point2hclm(field2(i,j,:));
            [Rd,Gd,Bd]=hclm2rgb(Hd,chroma,Lm);
            if Bd>1 || Gd>1 || Rd>1 || Bd<0 || Gd<0 || Rd<0
                count=count+1;
            end
            img2(i,j,1)=uint8(Rd*255.0);
            img2(i,j,2)=uint8(Gd*255.0);
            img2(i,j,3)=uint8(Bd*255.0);
        end
    end
    fprintf('Count=%d\n',count);
    imagesc(img2);
    imwrite(img2,'bilder/difference.png');
end

function out=EinsteinAdditionMod(in1,in2,fak1,fak2)    
    [N,M,~]=size(in1);
    out=zeros(size(in1));
    
    %counter1=1;
    %counter2=1;
    %counter3=1;
    for i=1:N
        for j=1:M
            % Get the two points
            p1=squeeze(in1(i,j,:))';
            p2=squeeze(in2(i,j,:))';
            
            % convert points from HCL~ to K_1
            p1=DCone2Sphere(p1(1),p1(2),p1(3))*p1;
            p2=DCone2Sphere(p2(1),p2(2),p2(3))*p2;
            
            % create the two (symmetric) matrices
            A=sign(fak1)*point2matrix(p1);
            B=sign(fak2)*point2matrix(p2);
            
            % Calculate the alphas 
            alphaA=sqrt(1-trace(A*A)); 
            if abs(imag(alphaA))>0
                alphaA=real(alphaA);
            end
            alphaB=sqrt(1-trace(B*B));
            if abs(imag(alphaB))>0
                alphaB=real(alphaB);
            end
            % Check the special cases
            if alphaA<1e-5 && alphaB<1e-5
                %counter1=counter1+1;
                if norm(A+B)<1e-7
                    mat=zeros(2,2); %EinsteinAddition(A,A);    % Achtung
                else
                    mat=EinsteinAddition(A,A);
                end
            elseif alphaA<1e-5
                %disp(sprintf('%d Only alphaA zero',counter2))
                %counter2=counter2+1;
                mat=EinsteinAddition(A,A);
            elseif alphaB<1e-5
                %disp(sprintf('%d Only alphaB zero',counter3))
                %counter3=counter3+1;
                mat=EinsteinAddition(B,B);
            else
                AM=(alphaA*B+alphaB*A)/(alphaA+alphaB);
                mat=EinsteinAddition(AM,AM);
            end
            
            % convert the result (a matrix) back to the point
            p=matrix2point(mat);
           
            % convert points from K_1 to HCL~
            p=Sphere2DCone(p(1),p(2),p(3))*p;
            
            % store the result
            out(i,j,:)=p;
        end
    end
end

function mat=EinsteinAddition(A,B)    
    trAB=trace(A*B);
    if abs(trAB+1)<1e-7
        disp('I hope I never get here')
        mat=zeros(2,2);
    else
        betaA=sqrt(1-trace(A*A));
        if abs(imag(betaA)) >0 % sometimes (due to rounding errors) which
            mat=A; % means betaA=0 => result is A
        else
            mat=(A+betaA*B+trAB/(1+betaA)*A)./(1+trAB);
        end
    end
end

function [Hd,chroma,Lm]=rgb2hclm(Rd,Gd,Bd)
    hsl=rgb2hsl([Rd,Gd,Bd]);
    Hd=hsl(1);
    Ld=hsl(3);
    chroma=max([Rd,Gd,Bd])-min([Rd,Gd,Bd]);
    Lm=2*Ld-1;
end

function [Rd,Gd,Bd]=hclm2rgb(Hd,chroma,Lm)
    Ld=(Lm+1)/2;
    if chroma<1e-4 
        Sd=0;
    else
        Sd=chroma/(1-abs(2*Ld-1));
    end
    
    rgb=hsl2rgb([Hd,Sd,Ld]);
    Rd=rgb(1);
    Gd=rgb(2);
    Bd=rgb(3);
end

function hsl=rgb2hsl(rgb)

%Converts Red-Green-Blue Color value to Hue-Saturation-Luminance Color value
%
%Usage
%       HSL = rgb2hsl(RGB)
%
%   converts RGB, a M X 3 color matrix with values between 0 and 1
%   into HSL, a M X 3 color matrix with values between 0 and 1
%
%See also hsl2rgb, rgb2hsv, hsv2rgb

%Suresh E Joel, April 26,2003

if nargin<1,
    error('Too few arguements for rgb2hsl');
    return;
elseif nargin>1,
    error('Too many arguements for rgb2hsl');
    return;
end;

if max(max(rgb))>1 || min(min(rgb))<0,
    error('RGB values have to be between 0 and 1');
    return;
end;

for i=1:size(rgb,1),
    mx=max(rgb(i,:));%max of the 3 colors
    mn=min(rgb(i,:));%min of the 3 colors
    imx=find(rgb(i,:)==mx);%which color has the max
    hsl(i,3)=(mx+mn)/2;%luminance is half of max value + min value
    if(mx-mn)==0,%if all three colors have same value, 
        hsl(i,2)=0;%then s=0 and 
        hsl(i,1)=0;%h is undefined but for practical reasons 0
        return;
    end;
    if hsl(i,3)<0.5,
        hsl(i,2)=(mx-mn)/(mx+mn);
    else
        hsl(i,2)=(mx-mn)/(2-(mx+mn));
    end;
    switch(imx(1))%if two colors have same value and be the maximum, use the first color
    case 1 %Red is the max color
        hsl(i,1)=((rgb(i,2)-rgb(i,3))/(mx-mn))/6;
    case 2 %Green is the max color
        hsl(i,1)=(2+(rgb(i,3)-rgb(i,1))/(mx-mn))/6;
    case 3 %Blue is the max color
        hsl(i,1)=(4+(rgb(i,1)-rgb(i,2))/(mx-mn))/6;
    end;
    if hsl(i,1)<0,hsl(i,1)=hsl(i,1)+1;end;%if hue is negative, add 1 to get it within 0 and 1
end;

hsl=round(hsl*100000)/100000; %Sometimes the result is 1+eps instead of 1 or 0-eps instead of 0 ... so to get rid of this I am rounding to 5 decimal places)
end

function rgb=hsl2rgb(hsl_in)
%Converts Hue-Saturation-Luminance Color value to Red-Green-Blue Color value
%
%Usage
%       RGB = hsl2rgb(HSL)
%
%   converts HSL, a M [x N] x 3 color matrix with values between 0 and 1
%   into RGB, a M [x N] X 3 color matrix with values between 0 and 1
%
%See also rgb2hsl, rgb2hsv, hsv2rgb

% (C) Vladimir Bychkovsky, June 2008
% written using: 
% - an implementation by Suresh E Joel, April 26,2003
% - Wikipedia: http://en.wikipedia.org/wiki/HSL_and_HSV

hsl=reshape(hsl_in, [], 3);

H=hsl(:,1);
S=hsl(:,2);
L=hsl(:,3);

lowLidx=L < (1/2);
q=(L .* (1+S) ).*lowLidx + (L+S-(L.*S)).*(~lowLidx);
p=2*L - q;
hk=H; % this is already divided by 360

t=zeros([length(H), 3]); % 1=R, 2=B, 3=G
t(:,1)=hk+1/3;
t(:,2)=hk;
t(:,3)=hk-1/3;

underidx=t < 0;
overidx=t > 1;
t=t+underidx - overidx;
    
range1=t < (1/6);
range2=(t >= (1/6) & t < (1/2));
range3=(t >= (1/2) & t < (2/3));
range4= t >= (2/3);

% replicate matricies (one per color) to make the final expression simpler
P=repmat(p, [1,3]);
Q=repmat(q, [1,3]);
rgb_c= (P + ((Q-P).*6.*t)).*range1 + ...
        Q.*range2 + ...
        (P + ((Q-P).*6.*(2/3 - t))).*range3 + ...
        P.*range4;
       
rgb_c=round(rgb_c.*10000)./10000; 
rgb=reshape(rgb_c, size(hsl_in));
end

% Check OK!
function p=hclm2point(H,C,Lm)
    factor=1; %sqrt(2); deleted the factor
    x=C*cos(2*pi*H)/factor;  % included a factor
    y=C*sin(2*pi*H)/factor;
    z=Lm/factor;
    
    p=[x,y,z];
end

% Check OK!
function mat=point2matrix(p)
    x=p(1);
    y=p(2);
    z=p(3);
    a=z-y;
    b=x;
    c=z+y;
    mat=[a/sqrt(2),b/sqrt(2);b/sqrt(2),c/sqrt(2)];
end

% Check OK!
function p=matrix2point(mat)
    a=mat(1,1);
    b=mat(1,2);
    c=mat(2,2);
    x=sqrt(2)*b;
    y=(c-a)/sqrt(2);
    z=(c+a)/sqrt(2);
    p=[x,y,z];
end

% Check OK!
function [H,C,Lm]=point2hclm(p)
    x=p(1);
    y=p(2);
    z=p(3);
    C=hypot(x,y);
    at=atan2(y,x);
    if at<0
        at=at+2*pi;
    end
    H=at/(2*pi);
    Lm=z; 
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

% Calculates the factor for the conversion from
% doublecone to sphere with verzerrung
% n=10
function a=DCone2Sphere(x,y,z)
    a=1+(hypot(x,y)+abs(z))^10*(1/warpfactor(x,y,z)-1);
end

% Calculates the factor for the conversion from
% sphere to doublecone with verzerrung
% n=10
function a=Sphere2DCone(x,y,z)
    cache=1-DCone2Sphere(x,y,z);
    if abs(cache)<1e-4 % avoid the problem with the
        a=1;           % north and south pole
    else
        r=1./roots([1,-1,0,0,0,0,0,0,0,0,0,cache]);
        a=r(1);
    end
end

% Calculates the original warp factor 
% (see notes of Andreas Kleefeld)
function d=warpfactor(x,y,z)
    sqrtx2y2=hypot(x,y);
    if sqrtx2y2==0 % in this case we have d=1
        d=1;
    else
        tanphi=abs(z)/sqrtx2y2;
        d=sqrt(1+tanphi*tanphi)/(1+tanphi);
    end
end