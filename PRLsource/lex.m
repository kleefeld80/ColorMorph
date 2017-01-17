% Copyright by Andreas Kleefeld
% Last updated 12/11/2013
function lex(filenamein,filenameout,choice,iter)
    
    % create the tensorfield
    % read the size of the image (N and M)
    img=im2double(imread(filenamein));
    [N,M,~]=size(img);
    fprintf('Filename:%s %d %d\n',filenamein,N,M);
    field2=zeros(N,M,3);   
    field=rgb2hsi(img);
    
    fprintf('Choice=%d, Iter=%d\n',choice,iter);
    if choice==1
        for i=1:iter
            field2=dilation(field);
            field=field2;
        end
    elseif choice==2
        for i=1:iter
            field2=erosion(field);
            field=field2;
        end
    elseif choice==3
        field2=opening(field); 
    elseif choice==4
        field2=closing(field);  
    else
        field2=field;
    end
    
    img=hsi2rgb(field2);
    img2=zeros(N,M,3,'uint8');
    
    for i=1:N
        for j=1:M
            img2(i,j,1)=uint8(img(i,j,1)*255.0);
            img2(i,j,2)=uint8(img(i,j,2)*255.0);
            img2(i,j,3)=uint8(img(i,j,3)*255.0);
        end
    end
    imagesc(img2);
    imwrite(img2,filenameout);
end
 
function p=myordermax(points)
    % Search for the maximum in the third component
    % S-Saturation with alpha=10
    maxi=max(ceil(points(:,3)*255/10));
    index=find(ceil(points(:,3)*255/10)==maxi);
    if length(index)==1
        p=points(index,:);
    else
        % Search for the maximum in the second component
        % I-Itensity
        remaining=points(index,:);
        maxi=max(remaining(:,2));
        index=find(remaining(:,2)==maxi);
        if length(index)==1
            p=remaining(index,:);
        else
            % Search for the minimum in the first component
            remaining2=remaining(index,:);
            mini=min(prozent(remaining2(:,1)));
            index=find(prozent(remaining2(:,1))==mini);
            if length(index)==1
                p=remaining2(index,:);
            else
                % Check a fourth condition
                p=remaining2(index(1),:);
            end
        end
    end
end

function p=myordermin(points)
    % Search for the minimum in the third component
    % S-Saturation with alpha=10
    maxi=min(ceil(points(:,3)*255/10));
    index=find(ceil(points(:,3)*255/10)==maxi);
    if length(index)==1
        p=points(index,:);
    else
        % Search for the minimum in the second component
        % I-Itensity
        remaining=points(index,:);
        maxi=min(remaining(:,2));
        index=find(remaining(:,2)==maxi);
        if length(index)==1
            p=remaining(index,:);
        else
            % Search for the maximum in the first component
            remaining2=remaining(index,:);
            mini=max(prozent(remaining2(:,1)));
            index=find(prozent(remaining2(:,1))==mini);
            if length(index)==1
                p=remaining2(index,:);
            else
                % Check a fourth condition
                p=remaining2(index(1),:);
            end
        end
    end
end

function out=prozent(in)
    out=in;
    for i=1:size(in,1)
        if abs(in(i))<0.5
            out(i)=abs(in(i));
        else
            out(i)=1-abs(in(i));
        end
    end
end

function out=dilation(in)
    global mask
    
    [N,M,P]=size(in);
    out=zeros(N,M,P);
    
    mask2=repmat(mask,[1,1,3]);
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
            window=in(indi,indj,:);
            points=reshape(window(mask2>0),size(window(mask2>0),1)/3,3);
            
            out(i,j,:)=myordermax(points);
        end
    end
end

function out=erosion(in)
    global mask
    
    [N,M,P]=size(in);
    out=zeros(N,M,P);
    
    mask2=repmat(mask,[1,1,3]);
    length=(size(mask)-1)/2;
    %count=1
    for i=1:N
        for j=1:M
            % get the points according to the mask
            inds=j-length:j+length;
            len=size(inds(inds>M),2);
            indj=[abs(inds(inds<=0))+1,inds(inds>=1 & inds<=M),M*ones(1,len)+1-(1:len)];
            inds=i-length:i+length;
            len=size(inds(inds>N),2);
            indi=[abs(inds(inds<=0))+1,inds(inds>=1 & inds<=N),N*ones(1,len)+1-(1:len)];
            window=in(indi,indj,:);
            points=reshape(window(mask2>0),size(window(mask2>0),1)/3,3);
            
            out(i,j,:)=myordermin(points);
        end
    end
end

function out=opening(in)
    out=dilation(erosion(in));
end

function out=closing(in)
    out=erosion(dilation(in));
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

function hsi = rgb2hsi(rgb)
%RGB2HSI Converts an RGB image to HSI.
%   HSI = RGB2HSI(RGB) converts an RGB image to HSI. The input image
%   is assumed to be of size M-by-N-by-3, where the third dimension
%   accounts for three image planes: red, green, and blue, in that
%   order. If all RGB component images are equal, the HSI conversion
%   is undefined. The input image can be of class double (with values
%   in the range [0, 1]), uint8, or uint16. 
%
%   The output image, HSI, is of class double, where:
%     hsi(:, :, 1) = hue image normalized to the range [0, 1] by
%                    dividing all angle values by 2*pi. 
%     hsi(:, :, 2) = saturation image, in the range [0, 1].
%     hsi(:, :, 3) = intensity image, in the range [0, 1].

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.5 $  $Date: 2005/01/18 13:44:59 $

% Extract the individual component images.
r = rgb(:, :, 1);
g = rgb(:, :, 2);
b = rgb(:, :, 3);

% Implement the conversion equations.
num = 0.5*((r - g) + (r - b));
den = sqrt((r - g).^2 + (r - b).*(g - b));
theta = acos(num./(den + eps));

H = theta;
H(b > g) = 2*pi - H(b > g);
H = H/(2*pi);

num = min(min(r, g), b);
den = r + g + b;
den(den == 0) = eps;
S = 1 - 3.* num./den;

H(S == 0) = 0;

I = (r + g + b)/3;

% Combine all three results into an hsi image.
hsi = cat(3, H, S, I);
end
function rgb = hsi2rgb(hsi)
%HSI2RGB Converts an HSI image to RGB.
%   RGB = HSI2RGB(HSI) converts an HSI image to RGB, where HSI is
%   assumed to be of class double with:  
%     hsi(:, :, 1) = hue image, assumed to be in the range
%                    [0, 1] by having been divided by 2*pi.
%     hsi(:, :, 2) = saturation image, in the range [0, 1].
%     hsi(:, :, 3) = intensity image, in the range [0, 1].
%
%   The components of the output image are:
%     rgb(:, :, 1) = red.
%     rgb(:, :, 2) = green.
%     rgb(:, :, 3) = blue.

%   Copyright 2002-2004 R. C. Gonzalez, R. E. Woods, & S. L. Eddins
%   Digital Image Processing Using MATLAB, Prentice-Hall, 2004
%   $Revision: 1.5 $  $Date: 2003/10/13 01:01:06 $

% Extract the individual HSI component images.
H = hsi(:, :, 1) * 2 * pi;
S = hsi(:, :, 2);
I = hsi(:, :, 3);

% Implement the conversion equations.
R = zeros(size(hsi, 1), size(hsi, 2));
G = zeros(size(hsi, 1), size(hsi, 2));
B = zeros(size(hsi, 1), size(hsi, 2));

% RG sector (0 <= H < 2*pi/3).
idx = find( (0 <= H) & (H < 2*pi/3));
B(idx) = I(idx) .* (1 - S(idx));
R(idx) = I(idx) .* (1 + S(idx) .* cos(H(idx)) ./ ...
                                          cos(pi/3 - H(idx)));
G(idx) = 3*I(idx) - (R(idx) + B(idx));

% BG sector (2*pi/3 <= H < 4*pi/3).
idx = find( (2*pi/3 <= H) & (H < 4*pi/3) );
R(idx) = I(idx) .* (1 - S(idx));
G(idx) = I(idx) .* (1 + S(idx) .* cos(H(idx) - 2*pi/3) ./ ...
                    cos(pi - H(idx)));
B(idx) = 3*I(idx) - (R(idx) + G(idx));

% BR sector.
idx = find( (4*pi/3 <= H) & (H <= 2*pi));
G(idx) = I(idx) .* (1 - S(idx));
B(idx) = I(idx) .* (1 + S(idx) .* cos(H(idx) - 4*pi/3) ./ ...
                                           cos(5*pi/3 - H(idx)));
R(idx) = 3*I(idx) - (G(idx) + B(idx));

% Combine all three results into an RGB image.  Clip to [0, 1] to
% compensate for floating-point arithmetic rounding effects.
rgb = cat(3, R, G, B);
rgb = max(min(rgb, 1), 0);
end
