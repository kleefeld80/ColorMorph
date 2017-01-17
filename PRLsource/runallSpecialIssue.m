% Copyright by Andreas Kleefeld
% Last updated 12/11/2013
function runallSpecialIssue()
    global mask
    
    % Example 1
    mask=ones(5,5);
    bas('bilderIn/gray.tiff','bilder/ex1dilation.png',1,1);
    bas('bilderIn/gray.tiff','bilder/ex1erosion.png',2,1);
    bas('bilderIn/gray.tiff','bilder/ex1closing.png',4,1);
    
    % Example 2
    mask=[0 1 0;...
          1 1 1;...
          0 1 0];
    loe('bilderIn/lenna.tiff','bilder/ex2wth.png',5,1);
    loe('bilderIn/lenna.tiff','bilder/ex2bth.png',6,1);
    loe('bilderIn/lenna.tiff','bilder/ex2sdth.png',7,1);
    loe('bilderIn/lenna.tiff','bilder/ex2beucher.png',8,1);
    loe('bilderIn/lenna.tiff','bilder/ex2internal.png',9,1);
    loe('bilderIn/lenna.tiff','bilder/ex2external.png',10,1);
    loe('bilderIn/lenna.tiff','bilder/ex2mlaplacian.png',11,1);
    loe('bilderIn/lenna.tiff','bilder/ex2shockfilter10x.png',13,10);
    loe('bilderIn/lenna.tiff','bilder/ex2shockfilter10x.png',13,100);
    mask=[0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
          0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0;
          0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0;
          0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0;
          0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0;
          0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0;
          0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0;
          0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0;
          0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0;
          0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0;
          1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;
          0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0;
          0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0;
          0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0;
          0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0;
          0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0;
          0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0;
          0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0;
          0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0;
          0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0;
          0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0];
    loe('bilderIn/lenna.tiff','bilder/ex2dilation.png',1,1);
    loe('bilderIn/lenna.tiff','bilder/ex2erosion.png',2,1);
    
    % Example 3
    mask=[0 1 0;...
          1 1 1;...
          0 1 0];
    loe('bilderIn/peppers.tiff','bilder/ex3beucher.png',8,1);
    loe('bilderIn/peppers.tiff','bilder/ex3internal.png',9,1);
    loe('bilderIn/peppers.tiff','bilder/ex3external.png',10,1);
    loe('bilderIn/peppers.tiff','bilder/ex3mlaplacian.png',11,1);
    com('bilderIn/peppers.tiff','bilder/ex3beucherCOMP.png',8,1);
    com('bilderIn/peppers.tiff','bilder/ex3internalCOMP.png',9,1);
    com('bilderIn/peppers.tiff','bilder/ex3externalCOMP.png',10,1);
    com('bilderIn/peppers.tiff','bilder/ex3mlaplacianCOMP.png',11,1);
    
    % Example 4
    mask=ones(11,11);
    loe('bilderIn/baboon.tiff','bilder/ex4dilation11x11x1.png',1,1);
    mask=ones(3,3);
    loe('bilderIn/baboon.tiff','bilder/ex4dilation3x3x6.png',1,6);
    Einsteindiff()
    
    % Example 5
    mask=ones(3,3);
    loe('bilderIn/house.tiff','bilder/ex5watershed.png',17,1);
    
    % Example 6
    mask=ones(15,15);
    loe('bilderIn/Ishihara.png','bilder/42dilation.png',1,1)
    loe('bilderIn/Ishihara.png','bilder/42erosion.png',2,1)
    loe('bilderIn/Ishihara.png','bilder/42opening.png',3,1)
    loe('bilderIn/Ishihara.png','bilder/42closing.png',4,1)
    lex('bilderIn/Ishihara.png','bilder/42dilationlex.png',1,1)
    lex('bilderIn/Ishihara.png','bilder/42erosionlex.png',2,1)
    lex('bilderIn/Ishihara.png','bilder/42openinglex.png',3,1)
    lex('bilderIn/Ishihara.png','bilder/42closinglex.png',4,1)
    
    % Example 7
    mask=ones(21,21);
    loe('bilderIn/color.tif','bilder/colordilation.png',1,1)
    lex('bilderIn/color.tif','bilder/colorlex.png',1,1)
    com('bilderIn/color.tif','bilder/colorcomp.png',1,1);
    loe('bilderIn/parrot.tiff','bilder/parrotdilation.png',1,1)
    lex('bilderIn/parrot.tiff','bilder/parrotlex.png',1,1)
    com('bilderIn/parrot.tiff','bilder/parrotcomp.png',1,1);
end