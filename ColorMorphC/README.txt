Welcome and thank you for using the CPU library for morphological operations on color images based on Loewner order and Einstein addition!

Basic requirements:
- g++ compiler (4.8 or higher)
- imagemagick (sudo apt-get install imagemagick php5-imagick)
- graphicsmagick (sudo apt-get install graphicsmagick)
- CImg library (already in this folder)

The folder contains the following:
	include – folder containing header files
	lib – folder containing library files
	src – folder containing source files
	ResultImages – folder containing result images
	TestImages – folder containing original images
	StructuringElements – folder containing files where several structuring elements are saved
	example.cpp – example program, compares results with the result images 
	loewner.cpp – simple program for running morphological operations (to be explained)
	loewner_morphology.h – basic header file
	README.txt – you are reading it
	Makefile

PLEASE DO NOT DELETE OR MOVE ANY OF THE FILES IN THIS FOLDER TO BE SURE EVERYTHING IS WORKING LIKE IT SUPPOSE TO!

For the installation, type the following:
	make install

During that action the following source files should be compiled:
	CimgFloatWrapper.cpp
	morph_color_matrix.cpp
	morph_circle.cpp
	morph_smallest_circle_mask.cpp
	einstein_operations.cpp
	openImage.cpp

Now you are ready for executing first examples.

Please type the following:
	make example

To execute the program a few command line arguments should be provided:
USAGE: ./example [image file] [structuring element dimension] [structuring element file]

Please type the following:
	./example TestImages/lena.tiff 3 StructuringElements/square3.txt

After that, all supported morphological operations should be executed on Lena image using 3 x 3 square structuring element (you should except the maximum average error between 1 and 1.5, that is alright).

Now, we can build the simple program for performing morphological operations:
	make loewner

USAGE: ./loewner [image file] [mask dimension] [mask file] [operation] [iterations] [result file]

The operation is determined by one of the following codes:
0) dilation
1) erosion
2) closing
3) opening
4) black top hat
5) white top hat
6) self dual top hat
7) beucher gradient
8) external gradient
9) internal gradient
10) morphological laplacian
11) shock filter

Example: ./loewner TestImages/lena.tiff 3 StructuringElements/square3.txt 11 5 test.tiff

The result is stored in the file test.tiff. To see it, open the image with your favorite image viewer, or type the following:
	./open_image test.tiff

To write your own programs please look at the source files example.cpp and loewner.cpp. Moreover, all header files in the include folder are well commented, you can find the documentation for all classes and methods in their header files.

To compile your own code, open the make file and replace “my_code” entry in variable SOURCE and EXECUTABLE with wanted information in the following part:

SOURCE=my_code.cpp
EXECUTABLE=my_code

run:
	$(CC) $(SOURCE) $(EXAMPLE_FLAGS) -o $(EXECUTABLE)

After that, the code can be compiled with the following:
	make run

Also, if you want to compile your code which is outside the LoewnerMorpholgy folder, just copy the Makefile in the folder you want to use it,find the entry DIR on the beginning of the Makefile and replace the current directory with the required path to the LoewnerMorphology folder.

To remove installed static libraries, just type make clean.

Enjoy using the library!

Best wishes,
Filip Srnec 
Andreas Kleefeld
