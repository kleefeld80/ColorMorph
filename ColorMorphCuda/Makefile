example: example.cu
	nvcc example.cu --prec-div=false --prec-sqrt=false -std=c++11 -O2  -lpthread -I/usr/X11R6/include -L/usr/X11R6/lib -lm -lX11 -L./lib -lcimgwrapper -Xcompiler -fopenmp -o example
example1: example1.cu
	nvcc example1.cu --prec-div=false --prec-sqrt=false -std=c++11 -O2  -lpthread -I/usr/X11R6/include -L/usr/X11R6/lib -lm -lX11 -L./lib -lcimgwrapper -Xcompiler -fopenmp -o example1
morph_open: openImage.cpp
	g++ openImage.cpp -std=c++11 -O2  -lpthread -I/usr/X11R6/include -L/usr/X11R6/lib -lm -lX11 -L.. -lcimgwrapper -o open
install: ./src/CImgFloatWrapper.cpp
	g++ -o CImgFloatWrapper.o -c ./src/CImgFloatWrapper.cpp -std=c++11 -O2 -lm -lpthread -I/usr/X11R6/include -L/usr/X11R6/lib -lm -lX11
	ar rcs ./lib/libcimgwrapper.a CImgFloatWrapper.o
	rm CImgFloatWrapper.o
