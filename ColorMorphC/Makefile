CC=g++
CFLAGS=-std=c++11 -O2 -lm -lpthread -fopenmp
DIR=.
#feel free to change DIR it if needed, there should be the root to LoewnerMorphology folder
LCIMG=-L$(DIR)/lib -lcimgwrapper
LLOE=-L$(DIR)/lib -lloewner
LX11=-I/usr/X11R6/include -L/usr/X11R6/lib -lX11

EXAMPLE_FLAGS=$(CFLAGS) $(LCIMG) $(LLOE) $(LX11)

#-----------------------------------------------------------------------------------------------------------------------------------------

install: 
	$(CC) -o CImgFloatWrapper.o -c ./src/CImgFloatWrapper.cpp $(CFLAGS) $(LX11)
	ar rcs ./lib/libcimgwrapper.a CImgFloatWrapper.o
	$(CC) -o morph_color_matrix.o -c ./src/morph_color_matrix.cpp $(CFLAGS) $(LCIMG) $(LX11)
	$(CC) -o morph_circle.o -c ./src/morph_circle.cpp $(CFLAGS) $(LCIMG) $(LX11)
	$(CC) -o morph_smallest_circle_mask.o -c ./src/morph_smallest_circle_mask.cpp $(CFLAGS) $(LCIMG) $(LX11)
	$(CC) -o einstein_operations.o -c ./src/einstein_operations.cpp $(CFLAGS) $(LCIMG) $(LX11)
	$(CC) -o morph_library.o -c ./src/morph_library.cpp $(CFLAGS) $(LCIMG) $(LX11)
	$(CC) ./src/openImage.cpp -o open_image $(CFLAGS) $(LCIMG) $(LX11)
	ar rcs ./lib/libloewner.a CImgFloatWrapper.o morph_color_matrix.o morph_circle.o morph_smallest_circle_mask.o einstein_operations.o morph_library.o
	rm CImgFloatWrapper.o morph_color_matrix.o morph_circle.o morph_smallest_circle_mask.o einstein_operations.o morph_library.o

clean:
	rm ./lib/libcimgwrapper.a ./lib/libloewner.a open_image

#-----------------------------------------------------------------------------------------------------------------------------------------

#Examples
example: example.cpp
	$(CC) example.cpp $(EXAMPLE_FLAGS) -o example
loewner: loewner.cpp
	$(CC) loewner.cpp $(EXAMPLE_FLAGS) -o loewner

#compile your own code: change "my_code" with your info
SOURCE=my_code.cpp
EXECUTABLE=my_code

my_code:
	$(CC) $(SOURCE) $(EXAMPLE_FLAGS) -o $(EXECUTABLE)


