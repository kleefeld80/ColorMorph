#include "../include/CImgFloatWrapper.h"
#include <stdio.h>

int main(int argc, char **argv) {
	if (argc != 2) {
		printf("Please provide the name of the image file.\n");
		exit(EXIT_FAILURE);
	}

	CImgFloatWrapper(argv[1]).display();
	
	return 0;
}
