#include <stdio.h>

#include "loewner_morphology.h"

// test program
int main(int argc, char **argv) {
        if (argc != 7) {
                printf("Illegal number of command line arguments. 6 required.\n");
              	printf("USAGE: ./morph [image file] [mask dimension] [mask file] [operation] [iterations] [result file]\n");
                exit(EXIT_FAILURE);
        }

	LoewnerMorphology::Morph morph(argv[1], argv[3], atoi(argv[2]));
	
	morph.displayOriginalImage();
	
	int type = atoi(argv[4]);
	int iter = atoi(argv[5]);

	switch(type) {
		case 0:
			morph.dilation(iter);
			break;
		case 1:
			morph.erosion(iter);
			break;
		case 2:
			morph.closing(iter);
			break;
		case 3:
			morph.opening(iter);
			break;
		case 4:
			morph.blackTopHat(iter);
			break;
		case 5:
			morph.whiteTopHat(iter);
			break;
		case 6:
			morph.selfDualTopHat(iter);
			break;
		case 7:
			morph.beucherGradient(iter);
			break;
		case 8:
			morph.externalGradient(iter);
			break;
		case 9:
			morph.internalGradient(iter);
			break;
		case 10:
			morph.laplacian(iter);
			break;
		case 11:
			morph.shockFilter(iter);
			break;
		default:
			printf("Operation %d not recognized.\n");
			break;
	}

	morph.displayResultImage();
	morph.saveResult(argv[6]);

        return 0;
}

