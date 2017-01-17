#include <stdio.h>

#include "loewner_morphology.h"
#include "include/CImgFloatWrapper.h"

// test program
int main(int argc, char **argv) {
        if (argc != 5) {
                printf("Illegal number of command line arguments. 4 required.\n");
              	printf("USAGE: ./morph [image file] [mask dimension] [mask file] [result file]\n");
                exit(EXIT_FAILURE);
        }

	LoewnerMorphology::Morph morph(argv[1], argv[3], atoi(argv[2]));
	
	morph.displayOriginalImage();
	
	float *test = NULL;	

	morph.dilation(1);
	test = morph.returnResult();
	morph.displayResultImage();
	CImgFloatWrapper img1("TestImages/andreas_lena_dilation.tiff");
	check_results_details(img1.data(), test, img1.width() * img1.height() * 3, false);
	free(test);
	
	test = NULL;

	morph.erosion();
	test = morph.returnResult();
	morph.displayResultImage();
	CImgFloatWrapper img2("TestImages/andreas_lena_erosion.tiff");
	check_results_details(img2.data(), test, img2.width() * img2.height() * 3, false);
	free(test);

	morph.shockFilter(50);
	morph.displayResultImage();
	morph.beucherGradient();
	morph.displayResultImage();
	morph.internalGradient();
	morph.displayResultImage();
	morph.externalGradient();
	morph.displayResultImage();

        return 0;
}

