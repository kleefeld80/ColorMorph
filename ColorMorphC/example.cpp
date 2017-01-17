#include <stdio.h>
#include <cmath>
#include <iostream>

#include "loewner_morphology.h"

void check_results_details(float *array1, float *array2, int size, bool printDifferences) {
	int count = 0;
	float max = 0.0;
	float sum = 0.0;

	for (int ix = 0; ix < size; ++ix) {
		float diff = std::abs(array1[ix] - array2[ix]);
		
		if (diff >= FLT_EPSILON) {
			if (diff > max) max = diff;

			if (printDifferences) {
				printf("Results differ at element %d\n", ix);
				printf("array1 value: %lf\n", array1[ix]);
				printf("array2 value: %lf\n", array2[ix]);
			}

			++count;
			sum += diff;
		}
	}

	printf("Number of errors: %d\n", count);
	printf("Max difference: %f\n", max);

	if (count != 0) {
		printf("Average difference: %f\n", sum / count);
	}
}

// test program
int main(int argc, char **argv) {
	if (argc != 4) {
		printf("Illegal number of command line arguments. 3 required.\n");
		printf("USAGE: ./example [image file] [structuring element dimension] [structuring element file]\n");
		exit(EXIT_FAILURE);
	}
	
	LoewnerMorphology::Morph morph(argv[1], argv[3], atoi(argv[2]), 8);
	
	morph.displayOriginalImage();
	
	float *test = NULL;	
	
	std::cout << "EROSION" << std::endl;
	morph.erosion();
	test = morph.returnResult();
	morph.displayResultImage();
	CImgFloatWrapper img1("TestImages/andreas_lena_erosion.tiff");
	check_results_details(img1.data(), test, img1.width() * img1.height() * 3, false);
	delete test;

	std::cout << "DILATION" << std::endl;
	morph.dilation();
	test = morph.returnResult();
	morph.displayResultImage();
	CImgFloatWrapper img2("TestImages/andreas_lena_dilation.tiff");
	check_results_details(img2.data(), test, img2.width() * img2.height() * 3, false);
	delete test;

	std::cout << "CLOSING" << std::endl;
	morph.closing();
	test = morph.returnResult();
	morph.displayResultImage();
	CImgFloatWrapper img3("TestImages/andreas_lena_closing.tiff");
	check_results_details(img3.data(), test, img3.width() * img2.height() * 3, false);
	delete test;

	std::cout << "OPENING" << std::endl;
	morph.opening();
	test = morph.returnResult();
	morph.displayResultImage();
	CImgFloatWrapper img4("TestImages/andreas_lena_opening.tiff");
	check_results_details(img4.data(), test, img3.width() * img2.height() * 3, false);
	delete test;
	
	std::cout << "BLACK TOP HAT" << std::endl;
	morph.blackTopHat();
	test = morph.returnResult();
	morph.displayResultImage();
	CImgFloatWrapper img5("TestImages/andreas_lena_black_top.tiff");
	check_results_details(img5.data(), test, img5.width() * img5.height() * 3, false);
	delete test;
	
	std::cout << "WHITE TOP HAT" << std::endl;
	morph.whiteTopHat();
	test = morph.returnResult();
	morph.displayResultImage();
	CImgFloatWrapper img6("TestImages/andreas_lena_white_top.tiff");
	check_results_details(img6.data(), test, img6.width() * img6.height() * 3, false);
	delete test;
	
	std::cout << "BEUCHER GRADIENT" << std::endl;
	morph.beucherGradient();
	test = morph.returnResult();
	morph.displayResultImage();
	CImgFloatWrapper img8("TestImages/andreas_lena_beucher.tiff");
	check_results_details(img8.data(), test, img8.width() * img8.height() * 3, false);
	delete test;
	
	std::cout << "SELF DUAL TOP HAT" << std::endl;
	morph.selfDualTopHat();
	test = morph.returnResult();
	morph.displayResultImage();
	CImgFloatWrapper img9("TestImages/andreas_lena_sdth.tiff");
	check_results_details(img9.data(), test, img9.width() * img9.height() * 3, false);
	delete test;

	std::cout << "EXTERNAL GRADIENT" << std::endl;
	morph.externalGradient();
	test = morph.returnResult();
	morph.displayResultImage();
	CImgFloatWrapper img10("TestImages/andreas_lena_external.tiff");
	check_results_details(img10.data(), test, img10.width() * img10.height() * 3, false);
	delete test;

	std::cout << "INTERNAL GRADIENT" << std::endl;
	morph.internalGradient();
	test = morph.returnResult();
	morph.displayResultImage();
	CImgFloatWrapper img11("TestImages/andreas_lena_internal.tiff");
	check_results_details(img11.data(), test, img11.width() * img11.height() * 3, false);
	delete test;

	std::cout << "LAPLACIAN" << std::endl;
	morph.laplacian();
	test = morph.returnResult();
	morph.displayResultImage();
	CImgFloatWrapper img12("TestImages/andreas_lena_laplacian.tiff");
	check_results_details(img12.data(), test, img12.width() * img12.height() * 3, false);
	delete test;

	std::cout << "SHOCKFILTER" << std::endl;
	morph.shockFilter();
	test = morph.returnResult();
	morph.displayResultImage();
	CImgFloatWrapper img13("TestImages/andreas_lena_shock.tiff");
	check_results_details(img13.data(), test, img13.width() * img13.height() * 3, false);
	delete test;

	std::cout << "SHOCKFILTER (50 iterations)" << std::endl;
	
	morph.shockFilter(50);
	morph.displayResultImage();

	return 0;
}

