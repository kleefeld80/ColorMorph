#include "../include/morph_library.h"

LoewnerMorphology::MorphColorMatrix LoewnerMorphology::Morph::morph_basic_operation(Circle *start, int pWidth, int *mask, int padding, bool type) {
	int n = 2 * padding + 1;
	
	MorphSmallestCircleProblemMask scp(start, mask, n, n, pWidth);

	Circle result = scp.compute();

	if (type) {
		result.returnMin();
	} else {
		result.returnMax();
	}
 
	return result.toMorphColorMatrixCone2Epsilon();
}

void LoewnerMorphology::Morph::readMaskFromFile(int *maskPointer, int maskDim, const char *fileName) {
	FILE *file = NULL;
	
	open_file(file, fileName, "r");

	for (int i = 0, n = maskDim * maskDim; i < n; i++) {
		if (fscanf(file, "%d", maskPointer + i) != 1) {
			printf("Error while reading file %s.\n", fileName);
			exit(EXIT_FAILURE);
		}
	}	

	close_file(file, fileName);
}

void LoewnerMorphology::Morph::print_vector(Circle *in, int width, int height, int lda) {
	Circle *current = in;

	for(int i = 0; i < height; i++) {
		for(int j = 0; j < width; j++) {
			current[j].print(); printf(" ");
		}
		printf("\n");

        	current += lda;
   	}
}

void LoewnerMorphology::Morph::print_vector(float *in, int width, int height, int lda) {
        float *current = in;
        
	for(int i = 0; i < height; i++) {
        	for(int j = 0; j < width; j++) {
                	printf("%5.2f ", current[j]);
        	}
        	printf("\n");

        	current += lda;
	}
}

void LoewnerMorphology::Morph::print_vector(LoewnerMorphology::MorphColorMatrix *in, int width, int height, int lda) {
        MorphColorMatrix *current = in;
        for(int i = 0; i < height; i++) {
		for(int j = 0; j < width; j++) {
                    current[j].printMorphColorMatrix(); printf(" ");
            	}
            printf("\n");

            current += lda;
        }
}

LoewnerMorphology::Morph::Morph(const char *imageFile, const char *maskFile, int maskDim, int numberOfThreads) {
	if (maskDim % 2 == 0 || maskDim * maskDim > 1024) {
        	printf("Mask dimension should be odd and its squere should be less than 1024, but %d provided.\n", maskDim);
        	exit(EXIT_FAILURE);
	}

	if (numberOfThreads <= 0) {
		printf("Number of threads must be a positive integer, but %d provided.\n", numberOfThreads);
	}
 	
	mask = new int[maskDim * maskDim];

	readMaskFromFile(mask, maskDim, maskFile);

	padding = maskDim / 2;

	omp_set_num_threads(numberOfThreads);
	
	inputImage = new CImgFloatWrapper(imageFile);
	outputImage = nullptr;

	width = inputImage->width();
	height = inputImage->height();
	spectrum = inputImage->spectrum();	
	size = width * height;

	double *image = new double[size * spectrum];
	double *data = new double [size * spectrum];
	
	matrices = new LoewnerMorphology::MorphColorMatrix[size];
	result = new LoewnerMorphology::MorphColorMatrix[size];	

	Conversions::type2double(inputImage->data(), image, size * spectrum);
	Conversions::rgb2mhcl(image, image + size, image + 2 * size, data, data + size, data + 2 * size, size);
	Conversions::mhcl2matrix(data, data + size, data + 2 * size, matrices, size);
	
	delete image;
	delete data;
}

LoewnerMorphology::Morph::~Morph() {
	delete mask;
	
	delete matrices;
	delete result;

	delete inputImage;
	delete outputImage;
}

void LoewnerMorphology::Morph::createOutputImage() {
	double *image = new double[size * spectrum];
	double *data = new double[size * spectrum];	
	float *out = new float[size * spectrum];

	Conversions::matrix2mhcl(result, data, data + size, data + 2 * size, size);
	Conversions::mhcl2rgb(data, data + size, data + 2 * size, image, image + size, image + 2 * size, size);
	Conversions::double2type(image, out, size * spectrum);
    
	if (outputImage != nullptr) {
		delete outputImage;
	}
	
	outputImage = new CImgFloatWrapper(out, width, height, spectrum);
	
	delete image;
	delete data;
	delete out;
}

void LoewnerMorphology::Morph::dilation(int iter) {
	morph_handle(matrices, result, width, height, padding, mask, 0, iter);
	createOutputImage();
}

void LoewnerMorphology::Morph::erosion(int iter) {
	morph_handle(matrices, result, width, height, padding, mask, 1, iter);
	createOutputImage();
}

void LoewnerMorphology::Morph::closing(int iter) {
	morph_handle(matrices, result, width, height, padding, mask, 2, iter);
	createOutputImage();
}

void LoewnerMorphology::Morph::opening(int iter) {
	morph_handle(matrices, result, width, height, padding, mask, 3, iter );
	createOutputImage();
}

void LoewnerMorphology::Morph::blackTopHat(int iter) {
	morph_handle(matrices, result, width, height, padding, mask, 4, iter);
	createOutputImage();
}

void LoewnerMorphology::Morph::whiteTopHat(int iter) {
	morph_handle(matrices, result, width, height, padding, mask, 5, iter);
	createOutputImage();
}

void LoewnerMorphology::Morph::selfDualTopHat(int iter) {
	morph_handle(matrices, result, width, height, padding, mask, 6, iter);
	createOutputImage();
}

void LoewnerMorphology::Morph::beucherGradient(int iter) {
	morph_handle(matrices, result, width, height, padding, mask, 7, iter);
	createOutputImage();
}

void LoewnerMorphology::Morph::externalGradient(int iter) {
	morph_handle(matrices, result, width, height, padding, mask, 8, iter);
	createOutputImage();
}

void LoewnerMorphology::Morph::internalGradient(int iter) {
	morph_handle(matrices, result, width, height, padding, mask, 9, iter);
	createOutputImage();
}

void LoewnerMorphology::Morph::laplacian(int iter) {
	morph_handle(matrices, result, width, height, padding, mask, 10, iter);
	createOutputImage();
}

void LoewnerMorphology::Morph::shockFilter(int iter) {
	morph_handle(matrices, result, width, height, padding, mask, 11, iter);
	createOutputImage();
}

void LoewnerMorphology::Morph::displayOriginalImage() {
	inputImage->display();
}

void LoewnerMorphology::Morph::displayResultImage() {
	if (outputImage == nullptr) {
		printf("There is no result to display.\n");
		return;
	}

	outputImage->display();
}

float *LoewnerMorphology::Morph::returnResult() {
	if (outputImage == nullptr) {
            printf("There is no result to return.\n");
            return nullptr;
    	}

	float *out = new float[size * spectrum];	
	copy(outputImage->data(), out, size * spectrum);

	return out;
}

void LoewnerMorphology::Morph::saveResult(const char *fileName) {
    if (outputImage == nullptr) {
            printf("There is no result to save.\n");
            return;
    }

    outputImage->save(fileName);
}
