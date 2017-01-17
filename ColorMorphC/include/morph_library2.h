#ifndef MORPH_LIBRARY_H
#define MORPH_LIBRARY_H

#include "CImgFloatWrapper.h"
#include "loewner_declaration.h"
#include "conversions.h"
#include "morph_color_matrix.h"
#include "morph_circle.h"
#include "morph_smallest_circle_mask.h"
#include "einstein_operations.h"

#include "omp.h"

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <errno.h>
#include <sys/time.h>
#include <string.h>

#define open_file(file_ptr, file_name, mode)\
	do {\
		if (((file_ptr) = fopen((file_name), (mode))) == NULL) {\
			fprintf(stderr, "%s:%d: error while opening file %s: %s\n", __FILE__, __LINE__, (file_name), strerror(errno));\
			exit(EXIT_FAILURE);\
		}\
	} while(0)

#define close_file(file_ptr, file_name)\
	do {\
		if (fclose((file_ptr)) != 0) {\
			fprintf(stderr, "%s:%d: error while closing file %s: %s\n", __FILE__, __LINE__, (file_name), strerror(errno));\
			exit(EXIT_FAILURE);\
		}\
	} while(0)

/*
 *==============================================================================================================
 * Class that contains morphological operations which are introduced in the paper of B. Burgeth and A. Kleefeld
 *==============================================================================================================
 */
class LoewnerMorphology::Morph {

	public:
		/*
		 * Constructor of the class Morph. It takes the name of the file where the image is stored, the name of the file where a 
		 * structuring element (a mask) is stored and dimension of the structuring elements as arguments.
		 */
		Morph(const char *imageFile, const char *maskFile, int maskDim, int numberOfThreads = 8);

		/*
		 * Destructor of the class Morph.
		 */
		~Morph();

		// MORPHOLOGICAL OPERATION

		/*
	  	 * Performs morphological opration dilation on the input image.
		 */
		void dilation(int iter = 1);

		/*
		 * Performs morphological operation erosion on the input image.
		 */
		void erosion(int iter = 1);
		
		/*
		 * Performs morphological operation closing on the input image.
		 */
		void closing(int iter = 1);

		/*
		 * Performs morphological operation opening on the input image.
		 */
		void opening(int iter = 1);

		/*
		 * Performs morphological operation black top hat on the input image.
		 */
		void blackTopHat(int iter = 1); 

		/*
		 * Performs morphological operation white top hat on the input image.
		 */
		void whiteTopHat(int iter = 1); 
				
		/*
		 * Performs morphological operation self-dual top hat on the input image.
		 */
		void selfDualTopHat(int iter = 1); 
			
		/*
		 * Performs morphological operation beucher gradient on the input image.
		 */
		void beucherGradient(int iter = 1); 	

		/*
		 * Performs morphological operation internal gradient on the input image.
		 */
		void externalGradient(int iter = 1); 
			
		/*
		 * Performs morphological operation internal gradient on the input image.
		 */
		void internalGradient(int iter = 1);

		/*
		 * Performs morphological operation morphological laplacian on the input image.
		 */
		void laplacian(int iter = 1);

		/*
		 * Performs morphological operation shock filter on the input image.
		 */
		void shockFilter(int iter = 1);		
			
		/*
		 * Displays the original image.
		 */
		void displayOriginalImage();
			
		/*
		 * Displays the result of the morphological operation if the operation was called.
		 */
		void displayResultImage();

		/*
		 * Returns the result image as an array of floats. It allocates memory equal to the size of the image times spectrum.
		 */
		float *returnResult();

		/*
	 	 * Saves the result image to the file which name is provided as a filename argument.
	 	 */
		void saveResult(const char *fileName);
		
	private:
		using Circle = LoewnerMorphology::MorphCircle;
		
		CImgFloatWrapper *inputImage;	// input image
		CImgFloatWrapper *outputImage;	// output image - after morphological operation

		int *mask;	// mask array
		int padding;	// mask padding
	
		LoewnerMorphology::MorphColorMatrix *matrices;	// input image converted to array of MorphColorMatrix objects
		LoewnerMorphology::MorphColorMatrix *result;	// result image converted to array of MorphColorMatrix objects

		int width;	// width of the image
		int height;	// height of the image
		int spectrum;	// spectrum of the image
		int size;	// size of the image

		// HANDLERS

		/*
		 * Method that performes modified commutative Einstein subtraction of two images that are given on memory lications image1 and 
		 * image2. Both images have the size width * hight, with respective leading dimensions, lda1 and lda2. Precisely, the operation
		 * image3 = image1 - image2 is perfomed. The result is stored on memory location image3, with leading dimension lda3. 
		 */
		template<typename T>
		static void morph_einstein_launcher(T *image1, T *image2, T *imageResult, int width, int height, int lda1, int lda2, int lda3); 
		
		/*
		 * The method responsible for calculating morphological operations on a (2 * padding + 1)-dimensional squared matrix stored on 
		 * memory location start. Argument pWidth is the appropriate lda matrix lda. The matrix contains MorphCircle objects as elements. 
		 * The calculation is performed using the approach presented in the paper, based on Loewner order. The method returns the result 
		 * of the wanted morphological operation in form of MorphColorMatrix.
		 *
		 * Argument type determines a morphological operation:
		 *      1) false -> DILATION
		 *      2) true -> EROSION
		 */
		static MorphColorMatrix morph_basic_operation(Circle *start, int pWidth, int *mask, int padding, bool type);
	
		/*
		 * The method responsible for invoking calculations needed for performing a basic morphological operation on given image vector 
		 * which has already beend prepared for calculations in the form of an array of Circle objects. The input vector inPrepared is 
		 * expected to be size of (2 * padding + width) * (2 * padding + height). The original image is stored as an array of objects 
		 * of type T on memory location in. Argument padding is a padding of the structuring element used in the morphological operation,
		 * which is pased as an integer matrix stored on memory location mask with both dimensions (2 * padding + 1). For example, if a
		 * mask has dimensions 5x5, the padding is 2. Output vector's size has to be width * height.
		 * Morphological operations are performed as explained in the paper, elements are compared using Loewner order, solving the
		 * smallest enclosing circle of circles problem. Type T should support the conversion to the type MorphCircle. In the other words,		   * a constructor MorphCircle(&T) must exist. 
		 *
		 * Argument type determines a morphological operation:
		 *      1) false -> DILATION
		 *      2) true -> EROSION
		 */
		template<typename T>
		static void morph_basic_handler(Circle *inPrepared, T* in, T *out, int width, int height, int padding, int *mask, bool type);

		/*
		 * The method responsible for perfoming a wanted basic morphological operation on given image vector. The input vector in is 
		 * expected to be an image matrix containing objects of type T as elements. The vector containing the image matrix must have size
		 * width * height. Argument padding is a padding of the structuring element used in the morphological operation, which is pased as		   * an integer matrix stored on memory location mask with both dimensions (2 * padding + 1). For example, if a mask has dimensions
		 * 5x5, the padding is 2. Output vector's size has to be width * height. Morphological operations are performed as explained in
		 * the paper, elements are compared using Loewner order, solving the smallest enclosing circle of circles problem.
		 * Type T should support the conversion to the type MorphCircle. In the other words, a constructor MorphCircle(&T) must exist. 
		 *
		 * Argument type determines a morphological operation:
		 *      1) false -> DILATION
		 *      2) true -> EROSION
		 */
		template<typename T>
		static void morph_basic_launcher(T *in, T *out, int width, int height, int padding, int *mask, bool type);

		/*
		 * The method responsible for invoking calculations needed for performing a morphological operation on given image vector 
		 * which has already beend prepared for calculations in the form of an array of Circle objects. The input vector inPrepared is 
		 * expected to be size of (2 * padding + width) * (2 * padding + height). The original image is stored as an array of objects 
		 * of type T on memory location in. Argument padding is a padding of the structuring element used in the morphological operation,
		 * which is pased as an integer matrix stored on memory location mask with both dimensions (2 * padding + 1). For example, if a
		 * mask has dimensions 5x5, the padding is 2. Output vector's size has to be width * height.
		 * Morphological operations are performed as explained in the paper, elements are compared using Loewner order, solving the
		 * smallest enclosing circle of circles problem. Type T should support the conversion to the type MorphCircle. In the other words,		   * a constructor MorphCircle(&T) must exist. 
		 *
		 * Argument type determines a morphological operation:
		 *      1) false -> CLOSING
		 *      2) true -> OPENING
		 */
		template<typename T>
		static void morph_second_order_handler(Circle *inPrepared, T *in, T *out, int width, int height, int padding, int *mask, bool type);		
		/*
		 * The method responsible for perfoming a wanted basic morphological operation on given image vector. The input vector in is 
		 * expected to be an image matrix containing objects of type T as elements. The vector containing the image matrix must have size
		 * width * height. Argument padding is a padding of the structuring element used in the morphological operation, which is pased as		   * an integer matrix stored on memory location mask with both dimensions (2 * padding + 1). For example, if a mask has dimensions
		 * 5x5, the padding is 2. Output vector's size has to be width * height. Morphological operations are performed as explained in 
		 * the paper, elements are compared using Loewner order, solving the smallest enclosing circle of circles problem.
		 * Type T should support the conversion to the type MorphCircle. In the other words, constructor MorphCircle(&T) must exist. 
		 *
		 * Argument type determines a morphological operation:
		 *      1) false -> CLOSING
		 *      2) true -> OPENING
		 */
		template<typename T>
		static void morph_second_order_launcher(T *in, T *out, int width, int height, int padding, int *mask, bool type);
		
		/*
		 * The method responsible for invoking calculations needed for performing a morphological operation on given image vector 
		 * which has already beend prepared for calculations in the form of an array of Circle objects. The input vector inPrepared is 
		 * expected to be size of (2 * padding + width) * (2 * padding + height). The original image is stored as an array of objects 
		 * of type T on memory location in. Argument padding is a padding of the structuring element used in the morphological operation,
		 * which is pased as an integer matrix stored on memory location mask with both dimensions (2 * padding + 1). For example, if a
		 * mask has dimensions 5x5, the padding is 2. Output vector's size has to be width * height.
		 * Morphological operations are performed as explained in the paper, elements are compared using Loewner order, solving the
		 * smallest enclosing circle of circles problem. Type T should support the conversion to the type MorphCircle. In the other words,		   * a constructor MorphCircle(&T) must exist. 
		 *
		 * Argument type determines a morphological operation:
		 *      1) false -> BLACK TOP HAT
		 *      2) true -> WHITE TOP HAT
		 */
		template<typename T>
		static void morph_hats_handler(Circle *inPrepared, T *in, T *out, int width, int height, int padding, int *mask, bool type);		
		/*
		 * The method responsible for perfoming a wanted morphological operation on given image vector. The input vector in is 
		 * expected to be an image matrix containing objects of type T as elements. The vector containing the image matrix must have size
		 * width * height. Argument padding is a padding of the structuring element used in the morphological operation, which is pased as		   * an integer matrix stored on memory location mask with both dimensions (2 * padding + 1). For example, if a mask has dimensions
		 * 5x5, the padding is 2. Output vector's size has to be width * height. Morphological operations are performed as explained in 
		 * the paper, elements are compared using Loewner order, solving the smallest enclosing circle of circles problem.
		 * Type T should support the conversion to the type MorphCircle. In the other words, constructor MorphCircle(&T) must exist. 
		 *
		 * Argument type determines a morphological operation:
		 *      1) false -> BLACK TOP HAT
		 *      2) true -> WHITE TOP HAT
		 */ 
		template<typename T>
		static void morph_hats_launcher(T *in, T *out, int width, int height, int padding, int *mask, bool type);
	
		/*
		 * The method responsible for perfoming a morphological operation Beucher gradient on given image vector. The input vector in is 
		 * expected to be an image matrix containing objects of type T as elements. The vector containing the image matrix must have size
		 * width * height. Argument padding is a padding of the structuring element used in the morphological operation, which is pased as		   * an integer matrix stored on memory location mask with both dimensions (2 * padding + 1). For example, if a mask has dimensions
		 * 5x5, the padding is 2. Output vector's size has to be width * height. Morphological operations are performed as explained in 
		 * the paper, elements are compared using Loewner order, solving the smallest enclosing circle of circles problem.
		 * Type T should support the conversion to the type MorphCircle. In the other words, constructor MorphCircle(&T) must exist. 
		 */ 		
		template<typename T>
		static void morph_beucher_launcher(T *in, T *out, int width, int height, int padding, int *mask);					
		/*
		 * The method responsible for perfoming a morphological operation self-dual top hat on given image vector. The input vector in is 
		 * expected to be an image matrix containing objects of type T as elements. The vector containing the image matrix must have size
		 * width * height. Argument padding is a padding of the structuring element used in the morphological operation, which is pased as		   * an integer matrix stored on memory location mask with both dimensions (2 * padding + 1). For example, if a mask has dimensions
		 * 5x5, the padding is 2. Output vector's size has to be width * height. Morphological operations are performed as explained in 
		 * the paper, elements are compared using Loewner order, solving the smallest enclosing circle of circles problem.
		 * Type T should support the conversion to the type MorphCircle. In the other words, constructor MorphCircle(&T) must exist. 
		 */
		template<typename T>	
		static void morph_sdth_launcher(T *in, T *out, int width, int height, int padding, int *mask);
	
		/*
		 * The method responsible for invoking calculations needed for performing a morphological operation on given image vector 
		 * which has already beend prepared for calculations in the form of an array of Circle objects. The input vector inPrepared is 
		 * expected to be size of (2 * padding + width) * (2 * padding + height). The original image is stored as an array of objects 
		 * of type T on memory location in. Argument padding is a padding of the structuring element used in the morphological operation,
		 * which is pased as an integer matrix stored on memory location mask with both dimensions (2 * padding + 1). For example, if a
		 * mask has dimensions 5x5, the padding is 2. Output vector's size has to be width * height.
		 * Morphological operations are performed as explained in the paper, elements are compared using Loewner order, solving the
		 * smallest enclosing circle of circles problem. Type T should support the conversion to the type MorphCircle. In the other words,		   * a constructor MorphCircle(&T) must exist. 
		 *
		 * Argument type determines a morphological operation:
		 *      1) false -> EXTERNAL GRADIENT
		 *      2) true -> INTERNAL GRADIENT
		 */
		template<typename T>
		static void morph_gradients_handler(Circle *inPrepared, T *in, T *out, int width, int height, int padding, int *mask, bool type);		
		/*
		 * The method responsible for perfoming a wanted morphological operation on given image vector. The input vector in is 
		 * expected to be an image matrix containing objects of type T as elements. The vector containing the image matrix must have size
		 * width * height. Argument padding is a padding of the structuring element used in the morphological operation, which is pased as		   * an integer matrix stored on memory location mask with both dimensions (2 * padding + 1). For example, if a mask has dimensions
		 * 5x5, the padding is 2. Output vector's size has to be width * height. Morphological operations are performed as explained in 
		 * the paper, elements are compared using Loewner order, solving the smallest enclosing circle of circles problem.
		 * Type T should support the conversion to the type MorphCircle. In the other words, constructor MorphCircle(&T) must exist. 
		 *
		 * Argument type determines a morphological operation:
		 *      1) false -> EXTERNAL GRADIENT
		 *      2) true -> EXTERNAL GRADIENT
		 */ 
		template<typename T>
		static void morph_gradients_launcher(T *in, T *out, int width, int height, int padding, int *mask, bool type);
		
		/*
		 * The method responsible for perfoming a morphological operation laplacian on given image vector. The input vector in is 
		 * expected to be an image matrix containing objects of type T as elements. The vector containing the image matrix must have size
		 * width * height. Argument padding is a padding of the structuring element used in the morphological operation, which is pased as		   * an integer matrix stored on memory location mask with both dimensions (2 * padding + 1). For example, if a mask has dimensions
		 * 5x5, the padding is 2. Output vector's size has to be width * height. Morphological operations are performed as explained in 
		 * the paper, elements are compared using Loewner order, solving the smallest enclosing circle of circles problem.
		 * Type T should support the conversion to the type MorphCircle. In the other words, constructor MorphCircle(&T) must exist. 
		 */
		template<typename T>
		static void morph_laplacian_launcher(T *in, T *out, int width, int height, int padding, int *mask);
		
		/*
		 * The method responsible for perfoming a morphological operation shockfilter on given image vector. The input vector in is 
		 * expected to be an image matrix containing objects of type T as elements. The vector containing the image matrix must have size
		 * width * height. Argument padding is a padding of the structuring element used in the morphological operation, which is pased as		   * an integer matrix stored on memory location mask with both dimensions (2 * padding + 1). For example, if a mask has dimensions
		 * 5x5, the padding is 2. Output vector's size has to be width * height. Morphological operations are performed as explained in 
		 * the paper, elements are compared using Loewner order, solving the smallest enclosing circle of circles problem.
		 * Type T should support the conversion to the type MorphCircle. In the other words, constructor MorphCircle(&T) must exist. 
		 */
		template<typename T>
		static void morph_shock_launcher(T *in, T *out, int width, int height, int padding, int *mask);

		/*
		 * A method responsible for calculating morphological operation shokfilter. Arguments preparedDilation and preparedErosion 
		 * represent memory locations where two (2 * padding + width) * (2 * padding + height) matrices representing the original image 
		 * that has already been prepared for performing dilation and erosion operations, respectively. Both matrices contain MorphCircle
		 * objects as elements. Also, a morphological laplacian of the original image has been stored on memory location laplacian, as a 
		 * width * height matrix of objects T. The morphological shockfilter is performend as follows: if trace(laplacian[pixel]) < 0, a 
		 * dilation of the selected pixel is performed, else, a erosion of the selected pixel is performed. The result is stored on memory
		 * location out.
		 */
		template<typename T>
		static void morph_shock_operation(Circle *prepareDilation, Circle *prepareErosion, T *laplacian, T *out, int width, int height, int padding, int *mask);		

		/*
		 * A basic handler method for invoking launcher methods for performing all morphological operations which are introduced in the 
		 * paper of B. Burgeth and A. Kleefeld. Memory location in must contain the original image matrix with elements of type T. The 
		 * result of the selected morphological operation will be stored on memory location out. This memory location should be 
		 * preallocated to the size of width * height. Argument padding is a padding of the given structural element (maks). For example,
		 * if the structuring element has dimensions 5x5, the padding is 2. 
                 * Argument iters defines number of iterations.
		 * 
		 * The morphological operation is determined by morphType argument:
		 * 0) DILATION
		 * 1) EROSION
		 * 2) CLOSING
		 * 3) OPENING
		 * 4) BLACK TOP HAT
		 * 5) WHITE TOP HAT
		 * 6) SELF-DUAL TOP HAT
		 * 7) BEUCHER GRADIENT
		 * 8) EXTERNAL GRADIENT
		 * 9) INTERNAL GRADIENT
		 * 10) MORPHOLOGICAL LAPLACIAN
		 * 11) SHOCK FILTER
		 */
		template<typename T>
		static void morph_handle(T *in, T *out, int width, int height, int padding, int *mask, int morphType, int iters = 0);
		
		// HELPER METHODS		

		/*
		 * Helper method that creates the output image (CImgFloatWrapper object) after performing the morphological operation.
		 */
		void createOutputImage();

		/*
		 * Helper method for filling the array in with the given size with the given element alpha.
		 */
		template<typename T>
		static void fill(T *in, int size, T alpha);

		/*
		 * Helper method that prepares the image vector for morphological operations. The image vector is stored on memory location in.
		 * Its length should be width * height. The result is stored on the memory location out. Memory allocation should be done before
		 * calling this method. Out should be allocated to the size of (width + (2 * padding)) * (height + (2 * padding)) * sizeof(T)
		 * because a vector used in morphological operations should have an appropriate padding.
		 *
		 * Argument type determines a type of morphological operation that the vector needs to be prepared for:
		 *      1) false -> DILATION
		 *      2) true -> EROSION
		 */
		template<typename T>
		static void prepare_vector(T *in, Circle *out, int width, int height, int padding, bool type);
	
		/*
		 * Helper method for copying one array to another.
		 */
		template<typename T>
		static void copy(T *in, T *out, int size);
		
		/*
		 * Reading a structuring element (a mask) from a file specified by the given file name. Also, a mask dimension needs to be 
		 * provided. The Mask is expected to be a maskDim * maskDim matrix containing only 0s and 1s.
		 */
		static void readMaskFromFile(int *maskPointer, int maskDim, const char *fileName);

		// DEBUGGING

		/*
		 * Helper method for printing the given matrix of MorphCircle objects to the standard output.
		 * Used for debbugging.
		 */
		static void print_vector(Circle *in, int width, int height, int lda);
        
		/*
		 * Helper method for printing the given matrix of MorphColorMatrix objects to the standard output.
		 * Used for debbugging.
		 */
		static void print_vector(LoewnerMorphology::MorphColorMatrix *in, int width, int height, int lda);
	
		/*
		 * Helper method for printing the given matrix of floats to the standard output.
		 * Used for debbugging.
		 */
		static void print_vector(float *in, int width, int height, int lda);
};

template<typename T>
void LoewnerMorphology::Morph::fill(T *in, int size, T alpha) {
	#pragma omp parallel for
	for (int i = 0; i < size; i++) {
        	in[i] = alpha;
    	}
}

template<typename T>
void LoewnerMorphology::Morph::prepare_vector(T *in, Circle *out, int width, int height, int padding, bool type) {
	Circle element = (type) ? Circle(T::max()).prepareMin() : Circle(T::min()).prepareMax();
	fill<Circle>(out, (width + 2 * padding) * (height + 2 * padding), element);
	
	int pWidth = width + 2 * padding;
	
	#pragma omp parallel for
	for (int i = 0; i < height; i++) {       
        	for(int j = 0; j < width; j++) {
			out[(i + padding) * pWidth + (j + padding)] = (type) ? Circle(in[i * width + j]).prepareMin() : Circle(in[i * width + j]).prepareMax();
		}
	}
}

template<typename T>
void LoewnerMorphology::Morph::morph_basic_handler(Circle *inPrepared, T* in, T* out, int width, int height, int padding, int *mask, bool type) {
	int pWidth = width + 2 * padding;
	prepare_vector<T>(in, inPrepared, width, height, padding, type);

	#pragma omp parallel for
	for (int i = 0; i < height; i++) {
        	for(int j = 0; j < width; j++) {
            		Circle *current = inPrepared + i * pWidth + j;
            		out[i * width + j] = morph_basic_operation(current, pWidth, mask, padding, type);              
		}
    	}
}

template<typename T>
void LoewnerMorphology::Morph::morph_basic_launcher(T *in, T *out, int width, int height, int padding, int *mask, bool type) {
	Circle *inPrepared = new Circle[(width + (2 * padding)) * (height + (2 * padding))];
	morph_basic_handler(inPrepared, in, out, width, height, padding, mask, type);
	delete inPrepared;
}

template<typename T>
void LoewnerMorphology::Morph::morph_second_order_handler(Circle *inPrepared, T* in, T* out, int width, int height, int padding, int *mask, bool type) {
	morph_basic_handler(inPrepared, in, out, width, height, padding, mask, type);
	morph_basic_handler(inPrepared, out, out, width, height, padding, mask, !type);
}

template<typename T>
void LoewnerMorphology::Morph::morph_second_order_launcher(T *in, T *out, int width, int height, int padding, int *mask, bool type) { 
	Circle *inPrepared = new Circle[(width + (2 * padding)) * (height + (2 * padding))];
	morph_second_order_handler(inPrepared, in, out, width, height, padding, mask, type);
	delete inPrepared;
}

template<typename T>
void LoewnerMorphology::Morph::morph_einstein_launcher(T *image1, T *image2, T *imageResult, int width, int height, int lda1, int lda2, int lda3) {
	#pragma omp parallel for
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			LoewnerMorphology::MorphColorMatrix m1 = Circle(image1[i * lda1 + j]).toMorphColorMatrixSphere();
                	LoewnerMorphology::MorphColorMatrix m2 = Circle(image2[i * lda2 + j]).toMorphColorMatrixSphere().negate();

                	imageResult[i * lda3 + j] = Circle(EinsteinOperations::einsteinAdditionMod(m1, m2)).toMorphColorMatrixCone2Epsilon();
		}
	}
}

template<typename T>
void LoewnerMorphology::Morph::morph_hats_handler(Circle *inPrepared, T *in, T *out, int width, int height, int padding, int *mask, bool type) {
	T *temp = new T[width * height];

	morph_second_order_handler(inPrepared, in, temp, width, height, padding, mask, type);

	if (type) {
		morph_einstein_launcher(in, temp, out, width, height, width, width, width);
	} else {
		morph_einstein_launcher(temp, in, out, width, height, width, width, width);
	}

	delete temp;
}

template<typename T>
void LoewnerMorphology::Morph::morph_hats_launcher(T *in, T *out, int width, int height, int padding, int *mask, bool type) {	  
	Circle *inPrepared = new Circle[(width + (2 * padding)) * (height + (2 * padding))];
	morph_hats_handler(inPrepared, in, out, width, height, padding, mask, type);
	delete inPrepared;
}

template<typename T>
void LoewnerMorphology::Morph::morph_beucher_launcher(T *in, T *out, int width, int height, int padding, int *mask) {
	Circle *inPrepared = new Circle[(width + (2 * padding)) * (height + (2 * padding))];
	T *temp1 = new T[width * height];
	T *temp2 = new T[width * height];

	morph_basic_handler(inPrepared, in, temp1, width, height, padding, mask, false);
	morph_basic_handler(inPrepared, in, temp2, width, height, padding, mask, true);
	
	morph_einstein_launcher(temp1, temp2, out, width, height, width, width, width);
	
	delete inPrepared;
	delete temp1;
	delete temp2;
}

template<typename T>
void LoewnerMorphology::Morph::morph_sdth_launcher(T *in, T *out, int width, int height, int padding, int *mask) {
	Circle *inPrepared = new Circle[(width + (2 * padding)) * (height + (2 * padding))];
	T *temp1 = new T[width * height];
	T *temp2 = new T[width * height];

	morph_second_order_handler(inPrepared, in, temp1, width, height, padding, mask, false);
	morph_second_order_handler(inPrepared, in, temp2, width, height, padding, mask, true);
	
	morph_einstein_launcher(temp2, temp2, out, width, height, width, width, width);
	
	delete inPrepared;
	delete temp1;
	delete temp2;
}

template<typename T>
void LoewnerMorphology::Morph::morph_gradients_handler(Circle *inPrepared, T *in, T *out, int width, int height, int padding, int *mask, bool type) {
	T *temp = new T[width * height];

	morph_basic_handler(inPrepared, in, temp, width, height, padding, mask, type);

	if (type) {
		morph_einstein_launcher(in, temp, out, width, height, width, width, width);
	} else {
		morph_einstein_launcher(temp, in, out, width, height, width, width, width);
	}

	delete temp;
}

template<typename T>
void LoewnerMorphology::Morph::morph_gradients_launcher(T *in, T *out, int width, int height, int padding, int *mask, bool type) { 
	Circle *inPrepared = new Circle[(width + (2 * padding)) * (height + (2 * padding))];
	morph_gradients_handler(inPrepared, in, out, width, height, padding, mask, type);
	delete inPrepared;
}

template<typename T>
void LoewnerMorphology::Morph::morph_laplacian_launcher(T *in, T *out, int width, int height, int padding, int *mask) {
	Circle *inPrepared = new Circle[(width + (2 * padding)) * (height + (2 * padding))];
	T *temp1 = new T[width * height];
	T *temp2 = new T[width * height];

	morph_gradients_handler(inPrepared, in, temp1, width, height, padding, mask, false);
	morph_gradients_handler(inPrepared, in, temp2, width, height, padding, mask, true);
	
	morph_einstein_launcher(temp1, temp2, out, width, height, width, width, width);
	
	delete inPrepared;
	delete temp2;
	delete temp2;
}

template<typename T>
void LoewnerMorphology::Morph::morph_shock_launcher(T *in, T *out, int width, int height, int padding, int *mask) {
	T *laplacian = new T[width * height];
	Circle *dilationPrepared = new Circle[(width + (2 * padding)) * (height + (2 * padding))];
	Circle *erosionPrepared = new Circle[(width + (2 * padding)) * (height + (2 * padding))];
	
	prepare_vector<T>(in, dilationPrepared, width, height, padding, false);
	prepare_vector<T>(in, erosionPrepared, width, height, padding, true);
		
	morph_laplacian_launcher(in, laplacian, width, height, padding, mask);

	morph_shock_operation(dilationPrepared, erosionPrepared, laplacian, out, width, height, padding, mask);

	delete dilationPrepared;
	delete erosionPrepared;
	delete laplacian;
}

template<typename T>
void LoewnerMorphology::Morph::morph_shock_operation(Circle *dilationPrepared, Circle *erosionPrepared, T *laplacian, T *out, int width, int height, int padding, int *mask) {
	int pWidth = width + 2 * padding;

	#pragma omp parallel for
	for (int i = 0; i < height; i++) {
        	for(int j = 0; j < width; j++) {
			int currentIdx = i * pWidth + j;
            		
			if (laplacian[i * width + j].trace() <= 0) {
				out[i * width + j] = morph_basic_operation(dilationPrepared + currentIdx, pWidth, mask, padding, false);
			} else {
				out[i * width + j] = morph_basic_operation(erosionPrepared + currentIdx, pWidth, mask, padding, true);
			}			
		}
    	}	
}

template<typename T>
void LoewnerMorphology::Morph::morph_handle(T *in, T *out, int width, int height, int padding, int *mask, int morphType, int iters) {
	if (iters < 1) {
		printf("Operation cannot be executed. Number of iterations must be greater than 0, but %d provided.\n", iters);
		exit(EXIT_FAILURE);
	} 

	switch (morphType) {
		case 0:
			for (int i = 0; i < iters; i++) {
				if (i == 0) {
					morph_basic_launcher<T>(in, out, width, height, padding, mask, false);
				} else {
					morph_basic_launcher<T>(out, out, width, height, padding, mask, false);
				}
			}
			break;
		case 1:
			for (int i = 0; i < iters; i++) {
				if (i == 0) {
					morph_basic_launcher<T>(in, out, width, height, padding, mask, true);
				} else {
					morph_basic_launcher<T>(out, out, width, height, padding, mask, true);
				}
			}
			break;
		case 2:
		 	for (int i = 0; i < iters; i++) {
		 		if (i == 0) {
		 			morph_second_order_launcher<T>(in, out, width, height, padding, mask, false);
		 		} else {
		 			morph_second_order_launcher<T>(out, out, width, height, padding, mask, false);
		 		}
		 	}
		 	break;
		case 3:
		 	for (int i = 0; i < iters; i++) {
		 		if (i == 0) {
		 			morph_second_order_launcher<T>(in, out, width, height, padding, mask, true);
		 		} else {
		 			morph_second_order_launcher<T>(out, out, width, height, padding, mask, true);
		 		}
		 	}
		 	break;
		case 4: 
		 	for (int i = 0; i < iters; i++) {
		 		if (i == 0) {
		 			morph_hats_launcher<T>(in, out, width, height, padding, mask, false);
		 		} else {
		 			morph_hats_launcher<T>(out, out, width, height, padding, mask, false);
		 		}
		 	}
		 	break;
		case 5: 
		 	for (int i = 0; i < iters; i++) {
		 		if (i == 0) {
		 			morph_hats_launcher<T>(in, out, width, height, padding, mask, true);
		 		} else {
		 			morph_hats_launcher<T>(out, out, width, height, padding, mask, true);
				}
		 	}
		 	break;
		 case 6:
		 	for (int i = 0; i < iters; i++) {
		 		if (i == 0) {
		 			morph_sdth_launcher<T>(in, out, width, height, padding, mask);
		 		} else {
		 			morph_sdth_launcher<T>(out, out, width, height, padding, mask);
		 		}
		 	}
		 	break;
		case 7:
		 	for (int i = 0; i < iters; i++) {
		 		if (i == 0) {
		 			morph_beucher_launcher<T>(in, out, width, height, padding, mask);
		 		} else {
		 			morph_beucher_launcher<T>(out, out, width, height, padding, mask);
				}
		 	}
		 	break;
		case 8:
		 	for (int i = 0; i < iters; i++) {
		 		if (i == 0) {
		 			morph_gradients_launcher<T>(in, out, width, height, padding, mask, false);
		 		} else {
		 			morph_gradients_launcher<T>(out, out, width, height, padding, mask, false);
		 		}
		 	}
		 	break;
		 case 9:
		 	for (int i = 0; i < iters; i++) {
		 		if (i == 0) {
		 			morph_gradients_launcher<T>(in, out, width, height, padding, mask, true);
		 		} else {
		 			morph_gradients_launcher<T>(out, out, width, height, padding, mask, true);
		 		}
		 	}
		 	break;
		 case 10:
		 	for (int i = 0; i < iters; i++) {
		 		if (i == 0) {
		 			morph_laplacian_launcher<T>(in, out, width, height, padding, mask);
		 		} else {
		 			morph_laplacian_launcher<T>(out, out, width, height, padding, mask);
		 		}
		 	}
		 	break;		
		case 11:
		 	for (int i = 0; i < iters; i++) {
		 		if (i == 0) {
					morph_shock_launcher<T>(in, out, width, height, padding, mask);
		 		} else {
		 			morph_shock_launcher<T>(out, out, width, height, padding, mask);
		 		}
		 	}
		 	break;
	}
}

template<typename T>
void LoewnerMorphology::Morph::copy(T *in, T *out, int size) {
	#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		out[i] = in[i];
	}
}

#endif

