#ifndef LOEWNER_DECLARATION_H
#define LOEWNER_DECLARATION_H

/*
 * Declaring namespaces that constist of classes that are used for performing morphological operations on color images based
 * on Loewner ordering and Einstein addition from the paper of B. Burgeth and A. Kleefeld.
 * 
 * The classes are described in details in appropriate header files.
 */ 
namespace LoewnerMorphology {
	class MorphColorMatrix;
	
	class Conversions;

	class MorphCircle;
	
	class MorphSmallestCircleProblemMask;

	class EinsteinOperations;

	class Morph;

	// operators implemented in morph_color_matrix

	MorphColorMatrix __host__ __device__ operator-(const MorphColorMatrix &a, const MorphColorMatrix &b); 
	MorphColorMatrix __host__ __device__ operator+(const MorphColorMatrix &a, const MorphColorMatrix &b);
	MorphColorMatrix __host__ __device__ operator*(const double alpha, const MorphColorMatrix &a);
	bool __host__ __device__ operator>=(const MorphColorMatrix &a, const MorphColorMatrix &b); 
	bool __host__ __device__ operator<=(const MorphColorMatrix &a, const MorphColorMatrix &b);
	bool __host__ __device__ operator>(const MorphColorMatrix &a, const MorphColorMatrix &b); 
	bool __host__ __device__ operator<(const MorphColorMatrix &a, const MorphColorMatrix &b);
	bool __host__ __device__ operator==(const MorphColorMatrix &a, const MorphColorMatrix &b); 
	bool __host__ __device__ operator!=(const MorphColorMatrix &a, const MorphColorMatrix &b);
	
	// KERNELS USED FOR MORPHOLOGY OPERATIONS

	/*
	 * Kernel responsible for fillng given matrix with the given element. Argument width is width of the matrix stored on a memory location in.
	 * Grid for starting this kernel should be preconfigured using method prepareGrid1.
	 */
	template<typename T>
	static void __global__ fill(T *in, int width, T element);

	/*
	 * The kernel responsible for applying basic morphological operations (dilation and erosion) on color images based on the paper of B. Burgeth and A. Kleefeld.
	 * Input image vector converted to a type T is stored on the memory location in. The result image will be stored on location out. Pointers in and out must be device pointers.
	 * Arguments width and height are initial width and height of the image. Arguments pWidth and pHeight must be multiples of THREADS_X and THREADS_Y constants, respectively.
	 * Morpe precisely, pWidth = ((width + THREADS_X - 1) / THREADS_X) * THREADS_X and pHeight = ((height + THREADS_Y - 1) / THREADS_Y) * THREADS_Y. Padding is appropriate
	 * mask padding. It is calculated as mask dimension / 2 where / is integer division. The type T should support the conversion to the type MorphCircle. In the other words, 
	 * constructor MorphCircle(&T) must exist. The type T should support the conversion to the type MorphCircle. In the other words, constructor MorphCircle(&T) must exist. 
	 * Array in must be preallocated on size (pWidth + 2 * padding) * (pHeight + 2 * padding) and array out must be preallocated on size (width * height).
	 * Type of operation is determined by template argument type. If type if false, dilation is performed, otherwise, erosion is performed. The grid for starting this kernel
	 * should be preconfigured using method prepareGrid2.
	 * 
	 */
	template<typename T, bool type>
	static void __global__ morph_kernel(T *in, T *out, int width, int height, int pWidth, int pHeight, int padding);

	/*
	 * The kernel responsible for applying basic morphological operation shockfilter on color images based on the paper of B. Burgeth and A. Kleefeld.
	 * Input image vector converted to a type T is stored on the memory location in. The result image will be stored on location out. Pointers in and out must be device pointers.
	 * Arguments width and height are initial width and height of the image. Arguments pWidth and pHeight must be multiples of THREADS_X and THREADS_Y constants, respectively.
	 * Morpe precisely, pWidth = ((width + THREADS_X - 1) / THREADS_X) * THREADS_X and pHeight = ((height + THREADS_Y - 1) / THREADS_Y) * THREADS_Y. Padding is appropriate
	 * mask padding. It is calculated as mask dimension / 2 wher / is integer division. The type T should support the conversion to the type MorphCircle. In the other words, constructor MorphCircle(&T) must exist. 
	 * Array in must be preallocated on size (pWidth + 2 * padding) * (pHeight + 2 * padding) and array out must be preallocated on size (width * height). The grid for starting this kernel
	 * should be preconfigured using method prepareGrid2.
	 */
	template<typename T>
	void __global__ shock_kernel(T *in1, T *in2, T *out, T* laplacian, int width, int height, int pWidth, int pHeight, int padding);

	/*
	 * Kernel responsible for performing Einstein substraction of the two images in terms of Einstein addition between elements of type T explained in the paper of B. Burgeth
	 * and A. Kleefeld. Width and height are original width and height of the images and lda1 and lda2 are leading dimensions of each image matrices. Subtraction is performed
	 * elementwise, so T is expected to be type which allows Einstein addition (for example, MorphColorMatrix). Grid for starting this kernel should be preconfigured using method
	 * prepareGrid3.
	 */
	template<typename T>
	void __global__ einstein_kernel(T *image1, T *image2, int width, int height, int lda1, int lda2);

	// KERNELS USED FOR DEBUGGING
	
	/*
	 * Kernel for printing array of MorphColorMatricex objects to the standard output.
	 * Used for debugging.
	 */
	static void __global__ print_kernel(MorphColorMatrix *m, int size);

	/*
	 * Kernel for printing matrix of MorphColorMatrix objects to the standard output.                               
	 * Used for debugging.
	 */
	static void __global__ print_kernel(MorphColorMatrix *in, int width, int height, int lda);

	/*
	 * Helper device method responsible for perfomring morphological operation dilation on the GPU on the given (shared) memory location.
	 * Erosion is performed in terms of Loewner ordering described in the paper of B. Burgeth and A. Kleefeld solving Smallest enclosing circle of circles problem.
	 * Argument smemWidth is original width of the shared memory, and padding is mask padding (mask dimension / 2). It returns MorphColorMatrix object, the result of the dilation.
	 */
	static MorphColorMatrix __device__ dilate_gpu(MorphCircle *smemStart, int smemWidth, int padding);

	/*
	 * Helper device method responsible for perfomring morphological operation erosion on thi GPU on the given (shared) memory location.
	 * Erosion is performed in terms of Loewner ordering described in the paper of B. Burgeth and A. Kleefeld solving Smallest enclosing circle of circles problem.
	 * Argument smemWidth is original width of the shared memory, and padding is mask padding (mask dimension / 2). It returns MorphColorMatrix object, the result of the erosion.
	 */
	static MorphColorMatrix __device__ erode_gpu(MorphCircle *smemStart, int smemWidth, int padding);
};

#endif
