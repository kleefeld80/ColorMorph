#include "../include/CImgFloatWrapper.h"
#include "../loewner_morphology.h"

#include "omp.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <thread>

#define THREADS_X 16
#define THREADS_Y 16

#define TILE_X 16
#define TILE_Y 16

#define BLOCK_SIZE 256

#define CONSTANT_SIZE 1024

#define OMP_THREAD_NUM 8
#define NUM_STREAMS 4

#define first(idx, n, dim)      (((idx) * (dim)) / (n))
#define last(idx, n, dim)       ((first(idx + 1, n, dim)) - 1)
#define size(idx, n, dim)       ((last(idx, n, dim)) - (first(idx, n, dim)) + 1)

int __constant__ maskMemory[CONSTANT_SIZE];	// constant memroy array
cudaStream_t streams[NUM_STREAMS];		// cuda streams

typedef LoewnerMorphology::MorphCircle Circle;

/*
 *==============================================================================================================
 * Class that contains morphological operations which are introduced in the paper of B. Burgeth and A. Kleefeld
 *==============================================================================================================
 */
class LoewnerMorphology::Morph {

	public:
		/*
		 * Constructor of the class Morph. It takes name of the file where image is stored, name of the file where mask (structural element) is stored
		 * and dimension of the mask as arguments.
		 */
		Morph(const char *imageFile, const char *maskFile, int maskDim);

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
		 * Displays original image.
	 	 */
		void displayOriginalImage();
		
		/*
         	 * Displays result of the morphological operation if the operation is called.
		 */
		void displayResultImage();

		/*
		 * Returns result image as an array of floats. It allocates memory equal to the size of the image times spectrum.
		 */
		float *returnResult();

		/*
		 * Saves result to the file which name is provided.
		 */
		void saveResult(const char *fileName);
		
	private:
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
		 * Invokes GPU kernel for performing modified Einstein subtraction of two given image vectors in device memory.
		 * Kernel is launched without further synchronization. Width and height are the dimensions of the original image. 
		 * Since image1 and image2 are image matrices stored in row-major vectorized format, one should provide lda for each of them.
		 */
		template<typename T>
		static void morph_einstein_async(T *image1, T *image2, int width, int height, int lda1, int lda2, cudaStream_t stream = 0); 
		
		/*
		 * Method that copies original image stored on memory location in to device pointer dev_out2, performs modified Einstein substraction
		 * between images stored on device on locations dev_out1 and dev_out2 and then copies the result to host on memory location out.
		 * Type of the subtraction is determined by template parameter type. If the type parameter is false, dev_out2 will be subtracted from
		 * dev_out1, if the type parameter is true, dev_out1 will be subtracted from dev_out2 in terms of Einstein subtraction. The operation
		 * is performed asynchronusely with N streams. Method does not do explicit synchronizations. 
		 */
		template<typename T, bool type, int N>
		static void morph_einstein_copy_original_launcher(T *dev_out1, T *dev_out2, T *in, T *out, int width, int height, cudaStream_t *streams);		

		/*
		 * Method modified Einstein substraction between images stored on device on locations dev_out1 and dev_out2 and then copies the result 
		 * to host on memory location out. More precisely, dev_out2 will be subtracted from dev_out2 will be subtracted from dev_out1 in terms of Einstein subtraction. The operation
		 * is performed asynchronusely with N streams. Method does not do explicit synchronizations.
		 */
		template<typename T, int N>
		static void morph_einstein_copy_launcher(T *dev_out1, T *dev_out2, T *out, int width, int height, cudaStream_t *streams);
		
		/*
		 * Method that invokes the kernels for basic morphological operations. Morphological operation is determined with template parameter type. If it is false, dilation
		 * is performed, in the other case, erosion is performed. Arguments dev_in and dev_out are device pointers. On the location dev_in there is 
		 * an array of size (pWidth + 2 * padding) * (pHeight + 2 * padding) containing image information with appropriate padding converthed to type T. Pointer dev_out contains preallocated
		 * memory for the result image. Size of the allocated memory is width * height where width and height are initial image dimensions. Also, pWidth and pHeight are expected to be calculated like this:
		 * pWidth = ((width + THREADS_X - 1) / THREADS_X) * THREADS_X and pHeight = ((height + THREADS_Y - 1) / THREADS_Y) * THREADS_Y. Padding is appropriate mask padding. 
		 * It is calculated as mask dimension / 2 where / is integer division. The type T should support the conversion to the type MorphCircle. In the other words, constructor MorphCircle(&T) must exist.
		 * Also, sheared memory size in bytes should be provided. On host memory location in, original image converted to type T must be stored.
		 */
		template<typename T, bool type>
		static void morph_basic_launcher(T *dev_in, T *dev_out, T *in, int width, int height, int pWidth, int pHeight, int padding, size_t sharedSize, cudaStream_t stream = 0);	
		
		/*
		 * Method that invokes the kernels for morphological operation shockfilter. Arguments dev_in and dev_out are device pointers. On the location dev_in there is 
		 * an array of size (pWidth + 2 * padding) * (pHeight + 2 * padding) containing image information with appropriate padding converthed to type T. Pointer dev_out contains preallocated
		 * memory for the result image. Size of the allocated memory is width * height where width and height are initial image dimensions. Also, pWidth and pHeight are expected to be calculated like this:
		 * pWidth = ((width + THREADS_X - 1) / THREADS_X) * THREADS_X and pHeight = ((height + THREADS_Y - 1) / THREADS_Y) * THREADS_Y. Padding is appropriate mask padding. 
		 * It is calculated as mask dimension / 2 where / is integer division. The type T should support the conversion to the type MorphCircle. In the other words, constructor MorphCircle(&T) must exist.
		 * Also, sheared memory size in bytes should be provided. On host memory location in, original image converted to type T must be stored. However, on device memory location laplacian, morphological
		 * laplacian of the original image has to be stored.
		 */
		template<typename T>
		static void morph_shock_launcher(T *dev_in1, T* dev_in2, T *dev_out, T *laplacian, T *in, int width, int height, int pWidth, int pHeight, int padding, size_t sharedSize, cudaStream_t stream = 0); 
		
		/*
		 * Invokes GPU kernel responsible for perfoming wanted basic morphological operation on given image vector. Input vector in is expected to be an image matrix containing objects of type T as elements. 
		 * The vector containing the image matrix must have size width * height. Argument padding is a padding of the given mask. For example, if mask has dimensions 5x5, padding is 2. 
		 * Output vector's size has to be width * height. Morphological operations are performed as explained in the paper, elements are compared using Loewner order, solving smallest circle problem.
		 * The type T should support the conversion to the type MorphCircle. In the other words, constructor MorphCircle(&T) must exist. 
		 *
		 * Template parameter type determines morphological operation:
		 *      1) false -> DILATION
		 *      2) true -> EROSION
		 */
		template<typename T, bool type>
		static void morph_basic(T *in, T *out, int width, int height, int padding);

		/*
		 * Invokes GPU kernel responsible for perfoming wanted basic morphological operation on given image vector. Input vector dev_in is expected to be an image matrix on GPU memory containing objects 
		 * of type T as elements. The vector containing the image matrix must have size size (pWidth + 2 * padding) * (pHeight + 2 * padding). Argument padding is a padding of the given mask. For example, 
		 * if mask has dimensions 5x5, padding is 2. Output vector's size has to be width * height. Morphological operations are performed as explained in the paper, elements are compared using Loewner order, 
		 * solving Smallest enclosing circle of circles problem. The type T should support the conversion to the type MorphCircle. In the other words, constructor MorphCircle(&T) must exist. 
		 * Also, shared memory size in bytes needs to be provided, as well as original image vector in host memory which elements are converted to type T on memory location in.
		 *
		 * Template parameter type determines morphological operation:
		 *      1) false -> CLOSING
		 *      2) true -> OPENING
		 */
		template<typename T, bool type>
		static void morph_second_order_launcher(T *dev_in, T *dev_out, T *in, int width, int height, int pWidth, int pHeight, int padding, size_t sharedSize, cudaStream_t stream = 0);

		/*
		 * Invokes GPU kernel responsible for perfoming wanted higher order morphological operation on given image vector. Input vector in is expected to be an image matrix containing objects of type T as elements. 
		 * The vector containing the image matrix must have size width * height. Argument padding is a padding of the given mask (structural element). For example, if mask has dimensions 5x5, padding is 2. 
		 * Output vector's size has to be width * height. Morphological operations are performed as explained in the paper, elements are compared using Loewner order, solving smallest circle problem.
		 * The type T should support the conversion to the type MorphCircle. In the other words, constructor MorphCircle(&T) must exist. 
		 *
		 * Template parameter type determines morphological operation:
		 *      1) false -> CLOSING
		 *      2) true -> OPENING
		 */
		template<typename T, bool type>
		static void morph_second_order(T *in, T *out, int width, int height, int padding);	

		/*
		 * Invokes GPU kernel responsible for perfoming morphological operations white top hat and black top hat on given image vector. Input vector in is expected to be an image matrix containing objects of type T as elements. 
		 * The vector containing the image matrix must have size width * height. Argument padding is a padding of the given mask (structural element). For example, if mask has dimensions 5x5, padding is 2. 
		 * Output vector's size has to be width * height. Morphological operations are performed as explained in the paper, elements are compared using Loewner order, solving smallest circle problem.
		 * The type T should support the conversion to the type MorphCircle. In the other words, constructor MorphCircle(&T) must exist. 
		 *
		 * Template parameter type determines morphological operation:
		 *      1) false -> BLACK TOP HAT
		 *      2) true -> WHITE TOP HAT
		 */
		template<typename T, bool type>
		static void morph_hats(T *in, T *out, int width, int height, int padding);
	
		/*
		 * Invokes GPU kernel responsible for perfoming morphological operation Beucher gradient on given image vector. Input vector in is expected to be an image matrix containing objects of type T as elements. 
		 * The vector containing the image matrix must have size width * height. Argument padding is a padding of the given mask (structural element). For example, if mask has dimensions 5x5, padding is 2. 
		 * Output vector's size has to be width * height. Morphological operations are performed as explained in the paper, elements are compared using Loewner order, solving smallest circle problem.
		 * The type T should support the conversion to the type MorphCircle. In the other words, constructor MorphCircle(&T) must exist. 
		 */
		template<typename T>
		static void morph_beucher(T *in, T *out, int width, int height, int padding, int *mask);					
			
		/*
		 * Invokes GPU kernel responsible for perfoming morphological operations self dual top hat on given image vector. Input vector in is expected to be an image matrix containing objects of type T as elements. 
		 * The vector containing the image matrix must have size width * height. Argument padding is a padding of the given mask (structural element). For example, if mask has dimensions 5x5, padding is 2. 
		 * Output vector's size has to be width * height. Morphological operations are performed as explained in the paper, elements are compared using Loewner order, solving smallest circle problem.
		 * The type T should support the conversion to the type MorphCircle. In the other words, constructor MorphCircle(&T) must exist. 
		 */
		template<typename T>	
		static void morph_sdth(T *in, T *out, int width, int height, int padding, int *mask);
	
		/*
		 * Invokes GPU kernel responsible for perfoming morphological operations internal gradient and external gradient on given image vector. Input vector in is expected to be an image matrix containing 
		 * objects of type T as elements. The vector containing the image matrix must have size width * height. Argument padding is a padding of the given mask (structural element). For example, if mask has 
		 * dimensions 5x5, padding is 2. Output vector's size has to be width * height. Morphological operations are performed as explained in the paper, elements are compared using Loewner order, solving 
		 * smallest circle problem. The type T should support the conversion to the type MorphCircle. In the other words, constructor MorphCircle(&T) must exist. 
		 *
		 * Template parameter type determines morphological operation:
		 *      1) false -> EXTERNAL GRADIENT
		 *      2) true -> INTERNAL GRADIENT
		 */
		template<typename T, bool type>
		static void morph_gradients(T *in, T *out, int width, int height, int padding);
		
		/*
		 * Invokes GPU kernel responsible for perfoming morphological operation morphological Laplacian. Input vector in is expected to be an image matrix containing objects of type T as elements. 
		 * The vector containing the image matrix must have size width * height. Argument padding is a padding of the given mask (structural element). For example, if mask has dimensions 5x5, padding is 2. 
		 * Output vector's size has to be width * height. Morphological operations are performed as explained in the paper, elements are compared using Loewner order, solving smallest circle problem.
		 * The type T should support the conversion to the type MorphCircle. In the other words, constructor MorphCircle(&T) must exist. 
		 */
		template<typename T>
		static void morph_laplacian(T *in, T *out, int width, int height, int padding, int *mask);
		
		/*
		 * Invokes GPU kernel responsible for perfoming morphological operation shock-filter Input vector in is expected to be an image matrix containing objects of type T as elements. 
		 * The vector containing the image matrix must have size width * height. Argument padding is a padding of the given mask (structural element). For example, if mask has dimensions 5x5, padding is 2. 
		 * Output vector's size has to be width * height. Morphological operations are performed as explained in the paper, elements are compared using Loewner order, solving smallest circle problem.
		 * The type T should support the conversion to the type MorphCircle. In the other words, constructor MorphCircle(&T) must exist. 
		 */
		template<typename T>
		static void morph_shock(T *in, T *out, int width, int height, int padding, int *mask);

		/*
		 * Basic handle for invoking launcher methods which are invoking GPU kernels for performing all morphological operations introduced in the paper from B. Burgeth and A. Kleefeld. Pointers in and out
		 * must be host pointer. Memory location in must contain original image matrix which elements are converted to the type T. Result of the selected morphological operation will be stored on memory location
		 * out. This memory location should be preallocated to the size of width * height. Argument padding is a padding of the given mask (structural element). For example, if mask has dimensions 5x5, padding is 2. 
                 * Argument iters defines number of iterations.
		 * 
		 * Morphological operation is determined by morphType argument:
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
		 * Helper method that creates output image (CImgFloatWrapper object) from an array of MorphColorMatrix objects result stored as a class variable.
		 */
		void createOutputImage();
	
		/*
		 * Helper method for copying one array to another.
		 */
		template<typename T>
		static void copy(T *in, T *out, int size);
		
		/*
		 * Helper method that prepares grid for fill kernel. 
		 */
		static inline void prepareGrid1(dim3 &gridDim, dim3 &blockDim, int height);

		/*
		 * Helper method that prepares grid for the morph kernel. Arguments pWidth and pHeight must be multiples of THREADS_X and THREADS_Y constants, respectively.
		 * Morpe precisely, pWidth = ((width + THREADS_X - 1) / THREADS_X) * THREADS_X and pHeight = ((height + THREADS_Y - 1) / THREADS_Y) * THREADS_Y.
		 */
		static inline void prepareGrid2(dim3 &gridDim, dim3 &blockDim, int pWidth, int pHeight);

		/*
		 * Helper method that prepares grid for einstein kernel.
		 */
		static inline void prepareGrid3(dim3 &gridDim, dim3 &blockDim, int width, int height);
		
		/*
		 * Reading mask from a file specified by given string. Also, mask dimension needs to be provided. Mask
		 * is expected to be a maskDim * maskDim matrix containing only 0 and 1.
		 */
		static void readMaskFromFile(int *maskPointer, int maskDim, const char *fileName);

		// DEBUGGING

		/*
		 * Helper method for printing matrix of MorphCircle objects to the standard output.
		 * Used for debbugging.
		 */
		static void __host__ __device__ print_shared_vector(Circle *in, int width, int height, int lda);
        
		/*
		 * Helper method for printing matrix of MorphColorMatrix objects to the standard output.
		 * Used for debbugging.
		 */
		static void __host__ __device__ print_shared_vector(LoewnerMorphology::MorphColorMatrix *in, int width, int height, int lda);
	
		/*
		 * Helper method for printing matrix of floats to the standard output.
		 * Used for debbugging.
		 */
		static void __host__ __device__ print_shared_vector(float *in, int width, int height, int lda);
};

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

void __host__ __device__ LoewnerMorphology::Morph::print_shared_vector(Circle *in, int width, int height, int lda) {
        Circle *current = in;
        for(int i = 0; i < height; i++) {
                for(int j = 0; j < width; j++) {
                        current[j].print(); printf(" ");
                }
                printf("\n");

                current += lda;
        }
}

void __host__ __device__ LoewnerMorphology::Morph::print_shared_vector(float *in, int width, int height, int lda) {
        float *current = in;
        
	for(int i = 0; i < height; i++) {
                for(int j = 0; j < width; j++) {
                        printf("%5.2f ", current[j]);
                }
                printf("\n");

                current += lda;
        }
}

void __host__ __device__ LoewnerMorphology::Morph::print_shared_vector(LoewnerMorphology::MorphColorMatrix *in, int width, int height, int lda) {
        MorphColorMatrix *current = in;
        for(int i = 0; i < height; i++) {
                for(int j = 0; j < width; j++) {
                        current[j].printMorphColorMatrix(); printf(" ");
                }
                printf("\n");

                current += lda;
        }
}

inline void LoewnerMorphology::Morph::prepareGrid1(dim3 &gridDim, dim3 &blockDim, int height) {
	blockDim.x = BLOCK_SIZE;
	blockDim.y = 1;
	gridDim.x = 1;
	gridDim.y = height;
}

inline void LoewnerMorphology::Morph::prepareGrid2(dim3 &gridDim, dim3 &blockDim, int pWidth, int pHeight) {
	blockDim.x = THREADS_X;
        blockDim.y = THREADS_Y;

        gridDim.x = pWidth / THREADS_X;
        gridDim.y = pHeight / THREADS_Y;	
}

inline void LoewnerMorphology::Morph::prepareGrid3(dim3 &gridDim, dim3 &blockDim, int width, int height) {
	blockDim.x = TILE_X;
        blockDim.y = TILE_Y;
	
	int pWidth = ((width + TILE_X - 1) / TILE_X) * TILE_X;  
        int pHeight = ((height + TILE_Y - 1) / TILE_Y) * TILE_Y;
	
        gridDim.x = pWidth / TILE_X;
        gridDim.y = pHeight / TILE_Y;	
}

template<typename T>
void LoewnerMorphology::Morph::morph_einstein_async(T *image1, T *image2, int width, int height, int lda1, int lda2, cudaStream_t stream) {
	dim3 blockDim;
	dim3 gridDim;
	
	prepareGrid3(gridDim, blockDim, width, height);
	LoewnerMorphology::einstein_kernel<T><<<gridDim, blockDim, 0, stream>>>(image1, image2, width, height, lda1, lda2);	
}

template<typename T, bool type, int N>
void LoewnerMorphology::Morph::morph_einstein_copy_original_launcher(T *dev_out1, T *dev_out2, T *in, T *out, int width, int height, cudaStream_t *streams) {  
	// calling einstein cernel asynchronusly with memory transfers
	#pragma unroll
	for (int i = 0; i < N; i++) {
		int first = first(i, N, height) * width;
		int chunkHeight = size(i, N, height);
		size_t size = chunkHeight * width * sizeof(T);

		cuda_exec(cudaMemcpyAsync(dev_out2 + first, in + first, size, cudaMemcpyHostToDevice, streams[i]));
		morph_einstein_async(((type) ? dev_out2 : dev_out1) + first, ((type) ? dev_out1 : dev_out2) + first, width, chunkHeight, width, width, streams[i]); 
		cuda_exec(cudaMemcpyAsync(out + first, ((type) ? dev_out2 : dev_out1) + first, size, cudaMemcpyDeviceToHost, streams[i]));
	}
}

template<typename T, int N>
void LoewnerMorphology::Morph::morph_einstein_copy_launcher(T *dev_out1, T *dev_out2, T *out, int width, int height, cudaStream_t *streams) {
	// calling einstein cernel asynchronusly with memory transfers
	#pragma unroll
	for (int i = 0; i < NUM_STREAMS; i++) {
		int first = first(i, NUM_STREAMS, height) * width;
		int chunkHeight = size(i, NUM_STREAMS, height);
		size_t size = chunkHeight * width * sizeof(T);

		morph_einstein_async(dev_out1 + first, dev_out2 + first, width, chunkHeight, width, width, streams[i]); 
		cuda_exec(cudaMemcpyAsync(out + first, dev_out1 + first, size, cudaMemcpyDeviceToHost, streams[i]));
	}
}

template<typename T, bool type>
void LoewnerMorphology::Morph::morph_basic_launcher(T *dev_in, T *dev_out, T *in, int width, int height, int pWidth, int pHeight, int padding, size_t sharedSize, cudaStream_t stream) {
        dim3 blockDim;
        dim3 gridDim;

        // filling device memory with appropriate value
        prepareGrid1(gridDim, blockDim, pHeight + 2 * padding);
        LoewnerMorphology::fill<T><<<gridDim, blockDim, 0, stream>>>(dev_in, pWidth + 2 * padding, (type) ? T::max() : T::min());

        // copying image to device memory
        cuda_exec(cudaMemcpy2D(dev_in + padding * (pWidth + 2 * padding) + padding, (pWidth + 2 * padding) * sizeof(T), in, width * sizeof(T), width * sizeof(T), height, cudaMemcpyHostToDevice));

        // invoking morph kernel
        prepareGrid2(gridDim, blockDim, pWidth, pHeight);
        LoewnerMorphology::morph_kernel<T, type><<<gridDim, blockDim, sharedSize, stream>>>(dev_in, dev_out, width, height, pWidth, pHeight, padding);
}

template<typename T>
void LoewnerMorphology::Morph::morph_shock_launcher(T *dev_in1, T* dev_in2, T *dev_out, T *laplacian, T *in, int width, int height, int pWidth, int pHeight, int padding, size_t sharedSize, cudaStream_t stream) {
	dim3 blockDim;
	dim3 gridDim;	

	// filling device memory with appropriate value
        prepareGrid1(gridDim, blockDim, pHeight + 2 * padding);
        LoewnerMorphology::fill<T><<<gridDim, blockDim, 0, stream>>>(dev_in1, pWidth + 2 * padding, T::min());
	LoewnerMorphology::fill<T><<<gridDim, blockDim, 0, stream>>>(dev_in2, pWidth + 2 * padding, T::max());
        cuda_exec(cudaStreamSynchronize(stream));

        // copying image to device memory
        cuda_exec(cudaMemcpy2D(dev_in1 + padding * (pWidth + 2 * padding) + padding, (pWidth + 2 * padding) * sizeof(T), in, width * sizeof(T), width * sizeof(T), height, cudaMemcpyHostToDevice));
        cuda_exec(cudaMemcpy2D(dev_in2 + padding * (pWidth + 2 * padding) + padding, (pWidth + 2 * padding) * sizeof(T), in, width * sizeof(T), width * sizeof(T), height, cudaMemcpyHostToDevice));

        // invoking morph kernela
        prepareGrid2(gridDim, blockDim, pWidth, pHeight);

        LoewnerMorphology::shock_kernel<T><<<gridDim, blockDim, 2 * sharedSize, stream>>>(dev_in1, dev_in2, dev_out, laplacian, width, height, pWidth, pHeight, padding);
        
	cuda_exec(cudaStreamSynchronize(stream));
}

template<typename T, bool type>
void LoewnerMorphology::Morph::morph_basic(T *in, T *out, int width, int height, int padding) {	  
	T *dev_in = NULL;       // device vector holding image matrix
        T *dev_out = NULL;     // device vector holding first output image

        int pWidth = ((width + THREADS_X - 1) / THREADS_X) * THREADS_X;         // width of the image (rounded to the multiple of THREAD_X)
        int pHeight = ((height + THREADS_Y - 1) / THREADS_Y) * THREADS_Y;       // hight of the image (rounded to the multiple of THREAD_Y)

        size_t inSize = (pWidth + 2 * padding) * (pHeight + 2 * padding) * sizeof(T);   // size of the input device vector (Circle is used for max and min calculations as described in the paper)
        size_t outSize = width * height * sizeof(T);                                    // size of the (each) output device vector
	
	int sharedWidth = THREADS_X + 2 * padding + 1;  // shared memory width (avoiding bank conflicts)
        int sharedHeight = THREADS_Y + 2 * padding;     // shared memory height
	size_t sharedSize = sharedWidth * sharedHeight * sizeof(Circle);	// size of the shared memory in bypes

	cuda_exec(cudaMalloc(&dev_in, inSize));
	cuda_exec(cudaMalloc(&dev_out, outSize));

	morph_basic_launcher<T, type>(dev_in, dev_out, in, width, height, pWidth, pHeight, padding, sharedSize);
	cuda_exec(cudaDeviceSynchronize());

	cuda_exec(cudaMemcpy(out, dev_out, outSize, cudaMemcpyDeviceToHost));
	
	cuda_exec(cudaFree(dev_in));
	cuda_exec(cudaFree(dev_out));
}

template<typename T, bool type>
void LoewnerMorphology::Morph::morph_second_order_launcher(T *dev_in, T *dev_out, T *in, int width, int height, int pWidth, int pHeight, int padding, size_t sharedSize, cudaStream_t stream) { 
	dim3 blockDim1, blockDim2;
	dim3 gridDim1, gridDim2;
	
	prepareGrid1(gridDim1, blockDim1, pHeight + 2 * padding);
	prepareGrid2(gridDim2, blockDim2, pWidth, pHeight);

	// filling device memory with appropriate value
	LoewnerMorphology::fill<T><<<gridDim1, blockDim1, 0, stream>>>(dev_in, pWidth + 2 * padding, (type) ? T::max() : T::min());

	// copying image to device memory
	cuda_exec(cudaMemcpy2DAsync(dev_in + padding * (pWidth + 2 * padding) + padding, (pWidth + 2 * padding) * sizeof(T), in, width * sizeof(T), width * sizeof(T), height, cudaMemcpyHostToDevice, stream));
	
	// invoking morph kernela
	LoewnerMorphology::morph_kernel<T, type><<<gridDim2, blockDim2, sharedSize, stream>>>(dev_in, dev_out, width, height, pWidth, pHeight, padding);
		
	// filling device memory with appropriate value
        LoewnerMorphology::fill<T><<<gridDim1, blockDim1, 0, stream>>>(dev_in, pWidth + 2 * padding, (!type) ? T::max() : T::min());
	
        // copying image to device memory
        cuda_exec(cudaMemcpy2DAsync(dev_in + padding * (pWidth + 2 * padding) + padding, (pWidth + 2 * padding) * sizeof(T), dev_out, width * sizeof(T), width * sizeof(T), height, cudaMemcpyDeviceToDevice, stream));

        // invoking morph kernela
        LoewnerMorphology::morph_kernel<T, !type><<<gridDim2, blockDim2, sharedSize, stream>>>(dev_in, dev_out, width, height, pWidth, pHeight, padding);
}

template<typename T, bool type>
void LoewnerMorphology::Morph::morph_second_order(T *in, T *out, int width, int height, int padding) {	  
	T *dev_in = NULL;       // device vector holding image matrix
        T *dev_out = NULL;     // device vector holding first output image

        int pWidth = ((width + THREADS_X - 1) / THREADS_X) * THREADS_X;         // width of the image (rounded to the multiple of THREAD_X)
        int pHeight = ((height + THREADS_Y - 1) / THREADS_Y) * THREADS_Y;       // hight of the image (rounded to the multiple of THREAD_Y)
        
	size_t inSize = (pWidth + 2 * padding) * (pHeight + 2 * padding) * sizeof(T);   // size of the input device vector (Circle is used for max and min calculations as described in the paper)
        size_t outSize = width * height * sizeof(T);                                    // size of the (each) output device vector
	
	int sharedWidth = THREADS_X + 2 * padding + 1;  // shared memory width (avoiding bank conflicts)
        int sharedHeight = THREADS_Y + 2 * padding;     // shared memory height
	size_t sharedSize = sharedWidth * sharedHeight * sizeof(Circle);	// size of the shared memory in bypes

	cuda_exec(cudaMalloc(&dev_in, inSize));
	cuda_exec(cudaMalloc(&dev_out, outSize));

	morph_second_order_launcher<T, type>(dev_in, dev_out, in, width, height, pWidth, pHeight, padding, sharedSize);
	
	cuda_exec(cudaDeviceSynchronize());	

	cuda_exec(cudaMemcpy(out, dev_out, outSize, cudaMemcpyDeviceToHost));
	
	cuda_exec(cudaFree(dev_in));
	cuda_exec(cudaFree(dev_out));
}

template<typename T, bool type>
void LoewnerMorphology::Morph::morph_hats(T *in, T *out, int width, int height, int padding) {	  
	T *dev_in = NULL;       // device vector holding image matrix (padding)
        T *dev_out1 = NULL;      // device vector holding first output image
	T *dev_out2 = NULL;	// device vector holding original image

        int pWidth = ((width + THREADS_X - 1) / THREADS_X) * THREADS_X;         // width of the image (rounded to the multiple of THREAD_X)
        int pHeight = ((height + THREADS_Y - 1) / THREADS_Y) * THREADS_Y;       // hight of the image (rounded to the multiple of THREAD_Y)

        size_t inSize = (pWidth + 2 * padding) * (pHeight + 2 * padding) * sizeof(T);   // size of the input device vector (Circle is used for max and min calculations as described in the paper)
        size_t outSize = width * height * sizeof(T);                                    // size of the (each) output device vector
	
	int sharedWidth = THREADS_X + 2 * padding + 1;  // shared memory width (avoiding bank conflicts)
        int sharedHeight = THREADS_Y + 2 * padding;     // shared memory height
	size_t sharedSize = sharedWidth * sharedHeight * sizeof(Circle);	// size of the shared memory in bypes

	cuda_exec(cudaMalloc(&dev_in, inSize));
	cuda_exec(cudaMalloc(&dev_out1, outSize));	
	cuda_exec(cudaMalloc(&dev_out2, outSize));

	morph_second_order_launcher<T, type>(dev_in, dev_out1, in, width, height, pWidth, pHeight, padding, sharedSize);
	cuda_exec(cudaDeviceSynchronize());

	morph_einstein_copy_original_launcher<T, type, NUM_STREAMS>(dev_out1, dev_out2, in, out, width, height, streams);

	#pragma unroll
	for (int i = 0; i < NUM_STREAMS; i++) {
		cuda_exec(cudaStreamSynchronize(streams[i]));
	}	
	
 	cuda_exec(cudaFree(dev_in));
	cuda_exec(cudaFree(dev_out1));
	cuda_exec(cudaFree(dev_out2));
}

template<typename T>
void LoewnerMorphology::Morph::morph_sdth(T *in, T *out, int width, int height, int padding, int *mask) {	  
	T *dev_in1 = NULL;      // device vector holding image matrix (padding)
       	T *dev_in2 = NULL;	// device vector holding image matrix (padding)
	T *dev_out1 = NULL;     // device vector holding first output image
	T *dev_out2 = NULL;	// device vector holding second output image
	T *dev_out_temp = NULL;	// device vector holding output image on second device (optional, if it is available)

        int pWidth = ((width + THREADS_X - 1) / THREADS_X) * THREADS_X;         // width of the image (rounded to the multiple of THREAD_X)
        int pHeight = ((height + THREADS_Y - 1) / THREADS_Y) * THREADS_Y;       // hight of the image (rounded to the multiple of THREAD_Y)

        size_t inSize = (pWidth + 2 * padding) * (pHeight + 2 * padding) * sizeof(T);   // size of the input device vector (Circle is used for max and min calculations as described in the paper)
        size_t outSize = width * height * sizeof(T);                                    // size of the (each) output device vector
	
	int sharedWidth = THREADS_X + 2 * padding + 1;  // shared memory width (avoiding bank conflicts)
        int sharedHeight = THREADS_Y + 2 * padding;     // shared memory height
	size_t sharedSize = sharedWidth * sharedHeight * sizeof(Circle);	// size of the shared memory in bypes

	int count = 0;

	cuda_exec(cudaGetDeviceCount(&count));
	
	if (count < 2) {
		// code executed on one GPU
		cuda_exec(cudaMalloc(&dev_in1, inSize));
		cuda_exec(cudaMalloc(&dev_in2, inSize));
		cuda_exec(cudaMalloc(&dev_out1, outSize));
		cuda_exec(cudaMalloc(&dev_out2, outSize));
		
		morph_second_order_launcher<T, false>(dev_in1, dev_out1, in, width, height, pWidth, pHeight, padding, sharedSize, streams[0]);
                morph_second_order_launcher<T, true>(dev_in2, dev_out2, in, width, height, pWidth, pHeight, padding, sharedSize, streams[1]);

		cuda_exec(cudaStreamSynchronize(streams[0]));	
		cuda_exec(cudaStreamSynchronize(streams[1]));	

		cuda_exec(cudaFree(dev_in2));
	} else {
		// code executed on two GPUs
		cuda_exec(cudaSetDevice(0));
		cuda_exec(cudaMalloc(&dev_in1, inSize));
                cuda_exec(cudaMalloc(&dev_out1, outSize));
                cuda_exec(cudaMalloc(&dev_out2, outSize));

		cuda_exec(cudaSetDevice(1));
		cuda_exec(cudaMalloc(&dev_in2, inSize));
		cuda_exec(cudaMalloc(&dev_out_temp, outSize));
		cuda_exec(cudaMemcpyToSymbol(maskMemory, mask, (2 * padding + 1) * (2 * padding + 1) * sizeof(int), cudaMemcpyHostToDevice));
	
		cuda_exec(cudaSetDevice(0));
		morph_second_order_launcher<T, false>(dev_in1, dev_out1, in, width, height, pWidth, pHeight, padding, sharedSize);
		cuda_exec(cudaSetDevice(1));	
		morph_second_order_launcher<T, true>(dev_in2, dev_out_temp, in, width, height, pWidth, pHeight, padding, sharedSize);
		
		cudaSetDevice(0);
		cuda_exec(cudaDeviceSynchronize());
		cudaSetDevice(1);
		cuda_exec(cudaDeviceSynchronize()); 
	
		cudaSetDevice(0);
		cudaDeviceEnablePeerAccess(1, 0);
		cuda_exec(cudaMemcpyPeer(dev_out2, 0, dev_out_temp, 1, outSize));
	
		cudaSetDevice(1);
		cuda_exec(cudaFree(dev_in2));
		cuda_exec(cudaFree(dev_out_temp));
		cudaSetDevice(0);
	}
	
	morph_einstein_copy_launcher<T, NUM_STREAMS>(dev_out1, dev_out2, out, width, height, streams);

	#pragma unroll
	for (int i = 0; i < NUM_STREAMS; i++) {
		cuda_exec(cudaStreamSynchronize(streams[i]));
	}	

	cuda_exec(cudaFree(dev_in1));
	cuda_exec(cudaFree(dev_out1));
	cuda_exec(cudaFree(dev_out2));
}

template<typename T>
void LoewnerMorphology::Morph::morph_beucher(T *in, T *out, int width, int height, int padding, int *mask) {	  
	T *dev_in1 = NULL;      // device vector holding image matrix (padding)
       	T *dev_in2 = NULL;	// device vector holding image matrix (padding)
	T *dev_out1 = NULL;     // device vector holding first output image
	T *dev_out2 = NULL;	// device vector holding second output image
	T *dev_out_temp = NULL;	// device vector holding output image on second device (optional, if it is available)

        int pWidth = ((width + THREADS_X - 1) / THREADS_X) * THREADS_X;         // width of the image (rounded to the multiple of THREAD_X)
        int pHeight = ((height + THREADS_Y - 1) / THREADS_Y) * THREADS_Y;       // hight of the image (rounded to the multiple of THREAD_Y)

        size_t inSize = (pWidth + 2 * padding) * (pHeight + 2 * padding) * sizeof(T);   // size of the input device vector (Circle is used for max and min calculations as described in the paper)
        size_t outSize = width * height * sizeof(T);                                    // size of the (each) output device vector
	
	int sharedWidth = THREADS_X + 2 * padding + 1;  // shared memory width (avoiding bank conflicts)
        int sharedHeight = THREADS_Y + 2 * padding;     // shared memory height
	size_t sharedSize = sharedWidth * sharedHeight * sizeof(Circle);	// size of the shared memory in bypes

	int count = 0;

	cuda_exec(cudaGetDeviceCount(&count));
	
	if (count < 2) {
		// code executed on one GPU
		cuda_exec(cudaMalloc(&dev_in1, inSize));
		cuda_exec(cudaMalloc(&dev_in2, inSize));
		cuda_exec(cudaMalloc(&dev_out1, outSize));
		cuda_exec(cudaMalloc(&dev_out2, outSize));
		
		morph_basic_launcher<T, false>(dev_in1, dev_out1, in, width, height, pWidth, pHeight, padding, sharedSize, streams[0]);
                morph_basic_launcher<T, true>(dev_in2, dev_out2, in, width, height, pWidth, pHeight, padding, sharedSize, streams[1]);

		cuda_exec(cudaStreamSynchronize(streams[0]));	
		cuda_exec(cudaStreamSynchronize(streams[1]));	

		cuda_exec(cudaFree(dev_in2));
	} else {
		// code executed on two GPUs
		cuda_exec(cudaSetDevice(0));
		cuda_exec(cudaMalloc(&dev_in1, inSize));
                cuda_exec(cudaMalloc(&dev_out1, outSize));
                cuda_exec(cudaMalloc(&dev_out2, outSize));

		cuda_exec(cudaSetDevice(1));
		cuda_exec(cudaMalloc(&dev_in2, inSize));
		cuda_exec(cudaMalloc(&dev_out_temp, outSize));
		cuda_exec(cudaMemcpyToSymbol(maskMemory, mask, (2 * padding + 1) * (2 * padding + 1) * sizeof(int), cudaMemcpyHostToDevice));
	
		cuda_exec(cudaSetDevice(0));
		morph_basic_launcher<T, false>(dev_in1, dev_out1, in, width, height, pWidth, pHeight, padding, sharedSize);
		cuda_exec(cudaSetDevice(1));	
		morph_basic_launcher<T, true>(dev_in2, dev_out_temp, in, width, height, pWidth, pHeight, padding, sharedSize);
		
		cudaSetDevice(0);
		cuda_exec(cudaDeviceSynchronize());
		cudaSetDevice(1);
		cuda_exec(cudaDeviceSynchronize()); 
	
		cudaSetDevice(0);
		cudaDeviceEnablePeerAccess(1, 0);
		cuda_exec(cudaMemcpyPeer(dev_out2, 0, dev_out_temp, 1, outSize));
	
		cudaSetDevice(1);
		cuda_exec(cudaFree(dev_in2));
		cuda_exec(cudaFree(dev_out_temp));
		cudaSetDevice(0);
	}

	morph_einstein_copy_launcher<T, NUM_STREAMS>(dev_out1, dev_out2, out, width, height, streams);
	
	#pragma unroll
	for (int i = 0; i < NUM_STREAMS; i++) {
		cuda_exec(cudaStreamSynchronize(streams[i]));
	}	

	cuda_exec(cudaFree(dev_in1));
	cuda_exec(cudaFree(dev_out1));
	cuda_exec(cudaFree(dev_out2));
}

template<typename T, bool type>
void LoewnerMorphology::Morph::morph_gradients(T *in, T *out, int width, int height, int padding) {	  
	T *dev_in = NULL;       // device vector holding image matrix (padding)
        T *dev_out = NULL;      // device vector holding first output image

        int pWidth = ((width + THREADS_X - 1) / THREADS_X) * THREADS_X;         // width of the image (rounded to the multiple of THREAD_X)
        int pHeight = ((height + THREADS_Y - 1) / THREADS_Y) * THREADS_Y;       // hight of the image (rounded to the multiple of THREAD_Y)

        size_t inSize = (pWidth + 2 * padding) * (pHeight + 2 * padding) * sizeof(T);   // size of the input device vector (Circle is used for max and min calculations as described in the paper)
        size_t outSize = width * height * sizeof(T);                                    // size of the (each) output device vector
	
	int sharedWidth = THREADS_X + 2 * padding + 1;  // shared memory width (avoiding bank conflicts)
        int sharedHeight = THREADS_Y + 2 * padding;     // shared memory height
	size_t sharedSize = sharedWidth * sharedHeight * sizeof(Circle);	// size of the shared memory in bypes

	cuda_exec(cudaMalloc(&dev_in, inSize));
	cuda_exec(cudaMalloc(&dev_out, outSize));	

	morph_basic_launcher<T, type>(dev_in, dev_out, in, width, height, pWidth, pHeight, padding, sharedSize);
	cuda_exec(cudaDeviceSynchronize());
	
	morph_einstein_copy_original_launcher<T, type, NUM_STREAMS>(dev_out, dev_in, in, out, width, height, streams);

	#pragma unroll
	for (int i = 0; i < NUM_STREAMS; i++) {
		cuda_exec(cudaStreamSynchronize(streams[i]));
	}	
	
 	cuda_exec(cudaFree(dev_in));
	cuda_exec(cudaFree(dev_out));
}

template<typename T>
void LoewnerMorphology::Morph::morph_laplacian(T *in, T *out, int width, int height, int padding, int *mask) {	  
	T *dev_in1 = NULL;      // device vector holding image matrix (padding)
       	T *dev_in2 = NULL;	// device vector holding image matrix (padding)
	T *dev_out1 = NULL;     // device vector holding first output image
	T *dev_out2 = NULL;	// device vector holding second output image
	T *dev_out_temp = NULL;	// device vector holding output image on second device (optional, if it is available)

        int pWidth = ((width + THREADS_X - 1) / THREADS_X) * THREADS_X;         // width of the image (rounded to the multiple of THREAD_X)
        int pHeight = ((height + THREADS_Y - 1) / THREADS_Y) * THREADS_Y;       // hight of the image (rounded to the multiple of THREAD_Y)

        size_t inSize = (pWidth + 2 * padding) * (pHeight + 2 * padding) * sizeof(T);   // size of the input device vector (Circle is used for max and min calculations as described in the paper)
        size_t outSize = width * height * sizeof(T);                                    // size of the (each) output device vector
	
	int sharedWidth = THREADS_X + 2 * padding + 1;  // shared memory width (avoiding bank conflicts)
        int sharedHeight = THREADS_Y + 2 * padding;     // shared memory height
	size_t sharedSize = sharedWidth * sharedHeight * sizeof(Circle);	// size of the shared memory in bypes

	int count = 0;

	cuda_exec(cudaGetDeviceCount(&count));
	
	if (count < 2) {
		cuda_exec(cudaMalloc(&dev_in1, inSize));
		cuda_exec(cudaMalloc(&dev_in2, inSize));
        	cuda_exec(cudaMalloc(&dev_out1, outSize));
        	cuda_exec(cudaMalloc(&dev_out2, outSize));

        	morph_basic_launcher<T, true>(dev_in1, dev_out1, in, width, height, pWidth, pHeight, padding, sharedSize, streams[0]);
        	morph_basic_launcher<T, false>(dev_in2, dev_out2, in, width, height, pWidth, pHeight, padding, sharedSize, streams[1]);
		
		cuda_exec(cudaMemcpyAsync(dev_in1, in, outSize, cudaMemcpyHostToDevice, streams[0]));
		morph_einstein_async(dev_out1, dev_in1, width, height, width, width, streams[0]); 		
		cuda_exec(cudaMemcpyAsync(dev_in2, in, outSize, cudaMemcpyHostToDevice, streams[1]));
		morph_einstein_async(dev_in2, dev_out2, width, height, width, width, streams[1]); 		
	
		cuda_exec(cudaStreamSynchronize(streams[0]));
		cuda_exec(cudaStreamSynchronize(streams[1]));
		
		morph_einstein_copy_launcher<T, NUM_STREAMS>(dev_out1, dev_in2, out, width, height, streams);

		#pragma unroll
		for (int i = 0; i < NUM_STREAMS; i++) {
			cuda_exec(cudaStreamSynchronize(streams[i]));
		}	

		cuda_exec(cudaFree(dev_in2));
	} else {
		// code executed on two GPUs
		cuda_exec(cudaSetDevice(0));
		cuda_exec(cudaMalloc(&dev_in1, inSize));
                cuda_exec(cudaMalloc(&dev_out1, outSize));
		cuda_exec(cudaMalloc(&dev_out2, outSize));

		cuda_exec(cudaSetDevice(1));
		cuda_exec(cudaMalloc(&dev_in2, inSize));
		cuda_exec(cudaMalloc(&dev_out_temp, outSize));		
		cuda_exec(cudaMemcpyToSymbol(maskMemory, mask, (2 * padding + 1) * (2 * padding + 1) * sizeof(int), cudaMemcpyHostToDevice));	
		cuda_exec(cudaSetDevice(0));
		morph_basic_launcher<T, true>(dev_in1, dev_out1, in, width, height, pWidth, pHeight, padding, sharedSize);
		cuda_exec(cudaMemcpyAsync(dev_in1, in, outSize, cudaMemcpyHostToDevice));
		morph_einstein_async(dev_out1, dev_in1, width, height, width, width); 		
		
		cuda_exec(cudaSetDevice(1));	
		morph_basic_launcher<T, false>(dev_in2, dev_out_temp, in, width, height, pWidth, pHeight, padding, sharedSize);
		cuda_exec(cudaMemcpyAsync(dev_in2, in, outSize, cudaMemcpyHostToDevice));
		morph_einstein_async(dev_in2, dev_out_temp, width, height, width, width);

		cudaSetDevice(0);
		cuda_exec(cudaDeviceSynchronize());
		cudaSetDevice(1);
		cuda_exec(cudaDeviceSynchronize()); 
	
		cudaSetDevice(0);
		cudaDeviceEnablePeerAccess(1, 0);
		cuda_exec(cudaMemcpyPeer(dev_out2, 0, dev_in2, 1, outSize));
	
		morph_einstein_copy_launcher<T, NUM_STREAMS>(dev_out1, dev_in2, out, width, height, streams);

		#pragma unroll
		for (int i = 0; i < NUM_STREAMS; i++) {
			cuda_exec(cudaStreamSynchronize(streams[i]));
		}
		
		cudaSetDevice(1);
		cuda_exec(cudaFree(dev_in2));
		cuda_exec(cudaFree(dev_out_temp));
		cudaSetDevice(0);
	}

	cuda_exec(cudaFree(dev_in1));
	cuda_exec(cudaFree(dev_out1));
	cuda_exec(cudaFree(dev_out2));
}

template<typename T>
void LoewnerMorphology::Morph::morph_shock(T *in, T *out, int width, int height, int padding, int *mask) {	  
	T *dev_in1 = NULL;      // device vector holding image matrix (padding)
       	T *dev_in2 = NULL;	// device vector holding image matrix (padding)
	T *dev_out1 = NULL;     // device vector holding first output image
	T *dev_out2 = NULL;	// device vector holding second output image
	T *dev_out_temp = NULL;	// device vector holding output image on second device (optional, if it is available)

        int pWidth = ((width + THREADS_X - 1) / THREADS_X) * THREADS_X;         // width of the image (rounded to the multiple of THREAD_X)
        int pHeight = ((height + THREADS_Y - 1) / THREADS_Y) * THREADS_Y;       // hight of the image (rounded to the multiple of THREAD_Y)

        size_t inSize = (pWidth + 2 * padding) * (pHeight + 2 * padding) * sizeof(T);   // size of the input device vector (Circle is used for max and min calculations as described in the paper)
        size_t outSize = width * height * sizeof(T);                                    // size of the (each) output device vector
	
	int sharedWidth = THREADS_X + 2 * padding + 1;  // shared memory width (avoiding bank conflicts)
        int sharedHeight = THREADS_Y + 2 * padding;     // shared memory height
	size_t sharedSize = sharedWidth * sharedHeight * sizeof(Circle);	// size of the shared memory in bypes

	int count = 0;

	cuda_exec(cudaGetDeviceCount(&count));
	
	if (count < 2) {
		cuda_exec(cudaMalloc(&dev_in1, inSize));
		cuda_exec(cudaMalloc(&dev_in2, inSize));
        	cuda_exec(cudaMalloc(&dev_out1, outSize));
        	cuda_exec(cudaMalloc(&dev_out2, outSize));

        	morph_basic_launcher<T, true>(dev_in1, dev_out1, in, width, height, pWidth, pHeight, padding, sharedSize, streams[0]);
        	morph_basic_launcher<T, false>(dev_in2, dev_out2, in, width, height, pWidth, pHeight, padding, sharedSize, streams[1]);
		
		cuda_exec(cudaMemcpyAsync(dev_in1, in, outSize, cudaMemcpyHostToDevice, streams[0]));
		morph_einstein_async(dev_out1, dev_in1, width, height, width, width, streams[0]); 		
		cuda_exec(cudaMemcpyAsync(dev_in2, in, outSize, cudaMemcpyHostToDevice, streams[1]));
		morph_einstein_async(dev_in2, dev_out2, width, height, width, width, streams[1]); 		
	
		cuda_exec(cudaStreamSynchronize(streams[0]));
		cuda_exec(cudaStreamSynchronize(streams[1]));	
		
		morph_einstein_async(dev_out1, dev_in2, width, height, width, width);
		cuda_exec(cudaDeviceSynchronize());

		cuda_exec(cudaFree(dev_in2));
	} else {
		// code executed on two GPUs
		cuda_exec(cudaSetDevice(0));
		cuda_exec(cudaMalloc(&dev_in1, inSize));
                cuda_exec(cudaMalloc(&dev_out1, outSize));
		cuda_exec(cudaMalloc(&dev_out2, outSize));

		cuda_exec(cudaSetDevice(1));
		cuda_exec(cudaMalloc(&dev_in2, inSize));
		cuda_exec(cudaMalloc(&dev_out_temp, outSize));		
		cuda_exec(cudaMemcpyToSymbol(maskMemory, mask, (2 * padding + 1) * (2 * padding + 1) * sizeof(int), cudaMemcpyHostToDevice));	
		cuda_exec(cudaSetDevice(0));
		morph_basic_launcher<T, false>(dev_in1, dev_out1, in, width, height, pWidth, pHeight, padding, sharedSize);
		cuda_exec(cudaMemcpyAsync(dev_in1, in, outSize, cudaMemcpyHostToDevice));
		morph_einstein_async(dev_out1, dev_in1, width, height, width, width); 		
		
		cuda_exec(cudaSetDevice(1));	
		morph_basic_launcher<T, true>(dev_in2, dev_out_temp, in, width, height, pWidth, pHeight, padding, sharedSize);
		cuda_exec(cudaMemcpyAsync(dev_in2, in, outSize, cudaMemcpyHostToDevice));
		morph_einstein_async(dev_in2, dev_out_temp, width, height, width, width);

		cudaSetDevice(0);
		cuda_exec(cudaDeviceSynchronize());
		cudaSetDevice(1);
		cuda_exec(cudaDeviceSynchronize()); 
	
		cudaSetDevice(0);
		cudaDeviceEnablePeerAccess(1, 0);
		cuda_exec(cudaMemcpyPeer(dev_out2, 0, dev_in2, 1, outSize));
	
		morph_einstein_async(dev_out1, dev_in2, width, height, width, width);		
		cuda_exec(cudaDeviceSynchronize());
		
		cudaSetDevice(1);
		cuda_exec(cudaFree(dev_in2));
		cuda_exec(cudaFree(dev_out_temp));
		cudaSetDevice(0);
	}

	// there is laplacian on location dev_out1
	cuda_exec(cudaMalloc(&dev_in2, inSize));
	morph_shock_launcher(dev_in1, dev_in2, dev_out2, dev_out1, in, width, height, pWidth, pHeight, padding, sharedSize);
	cuda_exec(cudaDeviceSynchronize());
	
	cuda_exec(cudaMemcpy(out, dev_out2, outSize, cudaMemcpyDeviceToHost));

	cuda_exec(cudaFree(dev_in1));
	cuda_exec(cudaFree(dev_in2));
	cuda_exec(cudaFree(dev_out1));
	cuda_exec(cudaFree(dev_out2));
}

template<typename T>
void LoewnerMorphology::Morph::morph_handle(T *in, T *out, int width, int height, int padding, int *mask, int morphType, int iters) {
	if (iters < 1) {
		printf("Operation cannot be executed. Number of iterations must be greater than 0. You provided %d.\n", iters);
		exit(EXIT_FAILURE);
	} 

	cuda_exec(cudaSetDevice(0));

	// copying mask to constant memory
	int maskSize = (2 * padding + 1) * (2 * padding + 1) * sizeof(int);
	cuda_exec(cudaMemcpyToSymbol(maskMemory, mask, maskSize, cudaMemcpyHostToDevice));

	#pragma unroll
	for (int i = 0; i < NUM_STREAMS; i++) {
		cuda_exec(cudaStreamCreate(&streams[i]));
	}

	switch (morphType) {
		case 0:
			for (int i = 0; i < iters; i++) {
				if (i == 0) {
					morph_basic<T, false>(in, out, width, height, padding);
				} else {
					morph_basic<T, false>(out, out, width, height, padding);
				}
			}

			break;
		case 1:
			for (int i = 0; i < iters; i++) {
				if (i == 0) {
					morph_basic<T, true>(in, out, width, height, padding);
				} else {
					morph_basic<T, true>(out, out, width, height, padding);
				}
			}
			break;
		case 2:
			for (int i = 0; i < iters; i++) {
				if (i == 0) {
					morph_second_order<T, false>(in, out, width, height, padding);
				} else {
					morph_second_order<T, false>(out, out, width, height, padding);
				}
			}
			break;
		case 3:
			for (int i = 0; i < iters; i++) {
				if (i == 0) {
					morph_second_order<T, true>(in, out, width, height, padding);
				} else {
					morph_second_order<T, true>(out, out, width, height, padding);
				}
			}
			break;
		case 4: 
			for (int i = 0; i < iters; i++) {
				if (i == 0) {
					morph_hats<T, false>(in, out, width, height, padding);
				} else {
					morph_hats<T, false>(out, out, width, height, padding);
				}
			}
			break;
		case 5: 
			for (int i = 0; i < iters; i++) {
				if (i == 0) {
					morph_hats<T, true>(in, out, width, height, padding);
				} else {
					morph_hats<T, true>(out, out, width, height, padding);
				}
			}
			break;
		case 6:
			for (int i = 0; i < iters; i++) {
				if (i == 0) {
					morph_sdth<T>(in, out, width, height, padding, mask);
				} else {
					morph_sdth<T>(out, out, width, height, padding, mask);
				}
			}
			break;
		case 7:
			for (int i = 0; i < iters; i++) {
				if (i == 0) {
					morph_beucher<T>(in, out, width, height, padding, mask);
				} else {
					morph_beucher<T>(out, out, width, height, padding, mask);
				}
			}
			break;
		case 8:
			for (int i = 0; i < iters; i++) {
				if (i == 0) {
					morph_gradients<T, false>(in, out, width, height, padding);
				} else {
					morph_gradients<T, false>(out, out, width, height, padding);
				}
			}
			break;
		case 9:
			for (int i = 0; i < iters; i++) {
				if (i == 0) {
					morph_gradients<T, true>(in, out, width, height, padding);
				} else {
					morph_gradients<T, true>(out, out, width, height, padding);
				}
			}
			break;
		case 10:
			for (int i = 0; i < iters; i++) {
				if (i == 0) {
					morph_laplacian<T>(in, out, width, height, padding, mask);
				} else {
					morph_laplacian<T>(out, out, width, height, padding, mask);
				}
			}
			break;		
		case 11:
			for (int i = 0; i < iters; i++) {
				if (i == 0) {
					morph_shock<T>(in, out, width, height, padding, mask);
				} else {
					morph_shock<T>(out, out, width, height, padding, mask);
				}
			}
			break;
	}
	
	#pragma unroll
	for (int i = 0; i < NUM_STREAMS; i++) {
		cuda_exec(cudaStreamDestroy(streams[i]));
	}
}

LoewnerMorphology::Morph::Morph(const char *imageFile, const char *maskFile, int maskDim) {
	double *image = NULL;
	double *data = NULL;

	if (maskDim % 2 == 0 || maskDim * maskDim > 1024) {
                printf("Mask dimension should be odd and its squere should be less than 1024.\n");
                exit(EXIT_FAILURE);
        }
 	
	mask = (int *)malloc(maskDim * maskDim * sizeof(int));
	readMaskFromFile(mask, maskDim, maskFile);

	padding = maskDim / 2;

	omp_set_num_threads(OMP_THREAD_NUM);
	
	inputImage = new CImgFloatWrapper(imageFile);
	outputImage = nullptr;

	width = inputImage->width();
	height = inputImage->height();
	spectrum = inputImage->spectrum();	
	size = width * height;

	image = (double *)malloc(size * spectrum * sizeof(double));
	data = (double *)malloc(size * spectrum * sizeof(double));
	
	cuda_exec(cudaMallocHost(&matrices, size * sizeof(LoewnerMorphology::MorphColorMatrix)));
	cuda_exec(cudaMallocHost(&result, size * sizeof(LoewnerMorphology::MorphColorMatrix)));	

	Conversions::type2double(inputImage->data(), image, size * spectrum);
	Conversions::rgb2mhcl(image, image + size, image + 2 * size, data, data + size, data + 2 * size, size);
	Conversions::mhcl2matrix(data, data + size, data + 2 * size, matrices, size);
	
	free(image);
	free(data);
}

LoewnerMorphology::Morph::~Morph() {
	free(mask);
	
	cuda_exec(cudaFreeHost(matrices));
        cuda_exec(cudaFreeHost(result));

	delete inputImage;
	delete outputImage;
}

void LoewnerMorphology::Morph::createOutputImage() {
	double *image = (double *)malloc(size * spectrum * sizeof(double));
        double *data = (double *)malloc(size * spectrum * sizeof(double));	
	float *out = (float *)malloc(size * spectrum * sizeof(float));

	Conversions::matrix2mhcl(result, data, data + size, data + 2 * size, size);
        Conversions::mhcl2rgb(data, data + size, data + 2 * size, image, image + size, image + 2 * size, size);
        Conversions::double2type(image, out, size * spectrum);
        
        if (outputImage != nullptr) {
		delete outputImage;
	}
	
	outputImage = new CImgFloatWrapper(out, width, height, spectrum);
	
	free(image);
        free(data);
	free(out);
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

template<typename T>
void LoewnerMorphology::Morph::copy(T *in, T *out, int size) {
	#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		out[i] = in[i];
	}
}

float *LoewnerMorphology::Morph::returnResult() {
	if (outputImage == nullptr) {
                printf("There is no result to return.\n");
                return nullptr;
        }

	float *out = (float *)malloc(size * spectrum * sizeof(float));	
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
