#ifndef MORPH_KERNELS_H
#define MORPH_KERNELS_H

#include "../loewner_morphology.h"

// HEADER THAT IMPLEMENTS TEMPLATED KERNELS INTRODUCED IN HEADER ../include/loewner_declaration.h

typedef LoewnerMorphology::MorphCircle Circle;

template<typename T>
void __global__ LoewnerMorphology::fill(T *in, int width, T element) {
        T *current = in + blockIdx.y * width;
        
        for (int i = threadIdx.x; i < width; i += blockDim.x) {
                current[i] = element;
        }
}

LoewnerMorphology::MorphColorMatrix __device__ LoewnerMorphology::dilate_gpu(Circle *smemStart, int smemWidth, int padding) {
        int n = 2 * padding + 1;

        Circle *current = smemStart - smemWidth * padding - padding;

        MorphSmallestCircleProblemMask scp(current, maskMemory, n, n, smemWidth);

        return scp.compute().returnMax().toMorphColorMatrixCone2Epsilon();
}

LoewnerMorphology::MorphColorMatrix __device__ LoewnerMorphology::erode_gpu(Circle *smemStart, int smemWidth, int padding) {
        int n = 2 * padding + 1;

        Circle *current = smemStart - smemWidth * padding - padding;

        MorphSmallestCircleProblemMask scp(current, maskMemory, n, n, smemWidth);

        return scp.compute().returnMin().toMorphColorMatrixCone2Epsilon();
}

template<typename T, bool type>
void __global__ LoewnerMorphology::morph_kernel(T *in, T *out, int width, int height, int pWidth, int pHeight, int padding) {
        extern __shared__ Circle smem[];

        // shared memory tile dimensions
        const int sharedWidth = blockDim.x + 2 * padding + 1;
        const int sharedHeight = blockDim.y + 2 * padding;

        // current thread idx
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int idy = blockIdx.y * blockDim.y + threadIdx.y;

        if (idx < pWidth && idy < pHeight) {
                const int in_lda = pWidth + 2 * padding;
                // setting current pointer
                T *in_ptr = in + blockIdx.y * blockDim.y * in_lda + blockIdx.x * blockDim.x;
                
                // loading data to shared memory (TILE with dimensions sharedWidth and sharedHeight)
                for (int i = threadIdx.y; i < sharedHeight; i += blockDim.y) {
                        for (int j = threadIdx.x; j < sharedWidth - 1; j += blockDim.x) {
                                smem[i * sharedWidth + j] = (type) ? Circle(in_ptr[i * in_lda + j]).prepareMin() : Circle(in_ptr[i * in_lda + j]).prepareMax();
                        }
                }
        }

        int i = threadIdx.y + padding;
        int j = threadIdx.x + padding;

        __syncthreads();

        if (idx < width && idy < height) {
                Circle *current = smem + i * sharedWidth + j;
                out[idy * width + idx] = (type) ? erode_gpu(current, sharedWidth, padding) : dilate_gpu(current, sharedWidth, padding);
        }
}

template<typename T>
void __global__ LoewnerMorphology::einstein_kernel(T *image1, T *image2, int width, int height, int lda1, int lda2) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int idy = blockIdx.y * blockDim.y + threadIdx.y;

        int offset1 = idy * lda1 + idx;
        int offset2 = idy * lda2 + idx;

        if (idx < width && idy < height) {
                LoewnerMorphology::MorphColorMatrix m1 = Circle(image1[offset1]).toMorphColorMatrixSphere();
                LoewnerMorphology::MorphColorMatrix m2 = Circle(image2[offset2]).toMorphColorMatrixSphere().negate();

                image1[offset1] = Circle(EinsteinOperations::einsteinAdditionMod(m1, m2)).toMorphColorMatrixCone2Epsilon();
        }
}

template<typename T>
void __global__ LoewnerMorphology::shock_kernel(T *in1, T *in2, T *out, T* laplacian, int width, int height, int pWidth, int pHeight, int padding) {
        extern __shared__ Circle smem[];

        // shared memory tile dimensions
        const int sharedWidth = blockDim.x + 2 * padding + 1;
        const int sharedHeight = blockDim.y + 2 * padding;

        Circle *smemDilation = smem;
        Circle *smemErosion = smem + sharedWidth * sharedHeight;

        // current thread idx
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int idy = blockIdx.y * blockDim.y + threadIdx.y;

        if (idx < pWidth && idy < pHeight) {
                const int in_lda = pWidth + 2 * padding;
                // setting current pointer
                T *in_ptr1 = in1 + blockIdx.y * blockDim.y * in_lda + blockIdx.x * blockDim.x;
                T *in_ptr2 = in2 + blockIdx.y * blockDim.y * in_lda + blockIdx.x * blockDim.x;
                
                // loading data to shared memory (TILE with dimensions sharedWidth and sharedHeight)
                for (int i = threadIdx.y; i < sharedHeight; i += blockDim.y) {
                        for (int j = threadIdx.x; j < sharedWidth - 1; j += blockDim.x) {
                                smemDilation[i * sharedWidth + j] = Circle(in_ptr1[i * in_lda + j]).prepareMax();
                                smemErosion[i * sharedWidth + j] = Circle(in_ptr2[i * in_lda + j]).prepareMin();
                        }
                }
        } 

        int i = threadIdx.y + padding;
        int j = threadIdx.x + padding;

        __syncthreads();

        if (idx < width && idy < height) {
                if (laplacian[idy * width + idx].trace() > 0) {
                        Circle *current = smemErosion + i * sharedWidth + j;
                        out[idy * width + idx] = erode_gpu(current, sharedWidth, padding);
                } else {
                        Circle *current = smemDilation + i * sharedWidth + j;
                        out[idy * width + idx] = dilate_gpu(current, sharedWidth, padding);
                }
        }
}

void __global__ LoewnerMorphology::print_kernel(LoewnerMorphology::MorphColorMatrix *m, int size) {
        for (int i = 0; i < size; i++) {
                m[i].printMorphColorMatrix();
                printf("\n");
        }
}

void __global__ LoewnerMorphology::print_kernel(LoewnerMorphology::MorphColorMatrix *in, int width, int height, int lda) {
        MorphColorMatrix *current = in;

        for(int i = 0; i < height; i++) {
                for(int j = 0; j < width; j++) {
                        current[j].printMorphColorMatrix(); printf(" ");
                }
                printf("\n");

                current += lda;
        }
}

#endif
