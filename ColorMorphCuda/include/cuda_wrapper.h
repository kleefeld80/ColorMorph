#ifndef CUDA_WRAPPER_H
#define CUDA_WRAPPER_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <errno.h>
#include <float.h>

// defining utilitty fuctions for implementing cuda algorithms
 
#define cuda_exec(func_call)\
	do {\
		cudaError_t error = (func_call);\
\
		if (error != cudaSuccess) {\
			fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));\
			exit(EXIT_FAILURE);\
		}\
	} while(0)

#define host_alloc(hst_A, type, dimension)\
	do {\
		if (((hst_A) = (type *)malloc((dimension) * sizeof(type))) == NULL) {\
			fprintf(stderr, "%s:%d: insufficient memory: %s\n", __FILE__, __LINE__, strerror(errno));\
			exit(EXIT_FAILURE);\
		}\
	} while(0)

#define host_free(hst_A) free(hst_A)

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

#define read_binary_file(ptr, type, n, file_ptr, file_name)\
	do {\
		if (fread((ptr), sizeof(type), (n), (file_ptr)) != (n)) {\
			fprintf(stderr, "%s:%d: error while reading file %s: %s\n", __FILE__, __LINE__, (file_name), strerror(errno));\
			exit(EXIT_FAILURE);\
		}\
	} while(0)


#define write_binary_file(ptr, type, n, file_ptr, file_name)\
	do {\
		if (fwrite((ptr), sizeof(type), (n), (file_ptr)) != (n)) {\
			fprintf(stderr, "%s:%d: error while writing file %s: %s\n", __FILE__, __LINE__, (file_name), strerror(errno));\
			exit(EXIT_FAILURE);\
		}\
	} while(0)

#define print_matrix(a, n, m)\
	do {\
		for (int i = 0; i < (n); i++) {\
			for(int j = 0; j < (m); j++) {\
				printf("%.6lg ", (a)[i * m + j]);\
			}\
			printf("\n");\
		}\
	} while(0)

template <typename T>
void init_matrix(T *a, int n, int m, int lda) {
	srand(time(NULL));

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			a[j] = ((double)rand()) / RAND_MAX;
		}

		a += lda;
	}
}

double timer() {
	struct timeval t;

	gettimeofday(&t, NULL);

	return (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
} 

void check_results_gpu(double *cpu_array, double *gpu_array, int size)
{
	for (int ix = 0; ix < size; ++ix)
		if (fabs(cpu_array[ix] - gpu_array[ix]) >= DBL_EPSILON) {
			printf("CPU and GPU results differ at element %d\n", ix);
			printf("CPU value: %lg\n", cpu_array[ix]);
			printf("GPU value: %lg\n", gpu_array[ix]);

			return;
		}

	printf("GPU result is correct\n");
}

void check_results_gpu(float *cpu_array, float *gpu_array, int size)
{
        for (int ix = 0; ix < size; ++ix)
                if (fabs(cpu_array[ix] - gpu_array[ix]) >= FLT_EPSILON) {
                        printf("CPU and GPU results differ at element %d\n", ix);
                        printf("CPU value: %lf\n", cpu_array[ix]);
                        printf("GPU value: %lf\n", gpu_array[ix]);

                        return;
                }

        printf("GPU result is correct\n");
}

void check_results(float *array1, float *array2, int size)
{
        for (int ix = 0; ix < size; ++ix)
                if (fabs(array1[ix] - array2[ix]) >= FLT_EPSILON) {
                        printf("Results differ at element %d\n", ix);
                        printf("array1 value: %10.8f\n", array1[ix]);
                        printf("array2 value: %10.8f\n", array2[ix]);

                        return;
                }

        printf("Result is correct!\n");
}

void check_results_details(float *array1, float *array2, int size, bool printDifferences)
{
	int count = 0;
	float max = 0.0;
	float sum = 0.0;

	for (int ix = 0; ix < size; ++ix) {
		float diff = fabs(array1[ix] - array2[ix]);
		
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
	printf("Average difference: %f\n", (count == 0) ? 0.0 : sum / count);
}


void check_results_epsilon(float *array1, float *array2, int size, float epsilon)
{
	int count = 0;
	float max = 0.0;
	float sum = 0.0;

	for (int ix = 0; ix < size; ++ix) {
		float diff = fabs(array1[ix] - array2[ix]);
		
		if (diff >= epsilon) {
			if (diff > max) max = diff;

			printf("Results differ at element %d\n", ix);
			printf("array1 value: %lf\n", array1[ix]);
			printf("array2 value: %lf\n", array2[ix]);
			++count;
			sum += diff;
		}
	}

	printf("Number of errors for given epsilon %f: %d\n", epsilon, count);
	printf("Max difference: %f\n", max);
	printf("Average difference: %f\n", (count == 0) ? 0.0 : sum / count);
}

template<typename T>
void count_differences(T *array1, T *array2, int size) {
	int count = 0;

	for (int i = 0; i < size; i++) {
		if (array1[i] != array2[i]) {
			++count;
			printf("Difference detected on element %d!\n", i);
		}
	}

	printf("Number of differences: %d.\n", count);
}

void print_vector(float *vector, int n) {
	for (int i = 0; i < n; i++) {
		printf("%f ", vector[i]);	
	}
	printf("\n");
}

#endif
