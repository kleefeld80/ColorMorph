#ifndef HSI_MATRIX_H
#define HSI_MATRIX_H

#include <stdio.h>
#include <math.h>

#ifndef RGB_MAX
#define RGB_MAX 255.0
#endif

#ifndef INTENSITY_ALPHA
#define INTENSITY_ALPHA 10
#endif

#ifndef INTENSITY_FACTOR
#define INTENSITY_FACTOR ((T) RGB_MAX / INTENSITY_ALPHA)
#endif

#ifndef HUE_TRESHOLD
#define HUE_TRESHOLD 0.5
#endif

#ifndef PI
#define PI 3.141592653589793
#endif

/*
 * Represents HSI matrix representing HSI image values. H stands for hue, S stands for saturation and I stands for intensity.
 * HSIMatrix is a 2x2 symetric matrix with hue value as top-left element, saturation values as both antidiagonal elements and
 * the intensity value as the bottom-right element.
 */
template<typename T>
struct HSIMatrix {
        T h;
        T s;
        T i;

        /*
	 * Default constructor of HSIMatrix object. Each elements are set to 0.
	 */
        __host__ __device__ HSIMatrix() : HSIMatrix(0, 0, 0) {}
        
	/*
	 * Constructs HSIMatrix from given values representing H, S and I values.
	 */
        __host__ __device__ HSIMatrix(T h, T s, T i) : h(h), s(s), i(i) {}
        
	/*
	 * Converts RGB image to vector containing HSIMatrix elements.
	 */
        template<typename S>
        static void __host__ __device__ rgb2HSIMatrixVector(S *r, S *g, S *b, HSIMatrix<T> *vector, int size);
        
	/*
	 * Converts vector containing HSIMatrix elements to RGB image values. R values will be stored on memory loaction r,
	 * G values will be stored on memory loaction b and B values will be stored on memory location b. These addresses
	 * should be allocated with the size of the size parameter.
	 */
        template<typename S>
        static void __host__ __device__ HSIMatrixVector2rgb(HSIMatrix<T> *vector, S *r, S *g, S *b, int size);
       
 	/*
	 * Prints HSIMatrix object to the standard output.
	 */
        void __host__ __device__ print();
        
	/*
	 * Prints only H values from the HSIMatrix objects from given matrix.
	 */
        static void __host__ __device__ printMatrix(HSIMatrix<T> *matrix, int width, int height, int lda);
        
	/*
	 * Prints only H values from the HSIMatrix objects from given matrix.
	 */
	static void __host__ __device__ printMatrixH(HSIMatrix<T> *matrix, int width, int height, int lda);
        
	/*
	 * Prints only S values from the HSIMatrix objects from given matrix.
	 */	
	static void __host__ __device__ printMatrixS(HSIMatrix<T> *matrix, int width, int height, int lda);
        
	/*
	 * Prints only I values from the HSIMatrix objects from given matrix.
	 */
	static void __host__ __device__ printMatrixI(HSIMatrix<T> *matrix, int width, int height, int lda);
        
	/*
	 * Utiliy method for finding minimum between three values of type T.
	 */
	static T __host__ __device__ min3(T a, T b , T c);

	/*
	 *  Returns maximum HSIMatrix in terms of lexicographic ordering.
	 */
        static HSIMatrix __host__ __device__ max() { return HSIMatrix(0.0, 1.0, 1.0); }
        
        /*
	 * Returns minimum HSIMatrix in terms of lexicographic ordering.
	 */
	static HSIMatrix __host__ __device__ min() { return HSIMatrix(1.0, 0.0, 0.0); }
        
	/*
	 * Comparisson operator < representing order using lexicographical cascades on the HSI-space with parameter
	 * alpha equal to 10.
	 */
        template<typename S>
        friend bool __host__ __device__ operator<(HSIMatrix<S> &a, HSIMatrix<S> &b);
        
	/*
         * Comparisson operator > representing order using lexicographical cascades on the HSI-space with parameter
         * alpha equal to 10.
         */
	template<typename S>
        friend bool __host__ __device__ operator>(HSIMatrix<S> &a, HSIMatrix<S> &b);
};

template<typename T>
bool __host__ __device__ operator<(HSIMatrix<T> &a, HSIMatrix<T> &b) {
        T val1 = ceil(a.i * 255.0 / 10.0);
        T val2 = ceil(b.i * 255.0 / 10.0);

        if (val1 < val2) {
                return true;
        } else if (val1 == val2) {
                if (a.s < b.s) {
                        return true;
                } else if (a.s == b.s) {
                        val1 = abs(a.h);
                        val2 = abs(b.h);

                        val1 = (val1 < HUE_TRESHOLD) ? val1 : 1 - val1;
                        val2 = (val2 < HUE_TRESHOLD) ? val2 : 1 - val2;

                        return (val1 > val2);
                }
        }

        return false;
}

template<typename T>
bool __host__ __device__ operator>(HSIMatrix<T> &a, HSIMatrix<T> &b) {
        T val1 = ceil(a.i * 255.0 / 10.0);
        T val2 = ceil(b.i * 255.0 / 10.0);

        if (val1 > val2) {
                return true;
        } else if (val1 == val2) {
                if (a.s > b.s) {
                        return true;
                } else if (a.s == b.s) {
                        val1 = abs(a.h);
                        val2 = abs(b.h);

                        val1 = (val1 < HUE_TRESHOLD) ? val1 : 1 - val1;
                        val2 = (val2 < HUE_TRESHOLD) ? val2 : 1 - val2;

                        return (val1 < val2);
                }
        }

        return false;
}

template<typename T>
T __host__ __device__ HSIMatrix<T>::min3(T a, T b, T c) {
        if (a < b) {
                if (a < c) {
                        return a;
                } else {
                        return (b < c) ? b : c;
                }
        } else {
                if (b < c) {
                        return b;
                } else {
                        return (a < c) ? a : c;
                }
        }
}


template<typename T>
void __host__ __device__ HSIMatrix<T>::print() {
        printf("(%.16f, %.16f, %.16f)", h, s, i);
}

template<typename T>
void __host__ __device__ HSIMatrix<T>::printMatrix(HSIMatrix<T> *matrix, int width, int height, int lda) {
        HSIMatrix *current = matrix;

        for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                        current[j].print();
                }
                printf("\n");
                
                current += lda;         
        }
}

template<typename T>
void __host__ __device__ HSIMatrix<T>::printMatrixH(HSIMatrix<T> *matrix, int width, int height, int lda) {
        HSIMatrix *current = matrix;

        for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                        printf("%g ", current[j].h);;
                }
                printf("\n");

                current += lda;
        }
}

template<typename T>
void __host__ __device__ HSIMatrix<T>::printMatrixS(HSIMatrix<T> *matrix, int width, int height, int lda) {
        HSIMatrix *current = matrix;

        for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                        printf("%g ", current[j].s);;
                }
                printf("\n");

                current += lda;
        }
}

template<typename T>
void __host__ __device__ HSIMatrix<T>::printMatrixI(HSIMatrix<T> *matrix, int width, int height, int lda) {
        HSIMatrix *current = matrix;

        for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                        printf("%g ", current[j].i);;
                }
                printf("\n");

                current += lda;
        }
}

template<typename T>
template<typename S>
void __host__ __device__ HSIMatrix<T>::rgb2HSIMatrixVector(S *r, S *g, S *b, HSIMatrix<T> *vector, int size) {
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
                T r_val =  r[i];
                T g_val =  g[i];
                T b_val =  b[i];

                T num = 0.5 * ((r_val - g_val) + (r_val - b_val));
                T den = sqrt((r_val - g_val) * (r_val - g_val) + (r_val - b_val) * (g_val - b_val));
                if (den == 0) den = FLT_EPSILON;
                T theta = (T) acos(num / den);
                
                num =min3(r_val, g_val, b_val);
                den  = r_val + g_val + b_val;           
                if (den == 0) den = FLT_EPSILON;

                vector[i].i = den / 3;
                vector[i].s = 1 - 3 * (num / den);
                vector[i].h =  (vector[i].s == 0) ? 0 : (((b_val > g_val) ? (2 * PI - theta) : theta) / (2 * PI)); 
                vector[i].i /= RGB_MAX;
        }
}

template<typename T>
template<typename S>
void __host__ __device__ HSIMatrix<T>::HSIMatrixVector2rgb(HSIMatrix<T> *vector, S *r, S *g, S *b, int size) {
        T c = PI / 3;
        
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
                T h_val = vector[i].h * 2 * PI;
                T s_val = vector[i].s;
                T i_val = vector[i].i * RGB_MAX;

                if (0 <= h_val && h_val < 2 * c) {
                        b[i] = i_val * (1 - s_val);
                        r[i] = i_val * (1 + s_val * cos(h_val) / cos(c - h_val));
                        g[i] = 3 * i_val - (r[i] + b[i]);
                } else if (2 * c <= h_val && h_val <= 4 * c) {
                        r[i] = i_val * (1 - s_val);
                        g[i] = i_val * (1 + s_val * cos(h_val - 2 * c) / cos(PI - h_val));
                        b[i] = 3 * i_val - (r[i] + g[i]);
                } else {
                        g[i] = i_val * (1 - s_val);
                        b[i] = i_val * (1 + s_val * cos(h_val - 4 * c) / cos(5 * c - h_val));
                        r[i] = 3 * i_val - (b[i] + g[i]);
                }
        }
}

#endif
