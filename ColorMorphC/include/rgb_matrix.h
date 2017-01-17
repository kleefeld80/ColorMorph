#ifndef RGB_MATRIX_H
#define RGB_MATRIX_H

#include <stdio.h>
#include <cmath>

#ifndef PIX_MAX_NORM
#define PIX_MAX_NORM 1
#endif

#ifndef PIX_MIN_NORM
#define PIX_MIN_NORM 0
#endif

/*
 * Represents symetric 2x2 matrix that contains RGB values of the given pixel.
 * Top left element represents R value, bottom left B value and other 2 elements G value.
 */
template<typename T>
struct RGBMatrix {
        T r;
        T g;
        T b;

        //Constructor of struct RGBmatrix
        RGBMatrix(T r, T g, T b) : r(r), g(g), b(b) {}
        //Default constructor of struct RGBMatrix. All values are set to 0.0
        RGBMatrix() : RGBMatrix(0.0, 0.0, 0.0) {}

        /*
         * Converts the image vector containing rgb values to the vector containing
         * rgb matrices. Memory for the new vector needs to be allocated and passed
         * as a vector argument. It should be array of RGBMatrix values which size is equal
         * as the size of the image. Pointers r, g and b are the pointers to the values
         * of the red, green and blue
         */
        static void rgb2matrix(T *r, T *g, T *b, RGBMatrix<T> *vector, int size);

        /*
         * Converts the vector containing RGBMatrices to vector containina RGB values.
         */
        static void matrix2rgb(RGBMatrix<T> *vector, T *r, T *g, T *b, int size);

        //Prints RGBMatrix
        void print();
        //Returns minimal RGBMatrix
        static RGBMatrix min();
        //Return maximal RGBMatrix
        static RGBMatrix max();
        //Represents operator <. Matrices are ordered in terms of lexicographic order due to the RGB values
        template<typename U>
        friend bool operator<(RGBMatrix<U> &a, RGBMatrix<U> &b);
        //Represents operator >. Matrices are ordered in terms of lexicographic order due to the RGB values
        template<typename U>
        friend bool operator>(RGBMatrix<U> &a, RGBMatrix<U> &b);
};

template<typename T>
void RGBMatrix<T>::print() {
        printf("(%f, %f, %f)", r, g, b);
}

template<typename T>
void RGBMatrix<T>::rgb2matrix(T *r, T *g, T *b, RGBMatrix<T> *vector, int size) {
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
                RGBMatrix *temp = vector + i;
                temp->r = r[i];
                temp->g = g[i];
                temp->b = b[i];
        }
}

template<typename T>
void RGBMatrix<T>::matrix2rgb(RGBMatrix<T> *vector, T *r, T *g, T *b, int size) {
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
                RGBMatrix *temp = vector + i;
                r[i] = temp->r;
                g[i] = temp->g;
                b[i] = temp->b;
        }
}

template<typename T>
bool operator<(RGBMatrix<T> &a, RGBMatrix<T> &b) {
        if (a.r < b.r) {
                return true;
        } else if (a.r == b.r) {
                if (a.g < b.g) {
                        return true;
                } else if (a.g == b.g) {
                        return (a.b < b.b);
                } else {
                        return false;
                }
        } else {
                return false;
        }
}

template<typename T>
bool operator>(RGBMatrix<T> &a, RGBMatrix<T> &b) {
        if (a.r > b.r) {
                return true;
        } else if (a.r == b.r) {
                if (a.g > b.g) {
                        return true;
                } else if (a.g == b.g) {
                        return (a.b > b.b);
                } else {
                        return false;
                }
        } else {
                return false;
        }
}

template<typename T>
RGBMatrix<T> RGBMatrix<T>::max() {
        return RGBMatrix<T>(PIX_MAX_NORM, PIX_MAX_NORM, PIX_MAX_NORM);
}

template<typename T>
RGBMatrix<T> RGBMatrix<T>::min() {
        return RGBMatrix<T>(PIX_MIN_NORM, PIX_MIN_NORM, PIX_MIN_NORM);
}

#endif
