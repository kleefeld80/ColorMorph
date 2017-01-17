#ifndef MORPH_COLOR_MATRIX_H
#define MORPH_COLOR_MATRIX_H

#include "omp.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

/*
 *===================================================================================================================================================
 * MorphColorMatrix represents symetric 2x2 matrix defined by A = ([a, b], [b, c]) with a = KAPPA * (z - y), b = KAPPA * x and c = KAPPA * (z + y)
 * where x, y and z are coordinates of a point of a bi-cone obtained by x = c * cos(2 * PI * h), y = c * sin(2 * PI * h) and z = l. Values h, c and l
 * are values reached by conversion of RGB-value image to M-HCL-value image. This structure is used for morphology operation based on the
 * Loewner order. All methods are suitable to use on both host and device.
 *
 * version: 1.0
 * author: Filip Srnec
 *===================================================================================================================================================
 */

#ifndef PI
#define PI 3.141592653589793
#endif

#ifndef KAPPA
#define KAPPA 0.707106781186547
#endif

#ifndef RGB_MAX
#define RGB_MAX 255.0
#endif

struct LoewnerMorphology::MorphColorMatrix {
	double a;	// element a in the matrix (top-left)
	double b;	// element b in the matrix (top-right and bottom-left)
	double c;	// element c in the matrix (bottom-right)
		
	/*
 	 * Default constructor of MorphColorMatrix object. It creates new MorphColorMatrix that
	 * contains only zero elements.
         */
	__host__ __device__ MorphColorMatrix() : MorphColorMatrix(0.0, 0.0, 0.0) {}
	
	/*
	 * Constructor of MorphColorMatrix object. It creates new MorphColorMatrix object with
	 * given a, b and c values.
	 */
	__host__ __device__ MorphColorMatrix(double a, double b, double c) : a(a), b(b), c(c) {}
	
	/*
	 * Returns minimal MorphColorMatrix in terms of Loewner order among all matrices derived
	 * from the points of a bi-cone. Minimal MorphColorMatrix is unit 2x2 matrix multiplied
	 * by scalar -1 / sqrt(2) (-KAPPA constant).
	 */
	static LoewnerMorphology::MorphColorMatrix __host__ __device__ min();
	
	/*
         * Returns maximal MorphColorMatrix in terms of Loewner order among all matrices derived
         * from the points of a bi-cone. Maximal MorphColorMatrix is unit 2x2 matrix multiplied
         * by scalar 1 / sqrt(2) (KAPPA constant).
         */
	static LoewnerMorphology::MorphColorMatrix __host__ __device__ max();
	
	/*
 	 * Prints MorphColorMatrix elements in form (a, b, c).
	 */
	void __host__ __device__ printMorphColorMatrix() const;
	
	/*
	 * Prints top left element (a) of each MorphColorMatrix in provided matrix of MorphColorMatrices.
	 */
	static void __host__ __device__ printMatrixA(const LoewnerMorphology::MorphColorMatrix *matrix, int width, int height, int lda);
	
	/*
	 * Prints antidiagonal element (b) of each MorphColorMatrix in provided matrix of MorphColoMatrices.
	 */
	static void __host__ __device__ printMatrixB(const LoewnerMorphology::MorphColorMatrix *matrix, int width, int height, int lda);
	
	/*
	 * Prints bottom right element (c) of each MorphColorMatrix in provided matrix of MorphColorMatrices.
	 */
	static void __host__ __device__ printMatrixC(const LoewnerMorphology::MorphColorMatrix *matrix, int width, int height, int lda);

	/*
	 * Performs shift of this MorphColorMatrix matrix. In other words, performs operation 
	 * M = M + alpha * I where alpha is double, and I 2x2 identity matrix.
	 */
	LoewnerMorphology::MorphColorMatrix& __host__ __device__ shift(double alpha);
	
	/*
	 * Negates each element of this MorphColorMatrix.
	 */
	LoewnerMorphology::MorphColorMatrix& __host__ __device__ negate();
	
	/*
	 * Returns trace of this MorphColorMatrix.
	 */
	double __host__ __device__ trace() const;
	
	/*
	 * Returns trace of the matrix A * B.
	 */
	static double __host__ __device__ traceProduct(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b);
	
	/*
	 * Returns Frobenius norm of this MorphColorMatrix.
	 */
	double __host__ __device__ norm() const;

	/*
	 * Operator that represents standard matrix subtraction of two MorphColorMatrix objects.
	 */
	friend LoewnerMorphology::MorphColorMatrix __host__ __device__ operator-(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b);
	
	/*
	 * Operator that represents standard matrix addition of two MorphColorMatrix objects.
         */
	friend LoewnerMorphology::MorphColorMatrix __host__ __device__ operator+(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b);

	/*
	 * Operator that represents standard scalar - MorphColorMatrix multiplication.
	 */
	friend LoewnerMorphology::MorphColorMatrix __host__ __device__ operator*(const double alpha, const LoewnerMorphology::MorphColorMatrix &a);
	
	/*
	 * Comparison operator >= which takes tho MorphColorMatrix objects as arguments.
	 * Matrices are compared using Loewner's order.
	 */
	friend bool __host__ __device__ operator>=(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b);
	
	/*
	 * Comparison operator <= which takes tho MorphColorMatrix objects as arguments.
	 * Matrices are compared using Loewner's order.
	 */
	friend bool __host__ __device__ operator<=(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b);
	
	/*
	 * Comparison operator =. MorphColorMatrix objects are equal if their elements are equal.
	 */
	friend bool __host__ __device__ operator==(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b);
	
	/*
	 * Comparison operator !=. MorphColorMatrix objects are equal if their elements are equal.
	 */
	friend bool __host__ __device__ operator!=(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b);
};

// IMPLEMENTATION

LoewnerMorphology::MorphColorMatrix __host__ __device__  LoewnerMorphology::MorphColorMatrix::min() {	
	return MorphColorMatrix(-KAPPA, 0, -KAPPA);
}	


LoewnerMorphology::MorphColorMatrix __host__ __device__ LoewnerMorphology::MorphColorMatrix::max() {
	return MorphColorMatrix(KAPPA, 0, KAPPA);
}

void __host__ __device__ LoewnerMorphology::MorphColorMatrix::printMatrixA(const LoewnerMorphology::MorphColorMatrix *matrix, int width, int height, int lda) {
        const LoewnerMorphology::MorphColorMatrix *current = matrix;

        for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                        printf("%10.6f", current[j].a);;
                }
                printf("\n");

                current += lda;
        }
}

void __host__ __device__ LoewnerMorphology::MorphColorMatrix::printMatrixB(const LoewnerMorphology::MorphColorMatrix *matrix, int width, int height, int lda) {
        const LoewnerMorphology::MorphColorMatrix *current = matrix;

        for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                        printf("%10.6f", current[j].b);;
                }
                printf("\n");

                current += lda;
        }
}

void __host__ __device__ LoewnerMorphology::MorphColorMatrix::printMatrixC(const LoewnerMorphology::MorphColorMatrix *matrix, int width, int height, int lda) {
        const LoewnerMorphology::MorphColorMatrix *current = matrix;

        for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                        printf("%10.6f", current[j].c);;
                }
                printf("\n");

                current += lda;
        }
}

void __host__ __device__ LoewnerMorphology::MorphColorMatrix::printMorphColorMatrix() const {
	printf("(%.15f %.15f %.15f)\n", a, b, c);
}


LoewnerMorphology::MorphColorMatrix __host__ __device__ LoewnerMorphology::operator-(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b) {
	return LoewnerMorphology::MorphColorMatrix(a.a - b.a, a.b - b.b, a.c - b.c);
}

LoewnerMorphology::MorphColorMatrix __host__ __device__ LoewnerMorphology::operator+(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b) {
        return LoewnerMorphology::MorphColorMatrix(a.a + b.a, a.b + b.b, a.c + b.c);
}

LoewnerMorphology::MorphColorMatrix __host__ __device__ LoewnerMorphology::operator*(const double alpha, const LoewnerMorphology::MorphColorMatrix &a) {
	return LoewnerMorphology::MorphColorMatrix(alpha * a.a, alpha * a.b, alpha * a.c);
}

bool __host__ __device__ LoewnerMorphology::operator>=(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b) {
	LoewnerMorphology::MorphColorMatrix temp = a - b;

	return (temp.a >= 0 && (temp.a * temp.c - temp.b * temp.b) >= 0);	
}

bool __host__ __device__ LoewnerMorphology::operator<=(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b) {
        return (b >= a);
}

bool __host__ __device__ LoewnerMorphology::operator>(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b) {
        LoewnerMorphology::MorphColorMatrix temp = a - b;

        return (temp.a > 0 && (temp.a * temp.c - temp.b * temp.b) >= 0);
}

bool __host__ __device__ LoewnerMorphology::operator<(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b) {
        return (b > a);
}

bool __host__ __device__ LoewnerMorphology::operator==(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b) {
	LoewnerMorphology::MorphColorMatrix c = a - b;

	return (abs(c.a) < FLT_EPSILON) && (abs(c.b) < FLT_EPSILON) && (abs(c.c) < FLT_EPSILON);
}

bool __host__ __device__ LoewnerMorphology::operator!=(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b) {
        return !(a == b);
}

double __host__ __device__ LoewnerMorphology::MorphColorMatrix::trace() const {
	return a + c;
}

double __host__ __device__ LoewnerMorphology::MorphColorMatrix::traceProduct(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b) {
	return a.a * b.a + 2 * a.b * b.b + a.c * b.c; 
} 

double __host__ __device__ LoewnerMorphology::MorphColorMatrix::norm() const {
	return sqrt(traceProduct(*this, *this));
}

LoewnerMorphology::MorphColorMatrix& __host__ __device__ LoewnerMorphology::MorphColorMatrix::shift(double alpha) {
	a += alpha;
	c += alpha;

	return *this;
}

LoewnerMorphology::MorphColorMatrix& __host__ __device__ LoewnerMorphology::MorphColorMatrix::negate() {
	a = -a;
	b = -b;
	c = -c;

	return *this;
}

#endif
