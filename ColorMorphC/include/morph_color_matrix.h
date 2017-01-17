#ifndef MORPH_COLOR_MATRIX_H
#define MORPH_COLOR_MATRIX_H

#include "omp.h"

#include "loewner_declaration.h"

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include <cmath>

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
	MorphColorMatrix() : MorphColorMatrix(0.0, 0.0, 0.0) {}
	
	/*
	 * Constructor of MorphColorMatrix object. It creates new MorphColorMatrix object with
	 * given a, b and c values.
	 */
	MorphColorMatrix(double a, double b, double c) : a(a), b(b), c(c) {}
	
	/*
	 * Returns minimal MorphColorMatrix in terms of Loewner order among all matrices derived
	 * from the points of a bi-cone. Minimal MorphColorMatrix is unit 2x2 matrix multiplied
	 * by scalar -1 / sqrt(2) (-KAPPA constant).
	 */
	static LoewnerMorphology::MorphColorMatrix min();
	
	/*
         * Returns maximal MorphColorMatrix in terms of Loewner order among all matrices derived
         * from the points of a bi-cone. Maximal MorphColorMatrix is unit 2x2 matrix multiplied
         * by scalar 1 / sqrt(2) (KAPPA constant).
         */
	static LoewnerMorphology::MorphColorMatrix max();
	
	/*
 	 * Prints MorphColorMatrix elements in form (a, b, c).
	 */
	void printMorphColorMatrix() const;
	
	/*
	 * Prints top left element (a) of each MorphColorMatrix in provided matrix of MorphColorMatrices.
	 */
	static void printMatrixA(const LoewnerMorphology::MorphColorMatrix *matrix, int width, int height, int lda);
	
	/*
	 * Prints antidiagonal element (b) of each MorphColorMatrix in provided matrix of MorphColoMatrices.
	 */
	static void printMatrixB(const LoewnerMorphology::MorphColorMatrix *matrix, int width, int height, int lda);
	
	/*
	 * Prints bottom right element (c) of each MorphColorMatrix in provided matrix of MorphColorMatrices.
	 */
	static void printMatrixC(const LoewnerMorphology::MorphColorMatrix *matrix, int width, int height, int lda);

	/*
	 * Performs shift of this MorphColorMatrix matrix. In other words, performs operation 
	 * M = M + alpha * I where alpha is double, and I 2x2 identity matrix.
	 */
	LoewnerMorphology::MorphColorMatrix& shift(double alpha);
	
	/*
	 * Negates each element of this MorphColorMatrix.
	 */
	LoewnerMorphology::MorphColorMatrix& negate();
	
	/*
	 * Returns new MorphColorMatrix representing the negation of this MorphColorMatrix.
	 */
	LoewnerMorphology::MorphColorMatrix negation();

	/*
	 * Returns trace of this MorphColorMatrix.
	 */
	double trace() const;
	
	/*
	 * Returns trace of the matrix A * B.
	 */
	static double traceProduct(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b);
	
	/*
	 * Returns Frobenius norm of this MorphColorMatrix.
	 */
	double norm() const;

	/*
	 * Operator that represents standard matrix subtraction of two MorphColorMatrix objects.
	 */
	friend LoewnerMorphology::MorphColorMatrix operator-(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b);
	
	/*
	 * Operator that represents standard matrix addition of two MorphColorMatrix objects.
         */
	friend LoewnerMorphology::MorphColorMatrix operator+(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b);

	/*
	 * Operator that represents standard scalar - MorphColorMatrix multiplication.
	 */
	friend LoewnerMorphology::MorphColorMatrix operator*(const double alpha, const LoewnerMorphology::MorphColorMatrix &a);
	
	/*
	 * Comparison operator >= which takes tho MorphColorMatrix objects as arguments.
	 * Matrices are compared using Loewner's order.
	 */
	friend bool operator>=(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b);
	
	/*
	 * Comparison operator <= which takes tho MorphColorMatrix objects as arguments.
	 * Matrices are compared using Loewner's order.
	 */
	friend bool operator<=(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b);
	
	/*
	 * Comparison operator =. MorphColorMatrix objects are equal if their elements are equal.
	 */
	friend bool operator==(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b);
	
	/*
	 * Comparison operator !=. MorphColorMatrix objects are equal if their elements are equal.
	 */
	friend bool operator!=(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b);
};

#endif
