#ifndef MORPH_CIRCLE_H
#define MORPH_CIRCLE_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "loewner_declaration.h"
#include "morph_color_matrix.h"

#ifndef KAPPA
#define KAPPA 0.707106781186547
#endif

#define N_MAPPING 10

#define NEWTON_EPSILON 1e-4
#define MAX_NEWTON_ITERATIONS 10000
#define NEWTON_ITERATIONS 4

/* 
 *================================================================================================================
 * Class representing a circle used in algorithm for solving smallest enclosing circle of circles problem.
 * It contains mathods for conversion to MorphColorMatrix object using different approaches, with or without
 * mapping the point from unit sphere to the m-hcl bi-cone described in the paper from B. Burgeth and A. Kleefeld.
 * 
 * AUTHOR: Filip Srnec
 * VERSION: 1.0
 *================================================================================================================
 */
class LoewnerMorphology::MorphCircle {
	public:
		double x;        // x coordinate of the centre
		double y;        // y coordinate of the centre
		double r;        // radius

		/*
	         * Constructs a MorphCircle with given centre (x, y) and radius r.
	   	 */
		__host__ __device__ MorphCircle(double x = 0, double y = 0, double r = 0) : x(x), y(y), r(r) {}
		
		/*
	  	 * Construct a MorphCircle from given MorphColorMatrix object (see "morph_color_matrix.h")
		 * using formula:
		 * 
		 *	x = 2 * KAPPA * b
		 *	y = KAPPA * (c - a)
		 *	r = KAPPA * (c + a)
		 *
		 * where (x, y) is a centre of newly constructed MorphCircle, r is a radius of MorphCircle and a, b and c be elements of given MorphColorMatrix ([a, b; b, c]).
		 */
		__host__ __device__ MorphCircle(const LoewnerMorphology::MorphColorMatrix &m);

		/*
	 	 * Performes basic conversion from MorphCircle to MorphColorMatrix without additional mapping.
	 	 * Conversion is performed using following formulas:
		 * 	
		 *      a = KAPPA * (z - y)
                 *      b = KAPPA * x
                 *      c = KAPPA * (z + y)
                 *
                 * where (x, y) is a centre of this MorphCircle, r is a radius of MorphCircle and a, b and c be elements of new MorphColorMatrix ([a, b; b, c]).
		 */
		LoewnerMorphology::MorphColorMatrix __host__ __device__ toMorphColorMatrix();
		
		/*
		 * Converts this MorphCircle to MorphColorMatrix using mapping from the unit sphere to the bi-cone from Theorem 4.1
		 * of the paper from B. Burgeth and A. Kleefeld.
		 */
		LoewnerMorphology::MorphColorMatrix __host__ __device__ toMorphColorMatrixCone1();
	
		/*
		 * Converts this MorphCircle to MorphColorMatrix using mapping from the unit sphere to the bi-cone from Theorem 4.4 from the paper of B. Burgeth and A. Kleefeld
		 * using Newton method with predifined number of iterations (constant NEWTON_ITERATIONS) for finding the root of the polynomial introduced in the paper.
		 */
		LoewnerMorphology::MorphColorMatrix __host__ __device__ toMorphColorMatrixCone2();
		
		/*
		 * Converts this MorphCircle to MorphColorMatrix using mapping from the unit sphere to the bi-cone from Theorem 4.4 from the paper of B. Burgeth and A. Kleefeld
		 * using Newton method with predfined epsilon constant (NEWTON_EPSILON) with predifined maximum number of iteration (NAX_NEWTON_ITERATIONS) for finding the root
		 * of the polynomial introduced in the paper.
		 */
		LoewnerMorphology::MorphColorMatrix __host__ __device__ toMorphColorMatrixCone2Epsilon();
		
		/*
		 * Converts tihs MorphCircle to MorphColorMatrix using mapping from bi-cone to the sphere from Corollary 4.3 from the paper of B. Burgeth and A. Kleefeld.
		 */
		LoewnerMorphology::MorphColorMatrix __host__ __device__ toMorphColorMatrixSphere();
		
		/*
		 * Prints this MorphCircle to the standard output in format '(x, y, r)'.
		 */
		void __host__ __device__ print();
		
		/*
		 * Resizes the radius of this MorphCircle.
		 */
		LoewnerMorphology::MorphCircle& __host__ __device__ resizeRadius(double r) { this->r = r; return *this; }
		
		/*
		 * Prepares this MorphCircle for calculating the maximum MorphCircle as described in paragraph 3 from the paper of B. Burgeth and A. Kleefeld.
		 */
		LoewnerMorphology::MorphCircle& __host__ __device__ prepareMax();
		
		/*
		 * Prepares this MorphCircle for calculating the minimum MorphCircle as discribed in paragraph 3 from the paper of B. Burgeth and A. Kleefeld.
		 */
		LoewnerMorphology::MorphCircle& __host__ __device__ prepareMin();
		

		/*
		 * Returns this MorphCircle in regular state after calculating the maximum MorphCircle as discrbed in paragraph 3 from the paper of B. Burgeth and A. Kleefeld
		 */
		LoewnerMorphology::MorphCircle& __host__ __device__ returnMax();


		/*
		 * Returns this MorphCircle in regular state after calculating the maximum MorphCircle as discrbed in paragraph 3 from the paper of B. Burgeth and A. Kleefeld
		 */
		LoewnerMorphology::MorphCircle& __host__ __device__ returnMin();
		
		/* 
		 * Checks if this MorphCircle corresponds to a point on the bi-cone.
		 */
		bool __host__ __device__ checkIfInCone();
		
		/*
		 * Prints x-coordinates of the center of all MorphCircle objects from the given matrix of MorphCircle objects.
		 */
		static void __host__ __device__ printMatrixX(LoewnerMorphology::MorphCircle *matrix, int width, int height, int lda);
		
		/*
		 * Prints y-coordinates of the center of all MorphCircle objects from the given matrix of MorphCircle objects.
		 */
		static void __host__ __device__ printMatrixY(LoewnerMorphology::MorphCircle *matrix, int width, int height, int lda);
		
		/*
		 * Prints radii of all MorphCircle objects from the given matrix of MorphCircle objects.
		 */
		static void __host__ __device__ printMatrixR(LoewnerMorphology::MorphCircle *matrix, int width, int height, int lda);
	
	private:
		/*
		 * Factory method that converts MorphCircle defined by given coordinates and radius to MorphColorMatrix object.
		 */
		static LoewnerMorphology::MorphColorMatrix __host__ __device__ inline toMorphColorMatrixFromCoordinates(double x, double y, double r);
		
		/*
		 * Computes lambda function introduced in paragraph 4 (2) from the paper of B. Burgeth and A. Kleefeld.
		 */
		double __host__ __device__ inline computeLambda();
		
		/*
		 * Evaluates the polynomial introduced in theorem 4.2 from the paper of B. Burgeth and A. Kleefeld (mi^11 - mi^10 + 1 - lambda).
		 * in given point mi and with given lambda.
		 */
		static double __host__ __device__ computeFunction(double mi, double lambda);
		
		/*
                 * Evaluates the derivation of the polynomial introduced in theorem 4.2 from the paper of B. Burgeth and A. Kleefeld (11 * mi^10 -10 *  mi^9).
                 * in given point mi.
                 */
		static double __host__ __device__ computeDerivation(double mi);

};


__host__ __device__ LoewnerMorphology::MorphCircle::MorphCircle(const LoewnerMorphology::MorphColorMatrix &m) : MorphCircle(2 * KAPPA * m.b, KAPPA * (m.c - m.a), KAPPA * (m.c + m.a)) {}


double __host__ __device__  inline LoewnerMorphology::MorphCircle::computeFunction(double mi, double lambda) {
        return pow(mi, 11) - pow(mi, 10) + 1.0 - lambda;
}

void __host__ __device__ LoewnerMorphology::MorphCircle::print() {
        printf("(%.15f,%.15f,%.15f)", x, y, r);
}

void __host__ __device__ LoewnerMorphology::MorphCircle::printMatrixX(LoewnerMorphology::MorphCircle *matrix, int width, int height, int lda) {
        MorphCircle *current = matrix;

        for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                        printf("%8.4f", current[j].x);;
                }
                printf("\n");

                current += lda;
        }
}

void __host__ __device__ LoewnerMorphology::MorphCircle::printMatrixY(LoewnerMorphology::MorphCircle *matrix, int width, int height, int lda) {
        MorphCircle *current = matrix;

        for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                        printf("%8.4f", current[j].y);;
                }
                printf("\n");

                current += lda;
        }
}

void __host__ __device__ LoewnerMorphology::MorphCircle::printMatrixR(LoewnerMorphology::MorphCircle *matrix, int width, int height, int lda) {
        MorphCircle *current = matrix;

        for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                        printf("%8.4f", current[j].r);;
                }
                printf("\n");

                current += lda;
        }
}

double __host__ __device__  inline LoewnerMorphology::MorphCircle::computeDerivation(double mi) {
        return 11.0 * pow(mi, 10) - 10.0 * pow(mi, 9);
}

LoewnerMorphology::MorphColorMatrix __host__ __device__ LoewnerMorphology::MorphCircle::toMorphColorMatrix() {
        return toMorphColorMatrixFromCoordinates(x, y, r);
}

LoewnerMorphology::MorphColorMatrix __host__ __device__ LoewnerMorphology::MorphCircle::toMorphColorMatrixCone1() {
        double lambda = computeLambda();

        return toMorphColorMatrixFromCoordinates(x / lambda, y / lambda, r / lambda);
}

LoewnerMorphology::MorphColorMatrix __host__ __device__ LoewnerMorphology::MorphCircle::toMorphColorMatrixCone2() {
        double lambda = 1.0 + pow(sqrt(x * x + y * y) + abs(r), N_MAPPING) * ((1.0 / computeLambda()) - 1);
        
        if (abs(1.0 - lambda) < NEWTON_EPSILON) {
                return toMorphColorMatrixFromCoordinates(x, y, r);
        }

        double x0 = 1.0;

        // Newton's iterations
        for (int i = 0; i < NEWTON_ITERATIONS; i++) {
                double f = computeFunction(x0, lambda);
                double fDerived = computeDerivation(x0);
                
                if (abs(fDerived) < FLT_EPSILON) {
                        printf("Derivation is too small to devide with.\n");
                        break;
                }

                x0 = x0 - f / fDerived;         
        }

        return toMorphColorMatrixFromCoordinates(x0 * x, x0 * y, x0 * r);
}       

LoewnerMorphology::MorphColorMatrix __host__ __device__ LoewnerMorphology::MorphCircle::toMorphColorMatrixCone2Epsilon() {
        double lambda = 1.0 + pow(sqrt(x * x + y * y) + abs(r), N_MAPPING) * (1.0 / computeLambda() - 1.0);
        
        if (abs(1.0 - lambda) < NEWTON_EPSILON) {
                return toMorphColorMatrixFromCoordinates(x, y, r);
        }

        double x0 = 1.0;
        bool solved = false;
        
        // Newton's iterations
        for (i = 0; i < MAX_NEWTON_ITERATIONS; i++) {
                double f = computeFunction(x0, lambda);
                double fDerived = computeDerivation(x0);

                if (abs(f) <= 1e-8) {
                        solved = true;
                        break;
                }

                if (abs(fDerived) < 1e-8) {
                        printf("Derivation is too small to devide with.\n");
                        break;
                }
                
                x0 = x0 - f / fDerived; 
        }

        if (!solved) {
                printf("Newton method did not converge. LAMBDA = %f f = %f, i %d\n", lambda, computeFunction(x0, lambda), i);
        }
        
        x0 = 1 / x0;

        return toMorphColorMatrixFromCoordinates(x0 * x, x0 * y, x0 * r);
}

bool __host__ __device__ LoewnerMorphology::MorphCircle::checkIfInCone() {
        return (x * x + y * y <= (1 - r) * (1 - r));
}

LoewnerMorphology::MorphColorMatrix __host__ __device__ inline LoewnerMorphology::MorphCircle::toMorphColorMatrixFromCoordinates(double x, double y, double r) {
        return MorphColorMatrix((r - y) * KAPPA, x * KAPPA, (r + y) * KAPPA);
}

double  __host__ __device__ inline LoewnerMorphology::MorphCircle::computeLambda() {
        if (x == 0 && y == 0) {
                return 1.0;
        }

        double lambda = abs(r)/ sqrt(x * x + y * y);

        return sqrt(1 + lambda * lambda) / (1 + lambda);
}

LoewnerMorphology::MorphCircle& __host__ __device__ LoewnerMorphology::MorphCircle::prepareMax() {
        r++;

        return *this;
}

LoewnerMorphology::MorphCircle& __host__ __device__ LoewnerMorphology::MorphCircle::prepareMin() {
        (this->r) = -(this->r) + 1;

        return *this;
}

LoewnerMorphology::MorphCircle& __host__ __device__ LoewnerMorphology::MorphCircle::returnMax() {
        (this->r)--;

        return *this;
}

LoewnerMorphology::MorphCircle& __host__ __device__ LoewnerMorphology::MorphCircle::returnMin() {
        (this->r) = -(this->r - 1);

        return *this;
}

LoewnerMorphology::MorphColorMatrix __host__ __device__ LoewnerMorphology::MorphCircle::toMorphColorMatrixSphere() {
	double lambda = 1.0 + pow(sqrt(x * x + y * y) + abs(r), N_MAPPING) * (1.0 / computeLambda() - 1.0);

	return toMorphColorMatrixFromCoordinates(lambda * x, lambda * y, lambda * r); 
}

#endif

