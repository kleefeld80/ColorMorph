#ifndef MORPH_SMALLEST_CIRCLE_MASK_H
#define MORPH_SMALLEST_CIRCLE_MASK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "loewner_declaration.h"
#include "morph_circle.h"

#define ALPHA_EPSILON 1e-5
#define LINE_SEARCH_EPSILON 1e-5
#define MAX_NUM_ITERATIONS 100
#define SUBGRADIENT_EPSILON 1e-8
#define GRAD_EPSILON 1e-7

#define DISTANCE_LIMIT 2.0       // after computing smaller circle problem solution this is the max distance from the origin
#define DISTANCE_CONSTANT 0.0001 // constant to be substracted from the radius after scaling

/*
 *===============================================================================================================================================
 * Class used for solving the Smallest circle of circles problem used in computation of morphological operations based on Loewner ordening which
 * is introduced in the paper of B. Burgeth and A. Kleefeld. Smallest circle of circles problem is solved numerically using subgradient method.
 *===============================================================================================================================================
 */
class LoewnerMorphology::MorphSmallestCircleProblemMask {
	
	public:
		/*
		 * Construct MorphSmallestCircleProblemMask object for solving Smallest enclosing circle of circles problem that consist of n MorphCircles 
		 * stored given memory location. MorphCircles are stored in a form of matrix with given width, hight and lda. Mask is a pointer to
	 	 * to the array of integer values representing mask for a given matrix, all fields with zeros are ignored during the computation. Mask has to
		 * have size width * height.
	 	 */
		__host__ __device__ MorphSmallestCircleProblemMask(LoewnerMorphology::MorphCircle *circles, int *mask, int width, int height, int lda);
		
		/*
		 * Computes the solution of Smalest enclosing circle of circles problem and returns it as a MorphCircle object.
		 */
		LoewnerMorphology::MorphCircle __host__ __device__ compute();

	private:
		int width;		// width of the matrix
		int height;		// height of the matrix
		int lda;		// lda of the matrix

		LoewnerMorphology::MorphCircle *circles;	// array of circles
		int *mask;		// mask pointer

		double xk;		// x coordinate of the centre of the circle in kth step
		double yk;		// y coordinate of the centre of the cricle in kth step

		double gradX;		// x coordinate of the computed gradient in kth step
		double gradY;		// y coordinate of the computed gradient in kth step

		/*
		 * Calculates the starting point for the subgradient method. In this implementation starting point is (0, 0).
		 */
		void __host__ __device__ startingPoint();

		/*
		 * Computes value of the goal function in given point (x, y).
		 */
		double __host__ __device__ computeFunction(double x, double y);
		
		/*
		 * Computes subgradient in kth step.
		 */
		void __host__ __device__ computeSubgradient();
	
		/*
		 * Computes alpha with a line search algorithm.
		 */
		double __host__ __device__ linesearchSubgradient();		 
};

// IMPLEMENTATION

__host__ __device__ LoewnerMorphology::MorphSmallestCircleProblemMask::MorphSmallestCircleProblemMask(LoewnerMorphology::MorphCircle *circles, int *mask, int width, int height, int lda) : width(width), height(height), lda(lda), circles(circles), mask(mask) {}

void __host__ __device__ LoewnerMorphology::MorphSmallestCircleProblemMask::startingPoint() {
	xk = 0.0;
	yk = 0.0;
}

void __host__ __device__ LoewnerMorphology::MorphSmallestCircleProblemMask::computeSubgradient() {
	gradX = 0.0;
	gradY = 0.0;
	double max = computeFunction(xk, yk);

	LoewnerMorphology::MorphCircle *current = circles;

        for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                        if (mask[i * width + j]) {
                                LoewnerMorphology::MorphCircle temp = current[j];
                        	double value  = sqrt((temp.x - xk) * (temp.x - xk) + (temp.y - yk) * (temp.y - yk)) + temp.r;

                        	if (abs(max - value) < SUBGRADIENT_EPSILON) {
                                	gradX += temp.x - xk;
                                	gradY += temp.y - yk;
                        	}
                        }
                }
                current += lda;
        }
}

double __host__ __device__ LoewnerMorphology::MorphSmallestCircleProblemMask::computeFunction(double x, double y) {
	double max = 0.0;

	LoewnerMorphology::MorphCircle *current = circles;

        for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                        if (mask[i * width + j]) {
                                LoewnerMorphology::MorphCircle temp = current[j];
                                double value = sqrt((temp.x - x) * (temp.x - x) + (temp.y - y) * (temp.y - y)) + temp.r;

                        	if (value > max) {
                                	max = value;
                        	}
                        }
                }
                current += lda;
        }

	return max;
}

double __host__ __device__ LoewnerMorphology::MorphSmallestCircleProblemMask::linesearchSubgradient() {
	double left = 0.0;
	double right = 1.0;
	double alpha = 0.0;

	while (right - left > LINE_SEARCH_EPSILON) {
		alpha = (right + left) / 2;
		double rr = computeFunction(xk + 1.01 * alpha * gradX, yk + 1.01 * alpha * gradY);
		double rl = computeFunction(xk + 0.99 * alpha * gradX, yk + 0.99 * alpha * gradY);

		if (rr > rl) {
			right = alpha;	
		} else if (rr == rl) {
			return alpha;
		} else {
			left = alpha;
		}
	}

	return alpha;
}

LoewnerMorphology::MorphCircle __host__ __device__ LoewnerMorphology::MorphSmallestCircleProblemMask::compute() {
	startingPoint();
	
	double alpha = 1.0;
	int counter = 0;

	while (true) {
		if (abs(alpha) <= ALPHA_EPSILON) break;
	
		computeSubgradient();
		alpha = linesearchSubgradient();

		xk += alpha * gradX;
		yk += alpha * gradY;

		if (sqrt(gradX * gradX + gradY * gradY) < GRAD_EPSILON) break;
		
		if (counter >= MAX_NUM_ITERATIONS) break;
	
		counter++;
	}

	double result = computeFunction(xk, yk);
	
	// checking if radius needs to be reduced
	double dist = sqrt(xk * xk + yk * yk);

	if (dist + result > DISTANCE_LIMIT) {
		result = DISTANCE_LIMIT - dist - DISTANCE_CONSTANT;
	}

	return LoewnerMorphology::MorphCircle(xk, yk, result);	
}

#endif
