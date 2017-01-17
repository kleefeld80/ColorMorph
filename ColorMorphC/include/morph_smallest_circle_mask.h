#ifndef MORPH_SMALLEST_CIRCLE_MASK_H
#define MORPH_SMALLEST_CIRCLE_MASK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cmath>

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
 * is introduced in the paper of B. Burgeth and A. Kleefeld. The smallest circle of circles problem is solved numerically using subgradient
 * method.
 *===============================================================================================================================================
 */
class LoewnerMorphology::MorphSmallestCircleProblemMask {
	
	public:
		/*
		 * Construct a MorphSmallestCircleProblemMask object for solving Smallest enclosing circle of circles problem that consist of
		 * n MorphCircles stored on given memory location. MorphCircles are stored in a form of a matrix with given width, height and	
		 * lda. Mask is a pointer to the array of integer values representing a structuring element, all fields that contain zeros are
		 * ignored during the computation. The structuring element (the mask) has to have size width * height.
	 	 */
		MorphSmallestCircleProblemMask(LoewnerMorphology::MorphCircle *circles, int *mask, int width, int height, int lda);
		
		/*
		 * Computes the solution of the Smalest enclosing circle of circles problem and returns it as a MorphCircle object.
		 */
		LoewnerMorphology::MorphCircle compute();

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
		 * Calculates the starting point for the subgradient method. In this implementation the starting point is (0, 0).
		 */
		void startingPoint();

		/*
		 * Computes the value of the goal function in given point (x, y).
		 */
		double computeFunction(double x, double y);
		
		/*
		 * Computes the subgradient in kth step.
		 */
		void computeSubgradient();
	
		/*
		 * Computes alpha with a line search algorithm.
		 */
		double linesearchSubgradient();		 
};

#endif
