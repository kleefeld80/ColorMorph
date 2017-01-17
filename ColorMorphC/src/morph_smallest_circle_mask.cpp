#include "../include/morph_smallest_circle_mask.h"

LoewnerMorphology::MorphSmallestCircleProblemMask::MorphSmallestCircleProblemMask(LoewnerMorphology::MorphCircle *circles, int *mask, int width, int height, int lda) : width(width), height(height), lda(lda), circles(circles), mask(mask) {}

void LoewnerMorphology::MorphSmallestCircleProblemMask::startingPoint() {
	xk = 0.0;
	yk = 0.0;
}

void LoewnerMorphology::MorphSmallestCircleProblemMask::computeSubgradient() {
	gradX = 0.0;
	gradY = 0.0;
	double max = computeFunction(xk, yk);

	LoewnerMorphology::MorphCircle *current = circles;

        for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                        if (mask[i * width + j]) {
                                LoewnerMorphology::MorphCircle temp = current[j];
                        	double value  = std::sqrt((temp.x - xk) * (temp.x - xk) + (temp.y - yk) * (temp.y - yk)) + temp.r;

                        	if (std::abs(max - value) < SUBGRADIENT_EPSILON) {
                                	gradX += temp.x - xk;
                                	gradY += temp.y - yk;
                        	}
                        }
                }
                current += lda;
        }
}

double LoewnerMorphology::MorphSmallestCircleProblemMask::computeFunction(double x, double y) {
	double max = 0.0;

	LoewnerMorphology::MorphCircle *current = circles;

        for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                        if (mask[i * width + j]) {
                                LoewnerMorphology::MorphCircle temp = current[j];
                                double value = std::sqrt((temp.x - x) * (temp.x - x) + (temp.y - y) * (temp.y - y)) + temp.r;

                        	if (value > max) {
                                	max = value;
                        	}
                        }
                }
                current += lda;
        }

	return max;
}

double LoewnerMorphology::MorphSmallestCircleProblemMask::linesearchSubgradient() {
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

LoewnerMorphology::MorphCircle LoewnerMorphology::MorphSmallestCircleProblemMask::compute() {
	startingPoint();
	
	double alpha = 1.0;
	int counter = 0;

	while (true) {
		if (std::abs(alpha) <= ALPHA_EPSILON) break;
	
		computeSubgradient();
		alpha = linesearchSubgradient();

		xk += alpha * gradX;
		yk += alpha * gradY;

		if (std::sqrt(gradX * gradX + gradY * gradY) < GRAD_EPSILON) break;
		
		if (counter >= MAX_NUM_ITERATIONS) break;
	
		counter++;
	}

	double result = computeFunction(xk, yk);
	
	// checking if radius needs to be reduced
	double dist = std::sqrt(xk * xk + yk * yk);

	if (dist + result > DISTANCE_LIMIT) {
		result = DISTANCE_LIMIT - dist - DISTANCE_CONSTANT;
	}

	return LoewnerMorphology::MorphCircle(xk, yk, result);	
}
