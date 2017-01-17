#ifndef EINSTEIN_OPERATIONS_H
#define EINSTEIN_OPERATIONS_H

#include "loewner_declaration.h"
#include "morph_color_matrix.h"

#define EINSTEIN_EPSILON 1e-7
#define EINSTEIN_MOD_EPSILON 1e-5

class LoewnerMorphology::EinsteinOperations {
	
	public:
		//Performs Einstein addition on tho MorphColorMatrix
		static LoewnerMorphology::MorphColorMatrix __host__ __device__ einsteinAddition(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b);
		//Performs modified commutative version of Einstein addition
		static LoewnerMorphology::MorphColorMatrix __host__ __device__ einsteinAdditionMod(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b);
};

/*
 * Performs Einstein addition of two given matrices.
 */
LoewnerMorphology::MorphColorMatrix __host__ __device__ LoewnerMorphology::EinsteinOperations::einsteinAddition(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b) {
        double trAB = LoewnerMorphology::MorphColorMatrix::traceProduct(a, b);

        if (abs(trAB + 1.0) < EINSTEIN_EPSILON) {
                // hope we'll never get there (:
                return LoewnerMorphology::MorphColorMatrix(0.0, 0.0, 0.0);
        }

        double alpha = 1.0 - LoewnerMorphology::MorphColorMatrix::traceProduct(a, a);

        if (alpha < 0) {
                return a;
        }

        alpha = sqrt(alpha);

        return (1.0 / (1.0 + trAB)) * ((alpha * b) + ((trAB / (1.0 + alpha) + 1.0) * a));
}

/*
 * Performs modified commutative version of Einstein addition explained in the paper.
 */
LoewnerMorphology::MorphColorMatrix __host__ __device__ LoewnerMorphology::EinsteinOperations::einsteinAdditionMod(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b) {
        double alphaA = 1 - LoewnerMorphology::MorphColorMatrix::traceProduct(a, a);
        double alphaB = 1 - LoewnerMorphology::MorphColorMatrix::traceProduct(b, b);

        if (alphaA < 0) {
                alphaA = 0;
        }

        if (alphaB < 0) {
                alphaB = 0;
        }

        alphaA = sqrt(alphaA);
        alphaB = sqrt(alphaB);

        if (alphaA < EINSTEIN_MOD_EPSILON && alphaB < EINSTEIN_MOD_EPSILON) {
                if (sqrt(LoewnerMorphology::MorphColorMatrix::traceProduct(a + b, a + b)) < EINSTEIN_EPSILON) {
                        //printf("TU SAM1\n");
                        return LoewnerMorphology::MorphColorMatrix(0.0, 0.0, 0.0);
                }

                //printf("TU SAM2\n");
                return einsteinAddition(a, a);
        }

        if (alphaA < EINSTEIN_MOD_EPSILON) {
                //printf("TU SAM3\n");
                return einsteinAddition(a, a);
        }

        if (alphaB < EINSTEIN_MOD_EPSILON) {
                //printf("TU SAM4\n");
                return einsteinAddition(b, b);
        }

        LoewnerMorphology::MorphColorMatrix temp = (1.0 / (alphaA + alphaB)) * ((alphaA * b) + (alphaB * a));
        //printf("TU SAM5\n");
        return einsteinAddition(temp, temp);
}

#endif
