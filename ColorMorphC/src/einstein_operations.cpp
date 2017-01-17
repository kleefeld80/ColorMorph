#include "../include/einstein_operations.h"

LoewnerMorphology::MorphColorMatrix LoewnerMorphology::EinsteinOperations::einsteinAddition(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b) {
        double trAB = LoewnerMorphology::MorphColorMatrix::traceProduct(a, b);

        if (std::abs(trAB + 1.0) < EINSTEIN_EPSILON) {
                // hope we'll never get there (:
		return LoewnerMorphology::MorphColorMatrix(0.0, 0.0, 0.0);
        }

        double alpha = 1.0 - LoewnerMorphology::MorphColorMatrix::traceProduct(a, a);

        if (alpha < 0.0) {
                return a;
        }

        alpha = std::sqrt(alpha);

        return (1.0 / (1.0 + trAB)) * ((alpha * b) + ((trAB / (1.0 + alpha) + 1.0) * a));
}

LoewnerMorphology::MorphColorMatrix LoewnerMorphology::EinsteinOperations::einsteinAdditionMod(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b) {
        double alphaA = 1.0 - LoewnerMorphology::MorphColorMatrix::traceProduct(a, a);
        double alphaB = 1.0 - LoewnerMorphology::MorphColorMatrix::traceProduct(b, b);

        if (alphaA < 0.0) {
                alphaA = 0.0;
        }

        if (alphaB < 0.0) {
                alphaB = 0.0;
        }

        alphaA = std::sqrt(alphaA);
        alphaB = std::sqrt(alphaB);

        if (alphaA < EINSTEIN_MOD_EPSILON && alphaB < EINSTEIN_MOD_EPSILON) {
                if (std::sqrt(LoewnerMorphology::MorphColorMatrix::traceProduct(a + b, a + b)) < EINSTEIN_EPSILON) {
                        return LoewnerMorphology::MorphColorMatrix(0.0, 0.0, 0.0);
                }

                return einsteinAddition(a, a);
        }

        if (alphaA < EINSTEIN_MOD_EPSILON) {
                return einsteinAddition(a, a);
        }

        if (alphaB < EINSTEIN_MOD_EPSILON) {
                return einsteinAddition(b, b);
        }

        LoewnerMorphology::MorphColorMatrix temp = (1.0 / (alphaA + alphaB)) * ((alphaA * b) + (alphaB * a));
        
        return einsteinAddition(temp, temp);
}
