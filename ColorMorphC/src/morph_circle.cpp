#include "../include/morph_circle.h"

LoewnerMorphology::MorphCircle::MorphCircle(const LoewnerMorphology::MorphColorMatrix &m) : MorphCircle(2 * KAPPA * m.b, KAPPA * (m.c - m.a), KAPPA * (m.c + m.a)) {}

double  inline LoewnerMorphology::MorphCircle::computeFunction(double mi, double lambda) {
        return pow(mi, 11) - pow(mi, 10) + 1.0 - lambda;
}

void LoewnerMorphology::MorphCircle::print() {
        printf("(%.15f,%.15f,%.15f)", x, y, r);
}

void LoewnerMorphology::MorphCircle::printMatrixX(LoewnerMorphology::MorphCircle *matrix, int width, int height, int lda) {
        MorphCircle *current = matrix;

        for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                        printf("%8.4f", current[j].x);;
                }
                printf("\n");

                current += lda;
        }
}

void LoewnerMorphology::MorphCircle::printMatrixY(LoewnerMorphology::MorphCircle *matrix, int width, int height, int lda) {
        MorphCircle *current = matrix;

        for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                        printf("%8.4f", current[j].y);;
                }
                printf("\n");

                current += lda;
        }
}

void LoewnerMorphology::MorphCircle::printMatrixR(LoewnerMorphology::MorphCircle *matrix, int width, int height, int lda) {
        MorphCircle *current = matrix;

        for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                        printf("%8.4f", current[j].r);;
                }
                printf("\n");

                current += lda;
        }
}

double  inline LoewnerMorphology::MorphCircle::computeDerivation(double mi) {
        return 11.0 * pow(mi, 10) - 10.0 * pow(mi, 9);
}

LoewnerMorphology::MorphColorMatrix LoewnerMorphology::MorphCircle::toMorphColorMatrix() {
        return toMorphColorMatrixFromCoordinates(x, y, r);
}

LoewnerMorphology::MorphColorMatrix LoewnerMorphology::MorphCircle::toMorphColorMatrixCone1() {
        double lambda = computeLambda();

        return toMorphColorMatrixFromCoordinates(x / lambda, y / lambda, r / lambda);
}

LoewnerMorphology::MorphColorMatrix LoewnerMorphology::MorphCircle::toMorphColorMatrixCone2() {
        double lambda = 1.0 + pow(sqrt(x * x + y * y) + std::abs(r), N_MAPPING) * ((1.0 / computeLambda()) - 1);
        
        if (std::abs(1.0 - lambda) < NEWTON_EPSILON) {
                return toMorphColorMatrixFromCoordinates(x, y, r);
        }

        double x0 = 1.0;

        // Newton's iterations
        for (int i = 0; i < NEWTON_ITERATIONS; i++) {
                double f = computeFunction(x0, lambda);
                double fDerived = computeDerivation(x0);
                
                if (std::abs(fDerived) < FLT_EPSILON) {
                        printf("Derivation is too small to devide with.\n");
                        break;
                }

                x0 = x0 - (f / fDerived);         
        }

        return toMorphColorMatrixFromCoordinates(x0 * x, x0 * y, x0 * r);
}       

LoewnerMorphology::MorphColorMatrix LoewnerMorphology::MorphCircle::toMorphColorMatrixCone2Epsilon() {
        double lambda = 1.0 + pow(sqrt(x * x + y * y) + std::abs(r), N_MAPPING) * (1.0 / computeLambda() - 1.0);
        
        if (std::abs(1.0 - lambda) < NEWTON_EPSILON) {
                return toMorphColorMatrixFromCoordinates(x, y, r);
        }

        double x0 = 1.0;
        bool solved = false;
        int i = 0;

        // Newton's iterations
        for (i = 0; i < MAX_NEWTON_ITERATIONS; i++) {
                double f = computeFunction(x0, lambda);
                double fDerived = computeDerivation(x0);

                if (std::abs(f) <= 1e-8) {
                        solved = true;
                        break;
                }

                if (std::abs(fDerived) < 1e-8) {
                        printf("Derivation is too small to devide with.\n");
                        break;
                }
                
                x0 = x0 - f / fDerived; 
        }

        if (!solved) {
                printf("Newton method did not converge. LAMBDA = %f f = %f, i %d\n", lambda, computeFunction(x0, lambda), i);
        }
        
        x0 = 1.0 / x0;

        return toMorphColorMatrixFromCoordinates(x0 * x, x0 * y, x0 * r);
}

bool LoewnerMorphology::MorphCircle::checkIfInCone() {
        return (x * x + y * y <= (1.0 - r) * (1.0 - r));
}

LoewnerMorphology::MorphColorMatrix inline LoewnerMorphology::MorphCircle::toMorphColorMatrixFromCoordinates(double x, double y, double r) {
	return MorphColorMatrix((r - y) * KAPPA, x * KAPPA, (r + y) * KAPPA);
}

double LoewnerMorphology::MorphCircle::computeLambda() {
        if (x == 0.0 && y == 0.0) {
                return 1.0;
        }

        double lambda = std::abs(r)/ sqrt(x * x + y * y);

        return sqrt(1.0 + lambda * lambda) / (1.0 + lambda);
}

LoewnerMorphology::MorphCircle& LoewnerMorphology::MorphCircle::prepareMax() {
        r++;

        return *this;
}

LoewnerMorphology::MorphCircle& LoewnerMorphology::MorphCircle::prepareMin() {
        (this->r) = -(this->r) + 1.0;

        return *this;
}

LoewnerMorphology::MorphCircle& LoewnerMorphology::MorphCircle::returnMax() {
        (this->r)--;

        return *this;
}

LoewnerMorphology::MorphCircle& LoewnerMorphology::MorphCircle::returnMin() {
        (this->r) = -(this->r - 1.0);

        return *this;
}

LoewnerMorphology::MorphColorMatrix LoewnerMorphology::MorphCircle::toMorphColorMatrixSphere() {
	double lambda = 1.0 + pow(sqrt(x * x + y * y) + std::abs(r), N_MAPPING) * (1.0 / computeLambda() - 1.0);

	return toMorphColorMatrixFromCoordinates(lambda * x, lambda * y, lambda * r); 
}

