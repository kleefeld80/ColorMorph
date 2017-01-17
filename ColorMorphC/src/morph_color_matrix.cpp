#include "../include/morph_color_matrix.h"

LoewnerMorphology::MorphColorMatrix  LoewnerMorphology::MorphColorMatrix::min() {	
	return MorphColorMatrix(-KAPPA, 0, -KAPPA);
}	

LoewnerMorphology::MorphColorMatrix LoewnerMorphology::MorphColorMatrix::max() {
	return MorphColorMatrix(KAPPA, 0, KAPPA);
}

void LoewnerMorphology::MorphColorMatrix::printMatrixA(const LoewnerMorphology::MorphColorMatrix *matrix, int width, int height, int lda) {
        const LoewnerMorphology::MorphColorMatrix *current = matrix;

        for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                        printf("%10.6f", current[j].a);;
                }
                printf("\n");

                current += lda;
        }
}

void LoewnerMorphology::MorphColorMatrix::printMatrixB(const LoewnerMorphology::MorphColorMatrix *matrix, int width, int height, int lda) {
        const LoewnerMorphology::MorphColorMatrix *current = matrix;

        for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                        printf("%10.6f", current[j].b);;
                }
                printf("\n");

                current += lda;
        }
}

void LoewnerMorphology::MorphColorMatrix::printMatrixC(const LoewnerMorphology::MorphColorMatrix *matrix, int width, int height, int lda) {
        const LoewnerMorphology::MorphColorMatrix *current = matrix;

        for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                        printf("%10.6f", current[j].c);;
                }
                printf("\n");

                current += lda;
        }
}

void LoewnerMorphology::MorphColorMatrix::printMorphColorMatrix() const {
	printf("(%.15f %.15f %.15f)\n", a, b, c);
}


LoewnerMorphology::MorphColorMatrix LoewnerMorphology::operator-(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b) {
	return LoewnerMorphology::MorphColorMatrix(a.a - b.a, a.b - b.b, a.c - b.c);
}

LoewnerMorphology::MorphColorMatrix LoewnerMorphology::operator+(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b) {
        return LoewnerMorphology::MorphColorMatrix(a.a + b.a, a.b + b.b, a.c + b.c);
}

LoewnerMorphology::MorphColorMatrix LoewnerMorphology::operator*(const double alpha, const LoewnerMorphology::MorphColorMatrix &a) {
	return LoewnerMorphology::MorphColorMatrix(alpha * a.a, alpha * a.b, alpha * a.c);
}

bool LoewnerMorphology::operator>=(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b) {
	LoewnerMorphology::MorphColorMatrix temp = a - b;

	return (temp.a >= 0 && (temp.a * temp.c - temp.b * temp.b) >= 0);	
}

bool LoewnerMorphology::operator<=(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b) {
        return (b >= a);
}

bool LoewnerMorphology::operator>(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b) {
        LoewnerMorphology::MorphColorMatrix temp = a - b;

        return (temp.a > 0 && (temp.a * temp.c - temp.b * temp.b) >= 0);
}

bool LoewnerMorphology::operator<(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b) {
        return (b > a);
}

bool LoewnerMorphology::operator==(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b) {
	LoewnerMorphology::MorphColorMatrix c = a - b;

	return (std::abs(c.a) < FLT_EPSILON) && (std::abs(c.b) < FLT_EPSILON) && (std::abs(c.c) < FLT_EPSILON);
}

bool LoewnerMorphology::operator!=(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b) {
        return !(a == b);
}

double LoewnerMorphology::MorphColorMatrix::trace() const {
	return a + c;
}

double LoewnerMorphology::MorphColorMatrix::traceProduct(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b) {
	return a.a * b.a + 2 * a.b * b.b + a.c * b.c; 
} 

double LoewnerMorphology::MorphColorMatrix::norm() const {
	return std::sqrt(traceProduct(*this, *this));
}

LoewnerMorphology::MorphColorMatrix& LoewnerMorphology::MorphColorMatrix::shift(double alpha) {
	a += alpha;
	c += alpha;

	return *this;
}

LoewnerMorphology::MorphColorMatrix& LoewnerMorphology::MorphColorMatrix::negate() {
	a = -a;
	b = -b;
	c = -c;

	return *this;
}

LoewnerMorphology::MorphColorMatrix LoewnerMorphology::MorphColorMatrix::negation() {
	return MorphColorMatrix(-a, -b, -c);
}

