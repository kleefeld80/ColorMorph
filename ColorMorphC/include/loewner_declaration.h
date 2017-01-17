#ifndef LOEWNER_DECLARATION_H
#define LOEWNER_DECLARATION_H

/*
 * Declaring namespaces that constist of classes that are used for performing morphological operations on color images based
 * on Loewner ordering and Einstein addition from the paper of B. Burgeth and A. Kleefeld.
 * 
 * The classes are described in details in appropriate header files.
 */ 
namespace LoewnerMorphology {
	class MorphColorMatrix;
	
	class Conversions;

	class MorphCircle;
	
	class MorphSmallestCircleProblemMask;

	class EinsteinOperations;

	class Morph;

	// operators implemented in morph_color_matrix.h

	MorphColorMatrix operator-(const MorphColorMatrix &a, const MorphColorMatrix &b); 
	MorphColorMatrix operator+(const MorphColorMatrix &a, const MorphColorMatrix &b);
	MorphColorMatrix operator*(const double alpha, const MorphColorMatrix &a);
	bool operator>=(const MorphColorMatrix &a, const MorphColorMatrix &b); 
	bool operator<=(const MorphColorMatrix &a, const MorphColorMatrix &b);
	bool operator>(const MorphColorMatrix &a, const MorphColorMatrix &b); 
	bool operator<(const MorphColorMatrix &a, const MorphColorMatrix &b);
	bool operator==(const MorphColorMatrix &a, const MorphColorMatrix &b); 
	bool operator!=(const MorphColorMatrix &a, const MorphColorMatrix &b);
};

#endif
