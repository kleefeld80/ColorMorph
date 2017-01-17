#ifndef EINSTEIN_OPERATIONS_H
#define EINSTEIN_OPERATIONS_H

#include "loewner_declaration.h"
#include "morph_color_matrix.h"

#include <cmath>

#define EINSTEIN_EPSILON 1e-7
#define EINSTEIN_MOD_EPSILON 1e-5

class LoewnerMorphology::EinsteinOperations {
	
	public:
		/*
		 * Performs Einstein addition on two MorphColorMatrix objects.
		 */
		static LoewnerMorphology::MorphColorMatrix einsteinAddition(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b);
		
		/* 
		 *Performs modified commutative version of Einstein addition on two MorphColorMatrix objects.
		 */
		static LoewnerMorphology::MorphColorMatrix einsteinAdditionMod(const LoewnerMorphology::MorphColorMatrix &a, const LoewnerMorphology::MorphColorMatrix &b);
};

#endif
