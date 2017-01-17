#ifndef CONVERSIONS_H
#define CONVERSIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "omp.h"

#include "loewner_declaration.h"
#include "morph_color_matrix.h"

/*
 *============================================================================================================================================
 * Class that contains methods for performing conversions between different image formats.
 * Algorithms for conversions are following the paper An approach to color-morphology based on Einstein addition and Loewner order by
 * Bernhard Burgeth and Andreas Kleefeld.
 *
 * Current version is implemented to perform conversions in parallel using OpenMP, with the number of threads predefined by user.
 *
 * version: 1.0
 * author: Filip Srnec
 *============================================================================================================================================
 */

#ifndef PI
#define PI 3.141592653589793
#endif

#ifndef KAPPA
#define KAPPA 0.707106781186547
#endif

#ifndef RGB_MAX
#define RGB_MAX 255.0
#endif

#ifndef INTENSITY_ALPHA
#define INTENSITY_ALPHA 10
#endif

#ifndef INTENSITY_FACTOR
#define INTENSITY_FACTOR ((T) RGB_MAX / INTENSITY_ALPHA)
#endif

#ifndef HUE_TRESHOLD
#define HUE_TRESHOLD 0.5
#endif

class LoewnerMorphology::Conversions {

	private:
		// Helper method for finding a minimum among 3 values. T must have operator <.
		template<typename T> 
		static T min3(T a, T b, T c);
		// Helper method for finding a maximum among 3 values. T must have operator >.
		template<typename T> 
		static T max3(T a, T b, T c);
		// Helper method that converts hue values to rgb values
		template<typename T>
		static T hueToRgb(T p, T q, T t);
		
	public:
		/*
		 * Performs a conversion from RGB-value image to M-HCL-value image. RGB values should be stored
		 * on memory locations r, g and b respectively. Result will be stored on locations h, c and l.
		 * All memory should be previously allocated. Argument size is the size of the initial image (width * height).

		 */
		template<typename T>
		static void rgb2mhcl(const T *r, const T *g, const T *b, T *h, T *c, T *l, int size);

		/*
 		 * Performs a conversion from M-HCL-value image to RGB-value image. H, C and L values should be stored on memory locations 
		 * r, g and b respectively. Result will be stored on locations r, g and b. All memory should be previously allocated. 
		 * Argument size is the size of the initial image (width * height).
		 */
		template<typename T>
		static void mhcl2rgb(const T *h, const T *c, const T *l, T *r, T *g, T *b, int size);
	
		/*
		 * Converts the image vector containing M-HCL values to the vector containing
		 * MorphColorMatrix objects. Memory for the new vector needs to be allocated and passed
		 * as a vector argument. It should be array of MorphColorMatrix values which size is equal
		 * as the size of the image. Pointers h, c and l are the pointers to the values
		 * of the hue, chroma and luminance.
		 */
		template<typename T>  
		static void mhcl2matrix(const T *h, const T *c, const T *l, LoewnerMorphology::MorphColorMatrix *vector, int size);

		/*
		 * Converts the vector containing MorphColorMatrix objects to M-HCL values. Memory for the new vectors needs 
		 * to be allocated and passed as a h,c and l  arguments. Size of each destination pointer must be equal
		 * to the size of the image. Pointers h, c and l are the pointers to the values of the hue, chroma and 
		 * lightness for the given image.
		 */ 
		template<typename T> 
		static void matrix2mhcl(const LoewnerMorphology::MorphColorMatrix *vector, T *h, T *c, T *l, int size);
		
		/*
		 * Converts image array of values of type T to the image array of double values. It is assumed that conversion is legal. Memory for both arrays must be allocated
		 * before entering the method.
		 */
		template<typename T>
		static void type2double(const T *imgType, double *imgDouble, int size);

		/*
		 * Converts image array of double values to the image array of values of type T. It is assumed that conversion is legal. Memory for both arrays should be allocated
		 * before entering the method.
		 */
		template<typename T>
		static void double2type(const double *imgDouble, T *imgType, int size);
};

// IMPLEMENTATION

template<typename T> 
T LoewnerMorphology::Conversions::min3(T a, T b, T c) {
	if (a < b) {
		if (a < c) {
			return a;
		} else {
			return (b < c) ? b : c;
		}
	} else {
		if (b < c) {
			return b;
		} else {
			return (a < c) ? a : c;
		}
	}	
}

template<typename T> 
T LoewnerMorphology::Conversions::max3(T a, T b, T c) {
	if (a > b) {
		if (a > c) {
			return a;
		} else {
			return (b > c) ? b : c;
		}
	} else {
		if (b > c) {
			return b;
		} else {
			return (a > c) ? a : c;
		}
	}	
}

template<typename T>
void LoewnerMorphology::Conversions::rgb2mhcl(const T *r, const T *g, const T *b, T *h, T *c, T *l, int size) {
	#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		T r1 = r[i] / RGB_MAX;
		T g1 = g[i] / RGB_MAX;
		T b1 = b[i] / RGB_MAX;

		T cMax = max3(r1, g1, b1);
		T cMin = min3(r1, g1, b1);

		T delta = cMax - cMin;

		if (delta == 0) {
			h[i] = 0;
		} else if (cMax == r1) {
			h[i] = ((g1 - b1) / delta) + (g1 < b1 ? 6.0 : 0.0);
		} else if (cMax == g1) {
			h[i] = (2 + ((b1 - r1) / delta));
		} else {
			h[i] = (4 + ((r1 - g1) / delta));
		}
		
		if (h[i] < 0) {
			h[i] += 1;
		}

		h[i] /= 6.0;	
		c[i] = delta;
		l[i] = cMin + cMax - 1;
	} 
}

template<typename T>
T LoewnerMorphology::Conversions::hueToRgb(T p, T q, T t) {
	if (t < 0)
		t += 1;
	if (t > 1)
		t -= 1;
	if (t < (T) 1 / 6)
		return p + (q - p) * 6 * t;
	if (t < (T) 1 / 2)
		return q;
	if (t < (T) 2 / 3)
		return p + (q - p) * ((T) 2 / 3 - t) * 6;
	return p;
}

template<typename T>
void LoewnerMorphology::Conversions::mhcl2rgb(const T *h, const T *c, const T *l, T *r, T *g, T *b, int size) {
	#pragma omp parallel for
        for (int i = 0; i < size; i++) {
		T delta = c[i];

		T l_val = (l[i] + 1) / 2;
		T s_val = (delta < 1e-4) ? 0 : delta / (1 - abs(l[i]));
		
		T q = (l_val < 0.5) ? (l_val * (1 + s_val)) : (l_val + s_val - (l_val * s_val));
		T p = 2 * l_val - q;

		r[i] = round(hueToRgb(p, q, h[i] + (T) 1 / 3) * RGB_MAX);		
		g[i] = round(hueToRgb(p, q, h[i]) * RGB_MAX);		
		b[i] = round(hueToRgb(p, q, h[i] - (T) 1 / 3) * RGB_MAX);			
	}
}

template<typename T>  
void LoewnerMorphology::Conversions::mhcl2matrix(const T *h, const T *c, const T *l, LoewnerMorphology::MorphColorMatrix *vector, int size) {
	#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		T hv = h[i];
		T cv = c[i];
		T z = l[i];

		T value = 2 * PI * hv;
		T x = cv * cos(value);
		T y = cv * sin(value);  
		
		LoewnerMorphology::MorphColorMatrix temp;
		temp.a = KAPPA * (z - y);
		temp.b = KAPPA * x;
		temp.c = KAPPA * (z + y);

		vector[i] = temp;
	}
}

template<typename T> 
void LoewnerMorphology::Conversions::matrix2mhcl(const LoewnerMorphology::MorphColorMatrix *vector, T *h, T *c, T *l, int size) {
	#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		LoewnerMorphology::MorphColorMatrix temp = vector[i];
		T x = 2 * KAPPA * temp.b;
		T y = KAPPA * (temp.c - temp.a);
		T z = KAPPA * (temp.c + temp.a);	
	
		T at = atan2(y, x);
	
		if (at < 0) {
			at = at + 2 * PI;
		}
	
		h[i] = at / (2 * PI);
		c[i] = sqrt(x * x + y * y);
		l[i] = z;
	}
}

template<typename T>
void LoewnerMorphology::Conversions::type2double(const T *imgOriginal, double *imgDouble, int size) {
	#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		imgDouble[i] = (double)imgOriginal[i];
	}
} 

template<typename T>
void LoewnerMorphology::Conversions::double2type(const double *imgDouble, T *imgType, int size) {
	#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		imgType[i] = (T)imgDouble[i];
	}
}

#endif
