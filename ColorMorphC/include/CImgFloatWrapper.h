#ifndef CIMGFLOATWRAPPER_H
#define CIMGFLOATWRAPPER_H

#include <stdio.h>
#include <stdlib.h>

/*
 *============================================================================================================================================
 * Library that contains functions used for image manipulation.
 * It contains a wrapper class for CImg library. All methods correspond to methods with the same 
 * declaration in original CImg library.
 * 
 * version: 1.0
 * author: Filip Srnec
 *============================================================================================================================================
 */

class CImgFloatWrapper {
	private:
		float *imgData;
		int imgWidth;
		int imgHeight;
		int imgSpectrum;
		int imgSize;
		

	public:
		CImgFloatWrapper(const char *c);
		CImgFloatWrapper(const float *data, const unsigned int width, const unsigned int height = 1, const unsigned int spectrum = 1);
		float *data();
		int width() const;
		int height() const;
		int spectrum() const;
		int size() const;
		void display();
		void save(const char *fileName);
		~CImgFloatWrapper();
};

#endif
