#include "../include/CImgFloatWrapper.h"
#include "../lib/Cimg/CImg-1.7.5_pre080216/CImg.h"
#include "omp.h"

void copy(const float *src, float *dest, int size) {
	#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		dest[i] = src[i];
	}
}

CImgFloatWrapper::CImgFloatWrapper(const char *c) {
	cimg_library::CImg<float> img(c);

	imgWidth = img.width();
	imgHeight = img.height();
	imgSpectrum = img.spectrum();
	imgSize = img.size();
	
	imgData = (float *)malloc(imgSize * sizeof(float));
	copy(img.data(), imgData, imgSize);
}

CImgFloatWrapper::~CImgFloatWrapper() {
	free(imgData);
}

CImgFloatWrapper::CImgFloatWrapper(const float *data, const unsigned int width, const unsigned int height, const unsigned int spectrum) {
	imgWidth = width;
	imgHeight = height;
	imgSpectrum = spectrum;
	imgSize = width * height * spectrum;
	
	imgData = (float *)malloc(imgSize * sizeof(float));
	copy(data, imgData, imgSize); 
}

float *CImgFloatWrapper::data() {
	return imgData;
}

int CImgFloatWrapper::width() const {
	return imgWidth;	
}

int CImgFloatWrapper::height() const {
	return imgHeight;
}

int CImgFloatWrapper::spectrum() const {
	return imgSpectrum;
}

int CImgFloatWrapper::size() const {
	return imgSize;
}

void CImgFloatWrapper::display() {
	cimg_library::CImg<float> img(imgData, imgWidth, imgHeight, 1, imgSpectrum, false);
	
	img.display();
}

void CImgFloatWrapper::save(const char *fileName) {
	cimg_library::CImg<float> img(imgData, imgWidth, imgHeight, 1, imgSpectrum, false);

	img.save(fileName);
}
