#ifndef SEAM_CARVING_H
#define SEAM_CARVING_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <jpeglib.h>

#include "utility.h"



// solbel filter
#define FILTER_WIDTH 3
__constant__ float dc_sobelX[FILTER_WIDTH * FILTER_WIDTH] = {-0.125, 0, 0.125, -0.25, 0, 0.25, -0.125, 0, 0.125};
__constant__ float dc_sobelY[FILTER_WIDTH * FILTER_WIDTH] = {-0.125, -0.25, -0.125, 0, 0, 0, 0.125, 0.25, 0.125};

// Function declarations for the seam carving process on the host
uint8_t *grayScaleImage(uint8_t *RGBImage, uint16_t width, uint16_t height);
float *filterImage(uint8_t *GrayImage, uint16_t width, uint16_t height, float *filter);
float *findEnergy(uint8_t *RGBImage, uint16_t width, uint16_t height);
void findEnergyMap(float *energy, float *energyMap, int *nextElements, int width, int height, int numberOfSeamRemoved);
void findSeam(float *energyMap, int *nextElements, int *seam, int width, int height, int numberOfSeamRemoved);
void removeSeam(int *seam, uint8_t *RGBImage, float *energy, uint16_t width, uint16_t height, int numberOfSeamRemoved);
uint8_t *seamCarvingOnHost(uint8_t *RGBImage, uint16_t width, uint16_t height, uint16_t newWidth);

// Function declarations for the seam carving process on the device
__global__ void grayScaleImageDeviceV1(uint8_t *RGBImage, uint8_t *grayImage, uint16_t width, uint16_t height);
__global__ void findEnergyDeviceV1(uint8_t *GrayImage, uint16_t width, uint16_t height, float *energy);
__global__ void findEnergyMapDeviceV1(float *energy, float *energyMap, int *nextElements, int width, int height, int numberOfSeamRemoved);
__global__ void findSeamDeviceV1(float *energyMap, int *nextElements, int *seam, int width, int height, int numberOfSeamRemoved);
__global__ void removeSeamDeviceV1(float *energy, float *tpmEnergy, uint8_t *RGBImage, uint8_t *tpmRGBImage, int *seam, int width, int height, int numberOfSeamRemoved);
uint8_t *seamCarvingOnDeviceV1(uint8_t *RGBImage, uint16_t width, uint16_t height, uint16_t newWidth, int blockSize);

// Function declarations for call seam carving and time measurement
uint8_t *seamCarving(uint8_t *RGBImage, uint16_t width, uint16_t height, uint16_t newWidth, int ver, int blockSize);

#endif // SEAM_CARVING_H
