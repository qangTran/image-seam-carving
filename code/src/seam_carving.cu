#include "../include/seam_carving.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <jpeglib.h>

uint8_t *grayScaleImage(uint8_t *RGBImage, uint16_t width, uint16_t height)
{
    uint8_t *grayImage = (uint8_t *)malloc(sizeof(uint8_t) * width * height);

    for (uint32_t i = 0; i < width * height; ++i)
    {
        // gray = 0.299*red + 0.587*green + 0.114*blue
        grayImage[i] = 0.299 * RGBImage[3 * i] + 0.587 * RGBImage[3 * i + 1] + 0.114 * RGBImage[3 * i + 2];
    }

    return grayImage;
}

float *filterImage(uint8_t *GrayImage, uint16_t width, uint16_t height, float *filter)
{
    float *resultImage = (float *)malloc(sizeof(float) * width * height);

    for (int pixelR = 0; pixelR < height; pixelR++)
    {
        for (int pixelC = 0; pixelC < width; pixelC++)
        {
            float sum = 0;
            // for each main pixel mutiple each pixels near main pixel with repective kernal value and sum them up
            for (int filterR = 0; filterR < FILTER_WIDTH; filterR++)
            {
                for (int filterC = 0; filterC < FILTER_WIDTH; filterC++)
                {
                    // calculate respective pixel of image along with filter
                    int respectivePixelR = pixelR - (FILTER_WIDTH / 2) + filterR;
                    int respectivePixelC = pixelC - (FILTER_WIDTH / 2) + filterC;

                    // handle edge case
                    int adjustedPixelR = min(max(respectivePixelR, 0), height - 1);
                    int adjustedPixelC = min(max(respectivePixelC, 0), width - 1);

                    int flat_pixel_index = adjustedPixelR * width + adjustedPixelC;
                    int flat_kernal_index = filterR * FILTER_WIDTH + filterC;
                    sum = sum + GrayImage[flat_pixel_index] * filter[flat_kernal_index];
                }
            }
            resultImage[pixelR * width + pixelC] = sum;
        }
    }
    return resultImage;
}

float *findEnergy(uint8_t *RGBImage, uint16_t width, uint16_t height)
{
    uint8_t *GrayImage = grayScaleImage(RGBImage, width, height);

    // Sobel operator for edge detection
    float sobelX[FILTER_WIDTH * FILTER_WIDTH] = {-0.125, 0, 0.125, -0.25, 0, 0.25, -0.125, 0, 0.125};
    float sobelY[FILTER_WIDTH * FILTER_WIDTH] = {-0.125, -0.25, -0.125, 0, 0, 0, 0.125, 0.25, 0.125};

    float *energyX = filterImage(GrayImage, width, height, sobelX);
    float *energyY = filterImage(GrayImage, width, height, sobelY);

    float *energy = (float *)malloc(width * height * sizeof(float));
    for (int i = 0; i < width * height; i++)
        energy[i] = sqrt(pow((float)energyX[i], 2) + pow((float)energyY[i], 2));

    free(energyX);
    free(energyY);
    free(GrayImage);

    return energy;
}

void findEnergyMap(float *energy, float *energyMap, int *nextElements, int width, int height, int numberOfSeamRemoved)
{
    // initialize first row of energy map
    for (int col = 0; col < width - numberOfSeamRemoved; col++)
        energyMap[(height - 1) * width + col] = energy[(height - 1) * width + col];

    // set up remain rows
    for (int row = height - 2; row >= 0; row--)
    {
        for (int col = 0; col < width - numberOfSeamRemoved; col++)
        {
            int left = max(0, col - 1);
            int right = min(col + 1, width - numberOfSeamRemoved - 1);

            int direction = (col == 0) ? 0 : -1; // -1: left, 0: middle, 1: right, default value start from left if not in first column
            int localEnergy = energyMap[(row + 1) * width + left];

            // find min of 3 neighbor elements above
            for (int k = 0; k < right - left; k++)
            {
                if (energyMap[(row + 1) * width + left + k + 1] < localEnergy)
                {
                    localEnergy = energyMap[(row + 1) * width + left + k + 1];
                    direction = k;
                }
            }
            energyMap[row * width + col] = localEnergy + energy[row * width + col];
            nextElements[row * width + col] = direction;
        }
    }
}

void findSeam(float *energyMap, int *nextElements, int *seam, int width, int height, int numberOfSeamRemoved)
{
    // find seam at minimum energy
    float minEnergy = energyMap[0];
    int minIndex = 0;

    for (int col = 1; col < width - numberOfSeamRemoved; col++)
    {
        if (energyMap[col] < minEnergy)
        {
            minEnergy = energyMap[col];
            minIndex = col;
        }
    }

    seam[0] = minIndex;

    for (int row = 1; row < height; row++)
    {
        seam[row] = seam[row - 1] + nextElements[(row - 1) * width + seam[row - 1]];
    }
}

void removeSeam(int *seam, uint8_t *RGBImage, float *energy, uint16_t width, uint16_t height, int numberOfSeamRemoved)
{
    for (int row = 0; row < height; row++)
    {
        // shift all element to left 1 position from seam to end of row
        for (int col = seam[row]; col < width - numberOfSeamRemoved - 1; col++)
        {
            energy[row * width + col] = energy[row * width + col + 1];
            for (int channel = 0; channel < 3; channel++)
            {
                RGBImage[(row * width + col) * 3 + channel] = RGBImage[(row * width + col) * 3 + channel + 3];
            }
        }
    }
}

uint8_t *seamCarvingOnHost(uint8_t *RGBImage, uint16_t width, uint16_t height, uint16_t newWidth)
{
    // Setup resized image
    size_t tmpSize = sizeof(uint8_t) * width * height * 3;
    uint8_t *resizedRGBImage = (uint8_t *)malloc(tmpSize);
    memcpy(resizedRGBImage, RGBImage, tmpSize);

    // Seam carving in vertical direction
    float *energy = findEnergy(RGBImage, width, height);

    float *energyMap = (float *)malloc(width * height * sizeof(float));
    int *nextElements = (int *)malloc(width * height * sizeof(int));
    int *seam = (int *)malloc(height * sizeof(int));

    for (int i = 0; i < width - newWidth; i++)
    {
        findEnergyMap(energy, energyMap, nextElements, width, height, i);

        findSeam(energyMap, nextElements, seam, width, height, i);

        removeSeam(seam, resizedRGBImage, energy, width, height, i);
    }

    // Free allocated data
    free(energy);
    free(energyMap);
    free(nextElements);
    free(seam);

    return resizedRGBImage;
}

__global__ void grayScaleImageDeviceV1(uint8_t *RGBImage, uint8_t *grayImage, uint16_t width, uint16_t height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        int i = row * width + col;
        grayImage[i] = 0.299 * RGBImage[3 * i] + 0.587 * RGBImage[3 * i + 1] + 0.114 * RGBImage[3 * i + 2];
    }
}

__global__ void findEnergyDeviceV1(uint8_t *GrayImage, uint16_t width, uint16_t height, float *energy)
{
    // Set up SMEM
    int s_GrayImage_height = blockDim.y + FILTER_WIDTH - 1;
    int s_GrayImage_width = blockDim.x + FILTER_WIDTH - 1;
    extern __shared__ uint8_t s_GrayImage[];

    // Calculate the top-left corner position of the block in the image
    int topLeftRow = blockIdx.y * blockDim.y - FILTER_WIDTH / 2;
    int topLeftCol = blockIdx.x * blockDim.x - FILTER_WIDTH / 2;

    // Load data into shared memory
    for (int s_row = threadIdx.y; s_row < s_GrayImage_height; s_row += blockDim.y)
    {
        for (int s_col = threadIdx.x; s_col < s_GrayImage_width; s_col += blockDim.x)
        {
            int globalRow = min(max(topLeftRow + s_row, 0), height - 1);
            int globalCol = min(max(topLeftCol + s_col, 0), width - 1);

            s_GrayImage[s_row * s_GrayImage_width + s_col] = GrayImage[globalRow * width + globalCol];
        }
    }

    __syncthreads();

    // Calculate global position of the thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Apply Sobel filters
    if (col < width && row < height)
    {
        float sumX = 0;
        float sumY = 0;

        // draw out to understand why s_inPixels[filterR + blockIdx.y][filterC + blockIdx.x]
        for (int filterR = 0; filterR < FILTER_WIDTH; filterR++)
        {
            for (int filterC = 0; filterC < FILTER_WIDTH; filterC++)
            {
                int localRow = threadIdx.y + filterR;
                int localCol = threadIdx.x + filterC;

                if (localRow < s_GrayImage_height && localCol < s_GrayImage_width)
                {
                    uint8_t pixelValue = s_GrayImage[localRow * s_GrayImage_width + localCol];

                    // Apply Sobel X and Y filters
                    sumX += pixelValue * dc_sobelX[filterR * FILTER_WIDTH + filterC];
                    sumY += pixelValue * dc_sobelY[filterR * FILTER_WIDTH + filterC];
                }
            }
        }

        // Store the combined energy value in the output array
        energy[row * width + col] = sqrt(sumX * sumX + sumY * sumY);
    }
}

__global__ void findEnergyMapDeviceV1(float *energy, float *energyMap, int *nextElements, int width, int height, int numberOfSeamRemoved)
{

    int ti = threadIdx.x;
    // copy the last row of energy map
    for (int stride = 0; stride <= (width - numberOfSeamRemoved) / blockDim.x; stride++)
    {
        int col = ti + (stride * blockDim.x);
        if (col < width - numberOfSeamRemoved)
            energyMap[ti + (stride * blockDim.x)] = energy[ti + (stride * blockDim.x)];
    }
    __syncthreads();

    // calculate remaining rows
    for (int rows = height - 2; rows >= 0; rows--)
    {
        for (int stride = 0; stride <= (width - numberOfSeamRemoved) / blockDim.x; stride++)
        {
            int col = ti + (stride * blockDim.x);
            if (col < width - numberOfSeamRemoved)
            {
                int left = max(0, col - 1);
                int right = min(col + 1, width - numberOfSeamRemoved - 1);

                int direction = (col == 0) ? 0 : -1; // -1: left, 0: middle, 1: right, default value start from left if not in first column
                int localEnergy = energyMap[(rows + 1) * width + left];

                for (int k = 0; k < right - left; k++)
                {
                    if (energyMap[(rows + 1) * width + left + k + 1] < localEnergy)
                    {
                        localEnergy = energyMap[(rows + 1) * width + left + k + 1];
                        direction = k;
                    }
                }
                energyMap[rows * width + col] = localEnergy + energy[rows * width + col];
                nextElements[rows * width + col] = direction;
            }
        }
        __syncthreads();
    }
}

__global__ void findSeamDeviceV1(float *energyMap, int *nextElements, int *seam, int width, int height, int numberOfSeamRemoved)
{
    // Find the index of the minimum energy in the first row
    int minIndex = 0;
    float minEnergy = energyMap[0];
    for (int col = 1; col < width - numberOfSeamRemoved; col++)
    {
        if (energyMap[col] < minEnergy)
        {
            minEnergy = energyMap[col];
            minIndex = col;
        }
    }
    seam[0] = minIndex;

    // using nextElement to find the rest of seam
    for (int row = 1; row < height; row++)
    {
        seam[row] = seam[row - 1] + nextElements[(row - 1) * width + seam[row - 1]];
    }
}

__global__ void removeSeamDeviceV1(float *energy, float *tpmEnergy, uint8_t *RGBImage, uint8_t *tpmRGBImage, int *seam, int width, int height, int numberOfSeamRemoved)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // if  thread (31,31,0) in block (47,31,0), print value of i

    // shift all element to left 1 position from seam to end of row
    if (row < height && col < width - 1 - numberOfSeamRemoved && col >= seam[row])
    {
        int i = row * width + col;
        for (int channel = 0; channel < 3; channel++)
        {
            RGBImage[i * 3 + channel] = tpmRGBImage[i * 3 + channel + 3];
        }
        energy[i] = tpmEnergy[i + 1];
    }
}

uint8_t *seamCarvingOnDevideV1(uint8_t *RGBImage, uint16_t width, uint16_t height, uint16_t newWidth, int blockSize)
{
    // set up nessessary variable
    size_t nFloats = width * height * sizeof(float);
    size_t n8Bits = width * height * sizeof(uint8_t);
    size_t nInts = width * height * sizeof(int);

    dim3 commonBlockSize(blockSize, blockSize); // blocksize x blocksize
    dim3 commonGridSize((width - 1) / commonBlockSize.x + 1, (height - 1) / commonBlockSize.y + 1);

    uint8_t *d_resizedRGBImage;
    CHECK(cudaMalloc(&d_resizedRGBImage, n8Bits * 3));
    CHECK(cudaMemcpy(d_resizedRGBImage, RGBImage, n8Bits * 3, cudaMemcpyHostToDevice));

    // convert RGB image to gray image
    uint8_t *d_resizedGrayImage;
    CHECK(cudaMalloc(&d_resizedGrayImage, n8Bits));
    grayScaleImageDeviceV1<<<commonGridSize, commonBlockSize>>>(d_resizedRGBImage, d_resizedGrayImage, width, height);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // find energy
    float *d_energy;
    CHECK(cudaMalloc(&d_energy, nFloats));
    size_t energySMEMSize = (commonBlockSize.x + FILTER_WIDTH - 1) * (commonBlockSize.y + FILTER_WIDTH - 1) * sizeof(uint8_t);
    findEnergyDeviceV1<<<commonGridSize, commonBlockSize, energySMEMSize>>>(d_resizedGrayImage, width, height, d_energy);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // declare neccessary variable for looping
    float *d_energyMap;
    CHECK(cudaMalloc(&d_energyMap, nFloats));
    int *d_nextElements;
    CHECK(cudaMalloc(&d_nextElements, nInts));
    int *d_seam;
    CHECK(cudaMalloc(&d_seam, height * sizeof(int)));
    uint8_t *d_tpmRGBImage;
    CHECK(cudaMalloc(&d_tpmRGBImage, n8Bits * 3));
    float *d_tpmEnergy;
    CHECK(cudaMalloc(&d_tpmEnergy, nFloats));

    // Seam carving in vertical direction
    for (int i = 0; i < width - newWidth; i++)
    {
        // calculate energy map and find seam
        findEnergyMapDeviceV1<<<1, 1024>>>(d_energy, d_energyMap, d_nextElements, width, height, i);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        // Find seam
        findSeamDeviceV1<<<1, 1>>>(d_energyMap, d_nextElements, d_seam, width, height, i);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        // remove seam on RGBImage, energy
        CHECK(cudaMemcpy(d_tpmRGBImage, d_resizedRGBImage, n8Bits * 3, cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(d_tpmEnergy, d_energy, nFloats, cudaMemcpyDeviceToDevice));
        removeSeamDeviceV1<<<commonGridSize, commonBlockSize>>>(d_energy, d_tpmEnergy, d_resizedRGBImage, d_tpmRGBImage, d_seam, width, height, i);
    }

    // make image in proper size
    uint8_t *resizedRGBImage = (uint8_t *)malloc(n8Bits * 3);
    CHECK(cudaMemcpy(resizedRGBImage, d_resizedRGBImage, n8Bits * 3, cudaMemcpyDeviceToHost));

    // free allocated data on device
    CHECK(cudaFree(d_energy));
    CHECK(cudaFree(d_energyMap));
    CHECK(cudaFree(d_nextElements));
    CHECK(cudaFree(d_seam));
    CHECK(cudaFree(d_resizedGrayImage));
    CHECK(cudaFree(d_resizedRGBImage));
    CHECK(cudaFree(d_tpmRGBImage));
    CHECK(cudaFree(d_tpmEnergy));

    return resizedRGBImage;
}

__global__ void findEnergyDeviceV2(uint8_t *RGBImage, float *energy, uint16_t width, uint16_t height)
{
    // Set up SMEM
    int s_GrayImage_height = blockDim.y + FILTER_WIDTH - 1;
    int s_GrayImage_width = blockDim.x + FILTER_WIDTH - 1;
    extern __shared__ uint8_t s_GrayImage[];

    // Calculate the top-left corner position of the block in the image
    int topLeftRow = blockIdx.y * blockDim.y - FILTER_WIDTH / 2;
    int topLeftCol = blockIdx.x * blockDim.x - FILTER_WIDTH / 2;

    // Convert data to gray scale and Load them into shared memory
    for (int s_row = threadIdx.y; s_row < s_GrayImage_height; s_row += blockDim.y)
    {
        for (int s_col = threadIdx.x; s_col < s_GrayImage_width; s_col += blockDim.x)
        {
            int globalRow = min(max(topLeftRow + s_row, 0), height - 1);
            int globalCol = min(max(topLeftCol + s_col, 0), width - 1);
            int tmp = (globalRow * width + globalCol) *3;

            s_GrayImage[s_row * s_GrayImage_width + s_col] = (RGBImage[tmp] * 0.299 + RGBImage[tmp + 1] * 0.587 + RGBImage[tmp + 2] * 0.114);
        }
    }

    __syncthreads();

    // Calculate global position of the thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Apply Sobel filters
    if (col < width && row < height)
    {
        float sumX = 0;
        float sumY = 0;

        // draw out to understand why s_inPixels[filterR + blockIdx.y][filterC + blockIdx.x]
        for (int filterR = 0; filterR < FILTER_WIDTH; filterR++)
        {
            for (int filterC = 0; filterC < FILTER_WIDTH; filterC++)
            {
                int localRow = threadIdx.y + filterR;
                int localCol = threadIdx.x + filterC;

                if (localRow < s_GrayImage_height && localCol < s_GrayImage_width)
                {
                    uint8_t pixelValue = s_GrayImage[localRow * s_GrayImage_width + localCol];

                    // Apply Sobel X and Y filters
                    sumX += pixelValue * dc_sobelX[filterR * FILTER_WIDTH + filterC];
                    sumY += pixelValue * dc_sobelY[filterR * FILTER_WIDTH + filterC];
                }
            }
        }

        // Store the combined energy value in the output array
        energy[row * width + col] = sqrt(sumX * sumX + sumY * sumY);
    }
}

uint8_t *seamCarvingOnDevideV2(uint8_t *RGBImage, uint16_t width, uint16_t height, uint16_t newWidth, int blockSize)
{
    // set up nessessary variable
    size_t nFloats = width * height * sizeof(float);
    size_t n8Bits = width * height * sizeof(uint8_t);
    size_t nInts = width * height * sizeof(int);

    dim3 commonBlockSize(blockSize, blockSize); // blocksize x blocksize
    dim3 commonGridSize((width - 1) / commonBlockSize.x + 1, (height - 1) / commonBlockSize.y + 1);

    uint8_t *d_resizedRGBImage;
    CHECK(cudaMalloc(&d_resizedRGBImage, n8Bits * 3));
    CHECK(cudaMemcpy(d_resizedRGBImage, RGBImage, n8Bits * 3, cudaMemcpyHostToDevice));

    // find energy
    float *d_energy;
    CHECK(cudaMalloc(&d_energy, nFloats));
    size_t energySMEMSize = (commonBlockSize.x + FILTER_WIDTH - 1) * (commonBlockSize.y + FILTER_WIDTH - 1) * sizeof(uint8_t);
    findEnergyDeviceV2<<<commonGridSize, commonBlockSize, energySMEMSize>>>(d_resizedRGBImage, d_energy, width, height);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // declare neccessary variable for looping
    float *d_energyMap;
    CHECK(cudaMalloc(&d_energyMap, nFloats));
    int *d_nextElements;
    CHECK(cudaMalloc(&d_nextElements, nInts));
    int *d_seam;
    CHECK(cudaMalloc(&d_seam, height * sizeof(int)));
    uint8_t *d_tpmRGBImage;
    CHECK(cudaMalloc(&d_tpmRGBImage, n8Bits * 3));
    float *d_tpmEnergy;
    CHECK(cudaMalloc(&d_tpmEnergy, nFloats));

    // Seam carving in vertical direction
    for (int i = 0; i < width - newWidth; i++)
    {
        // calculate energy map and find seam
        findEnergyMapDeviceV1<<<1, 1024>>>(d_energy, d_energyMap, d_nextElements, width, height, i);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        // Find seam
        findSeamDeviceV1<<<1, 1>>>(d_energyMap, d_nextElements, d_seam, width, height, i);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        // remove seam on RGBImage, energy
        CHECK(cudaMemcpy(d_tpmRGBImage, d_resizedRGBImage, n8Bits * 3, cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpy(d_tpmEnergy, d_energy, nFloats, cudaMemcpyDeviceToDevice));
        removeSeamDeviceV1<<<commonGridSize, commonBlockSize>>>(d_energy, d_tpmEnergy, d_resizedRGBImage, d_tpmRGBImage, d_seam, width, height, i);
    }

    // make image in proper size
    uint8_t *resizedRGBImage = (uint8_t *)malloc(n8Bits * 3);
    CHECK(cudaMemcpy(resizedRGBImage, d_resizedRGBImage, n8Bits * 3, cudaMemcpyDeviceToHost));

    // free allocated data on device
    CHECK(cudaFree(d_energy));
    CHECK(cudaFree(d_energyMap));
    CHECK(cudaFree(d_nextElements));
    CHECK(cudaFree(d_seam));
    CHECK(cudaFree(d_resizedRGBImage));
    CHECK(cudaFree(d_tpmRGBImage));
    CHECK(cudaFree(d_tpmEnergy));

    return resizedRGBImage;
}

uint8_t *seamCarving(uint8_t *RGBImage, uint16_t width, uint16_t height, uint16_t newWidth, int ver, int blockSize)
{

    GpuTimer timer;
    timer.Start();

    uint8_t *resizedImage;
    if (ver == 0)
    {
        printf("\nSeam Carving by host\n");
        resizedImage = seamCarvingOnHost(RGBImage, width, height, newWidth);
    }
    if (ver == 1)
    {
        printf("\nSeam Carving by device V1\n");
        resizedImage = seamCarvingOnDevideV1(RGBImage, width, height, newWidth, blockSize);
    }
    if (ver == 2)
    {
        printf("\nSeam Carving by device V2\n");
        resizedImage = seamCarvingOnDevideV2(RGBImage, width, height, newWidth, blockSize);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());

    return resizedImage;
}
