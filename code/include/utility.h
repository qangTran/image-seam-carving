#ifndef UTILITY_H
#define UTILITY_H
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <jpeglib.h>

// Timer structure for GPU operations
struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

#define CHECK(call)                                                \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess)                                  \
        {                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
            exit(1);                                               \
        }                                                          \
    }

// Function declarations
uint8_t *readJPG(const char *filename, uint16_t *width, uint16_t *height);
void writeResizedJPG(const char *filename, uint8_t *pixels, uint16_t originalWidth, uint16_t height, uint16_t newWidth);
float computeError(uint8_t *RGBImageV1, uint8_t *RGBImageV2, int width, int height);
void printDeviceInfo();

#endif // UTILITY_H
