#include "../include/utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <jpeglib.h>

uint8_t *readJPG(const char *filename, uint16_t *width, uint16_t *height)
{
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    FILE *file = fopen(filename, "rb");
    if (file == NULL)
    {
        fprintf(stderr, "Error opening file %s\n", filename);
        return NULL;
    }

    // Check for JPEG magic number
    unsigned char magicBuffer[3];
    if (fread(magicBuffer, 1, 3, file) != 3)
    {
        fprintf(stderr, "Error reading file %s\n", filename);
        fclose(file);
        return NULL;
    }

    // Reset file pointer to the beginning
    rewind(file);

    // Verify JPEG magic number
    if (magicBuffer[0] != 0xFF || magicBuffer[1] != 0xD8 || magicBuffer[2] != 0xFF)
    {
        fprintf(stderr, "File %s is not a valid JPEG file\n", filename);
        fclose(file);
        return NULL;
    }

    // Initialize the JPEG decompression object
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    // Specify the source file
    jpeg_stdio_src(&cinfo, file);

    // Read the file parameters
    jpeg_read_header(&cinfo, TRUE);

    // Start decompression
    jpeg_start_decompress(&cinfo);

    // Set width and height
    *width = cinfo.output_width;
    *height = cinfo.output_height;

    // Allocate memory for pixel data
    uint8_t *pixels = (uint8_t *)malloc(sizeof(uint8_t) * (*width) * (*height) * 3);

    if (pixels == NULL)
    {
        fprintf(stderr, "Memory allocation error\n");
        fclose(file);
        jpeg_destroy_decompress(&cinfo);
        return NULL;
    }

    // Read scanlines and fill the pixel data
    JSAMPARRAY buffer = (JSAMPARRAY)malloc(sizeof(JSAMPROW) * cinfo.output_height);

    for (uint32_t i = 0; i < cinfo.output_height; ++i)
    {
        buffer[i] = (JSAMPROW)malloc(sizeof(JSAMPLE) * cinfo.output_width * cinfo.output_components);
    }

    while (cinfo.output_scanline < cinfo.output_height)
    {
        jpeg_read_scanlines(&cinfo, buffer + cinfo.output_scanline, cinfo.output_height - cinfo.output_scanline);
    }

    // Convert libjpeg RGB to flat RGB array
    for (uint32_t i = 0; i < *height; ++i)
    {
        for (uint32_t j = 0; j < *width; ++j)
        {
            pixels[(i * (*width) + j) * 3] = buffer[i][j * cinfo.output_components];         // Red
            pixels[(i * (*width) + j) * 3 + 1] = buffer[i][j * cinfo.output_components + 1]; // Green
            pixels[(i * (*width) + j) * 3 + 2] = buffer[i][j * cinfo.output_components + 2]; // Blue
        }
    }

    // Cleanup
    for (uint32_t i = 0; i < cinfo.output_height; ++i)
    {
        free(buffer[i]);
    }
    free(buffer);

    // Finish decompression
    jpeg_finish_decompress(&cinfo);

    // Clean up and release resources
    fclose(file);
    jpeg_destroy_decompress(&cinfo);

    return pixels;
}

void writeResizedJPG(const char *filename, uint8_t *pixels, uint16_t originalWidth, uint16_t height, uint16_t newWidth)
{
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;;

    FILE *file = fopen(filename, "wb");
    if (file == NULL)
    {
        fprintf(stderr, "Error opening file %s for writing\n", filename);
        return;
    }

    // Initialize the JPEG compression object
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    // Specify the destination file
    jpeg_stdio_dest(&cinfo, file);

    // Set image parameters for resized image
    cinfo.image_width = newWidth;
    cinfo.image_height = height;
    cinfo.input_components = 3; // RGB
    cinfo.in_color_space = JCS_RGB;

    // Set default compression parameters
    jpeg_set_defaults(&cinfo);

    // Set quality (0 to 100, higher is better quality)
    jpeg_set_quality(&cinfo, 75, TRUE);

    // Start compression
    jpeg_start_compress(&cinfo, TRUE);

    JSAMPROW row_pointer[1];
    while (cinfo.next_scanline < cinfo.image_height)
    {
        // Here, we need to point to the correct place in the pixel array for each row
        row_pointer[0] = &pixels[cinfo.next_scanline * originalWidth * cinfo.input_components];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    // Finish compression
    jpeg_finish_compress(&cinfo);

    // Clean up and release resources
    fclose(file);
    jpeg_destroy_compress(&cinfo);
}

float computeError(uint8_t *RGBImageV1, uint8_t *RGBImageV2, int width, int height)
{
    float error = 0;
    for (int i = 0; i < width * height * 3; i++)
    {
        error += abs(RGBImageV1[i] - RGBImageV2[i]);
    }
    error /= (width * height * 3);
    return error;
}

void printDeviceInfo() {
    cudaDeviceProp devProp; // Change variable name to devProp
    CHECK(cudaGetDeviceProperties(&devProp, 0));

    printf("__________ GPU Device Information __________\n");
    printf("| %-30s %s\n", "Name:", devProp.name);
    printf("| %-30s %d.%d\n", "Compute Capability:", devProp.major, devProp.minor);
    printf("| %-30s %d\n", "Num SMs:", devProp.multiProcessorCount);
    printf("| %-30s %d\n", "Max Threads per SM:", devProp.maxThreadsPerMultiProcessor);
    printf("| %-30s %d\n", "Max Warps per SM:", devProp.maxThreadsPerMultiProcessor / devProp.warpSize);
    printf("| %-30s %zu bytes\n", "Global Memory (GMEM):", devProp.totalGlobalMem);
    printf("| %-30s %zu bytes\n", "Shared Memory per SM:", devProp.sharedMemPerMultiprocessor);
    printf("| %-30s %zu bytes\n", "Shared Memory per Block:", devProp.sharedMemPerBlock);
    printf("|____________________________________________\n");
}