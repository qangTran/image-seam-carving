#include "include/utility.h"
#include "include/seam_carving.h"
#include <stdint.h>

int main(int argc, char **argv)
{
    printDeviceInfo();

    // check if number of parameter is correct else return error
    if (argc != 7)
    {
        printf("ERROR: Wrong number of parameters!!!\n");
        return 1;
    }

    // read parameter
    char *imageFilePath;
    uint16_t newWidth;
    int blockSize;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-i") == 0)
        {
            imageFilePath = argv[i + 1];
        }
        else if (strcmp(argv[i], "-nw") == 0)
        {
            newWidth = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-bz") == 0)
        {
            blockSize = atoi(argv[i + 1]);
        }
    }

    // Read image
    uint16_t width, height;
    uint8_t *RGBImage = readJPG(imageFilePath, &width, &height);
    if (RGBImage == NULL)
    {
        fprintf(stderr, "Error reading file %s\n", imageFilePath);
        return 1;
    }
    printf("Image size: %dx%d\n", width, height);

    // Check if the new size is acceptable
    if (newWidth > width)
    {
        printf("ERROR: Resolution not acceptable!!!\n");
        return 1;
    }
    printf("Resized image size: %dx%d\n", newWidth, height);

    // Seam carving on host
    uint8_t *resizedImageUsingHost = seamCarving(RGBImage, width, height, newWidth, 0, blockSize);
    writeResizedJPG("resizedImageUsingHost.jpg", resizedImageUsingHost, width, height, newWidth);

    // Seam carving on device V1
    uint8_t *resizedImageUsingDeviceV1 = seamCarving(RGBImage, width, height, newWidth, 1, blockSize);
    writeResizedJPG("resizedImageUsingDeviceV1.jpg", resizedImageUsingDeviceV1, width, height, newWidth);
    printf("Error Distance: %.3f\n", computeError(resizedImageUsingDeviceV1, resizedImageUsingHost, newWidth, height));

    // Seam carving on device v2
    uint8_t *resizedImageUsingDeviceV2 = seamCarving(RGBImage, width, height, newWidth, 2, blockSize);
    writeResizedJPG("resizedImageUsingDeviceV2.jpg", resizedImageUsingDeviceV2, width, height, newWidth);
    printf("Error Distance: %.3f\n", computeError(resizedImageUsingDeviceV2, resizedImageUsingHost, newWidth, height));


    // Free memory
    free(RGBImage);
    free(resizedImageUsingHost);
    free(resizedImageUsingDeviceV1);
    free(resizedImageUsingDeviceV2);

    return 0;
}
