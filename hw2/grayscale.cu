/*
 *  grayscale.cu: an application for converting png files to color to grayscale
 *
 *  @version: 1.0 
 *  @original: Lamont Samuels 2024 
 *
 */
extern "C" {
    #include "png_flatten.h"
}
#include <stdio.h>


/*
 * image_to_grayscale - converts a flatten color image to grayscale 
 *
 * image: the image of pixels 
 *
 * width: a pointer to an integer where the function will store the width read from the png file. 
 *
 * height: a pointer to an integer where the function will store the height read from the png file.  
 *
 * Returns: Nothing. 
 * 
 */
void image_to_grayscale(unsigned char* image, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = (y * width + x) * 4;
            unsigned char r = image[index + 0];
            unsigned char g = image[index + 1];
            unsigned char b = image[index + 2];
            unsigned char gray = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
            image[index + 0] = gray;
            image[index + 1] = gray;
            image[index + 2] = gray;
        }
    }
}

int main(int argc, char *argv[]) {

    // Read in the number of arguments 
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_png_file> <output_png_file>\n", argv[0]);
        return 1;
    }

    //Load in the image 
    int width, height;
    unsigned char *image = png_flatten_load(argv[1], &width, &height);

    // Verify the image was loaded correclty 
    if (!image) {
        return EXIT_FAILURE;
    }

    //Convert the image to grayscale 
    image_to_grayscale(image, width, height);

    //Save teh coverted image to the output png file. 
    if (png_flatten_save(argv[2], image, width, height) != 0) {
        free(image);
        return EXIT_FAILURE;
    }

    free(image);
    return 0;
}
