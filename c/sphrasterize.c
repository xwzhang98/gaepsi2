/**
 * sphrasterize.c - Core SPH rasterization implementation
 * 
 * This file implements the algorithm for rendering Smoothed Particle Hydrodynamics (SPH)
 * particles onto a 2D image. It handles converting 3D particle data with smoothing
 * lengths into pixel values with proper kernel weighting.
 *
 * The main functions are:
 * - gsph_image_init: Initialize an image structure
 * - gsph_rasterize: Render a particle onto the image
 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "gaepsi.h"

/**
 * Write a single-precision value to an image at the specified position
 *
 * This is an optimized function for writing float values to an image.
 * It uses OpenMP atomic operations to safely update pixel values from multiple threads.
 *
 * @param img   Pointer to the image structure
 * @param x     X-coordinate in the image
 * @param y     Y-coordinate in the image
 * @param value Array of channel values to write
 */
inline void gsph_image_write_single(GSPHImage * img, int x, int y, double * value) {
    int c;  /* Channel index */
    char * ptr = img->data;  /* Pointer to the image data */
    
    /* Calculate the offset to the pixel position using strides */
    ptr += x * img->strides[0] + y * img->strides[1];
           
    /* For each channel in the image */
    for(c = 0; c < img->size[2]; c++, ptr += img->strides[2]) {
#pragma omp atomic
        /* Atomically add the value to the pixel (allows thread-safe updates) */
        *((float*) ptr) += value[c];
    }
}

/**
 * Write a double-precision value to an image at the specified position
 *
 * Similar to gsph_image_write_single but for double-precision images.
 *
 * @param img   Pointer to the image structure
 * @param x     X-coordinate in the image
 * @param y     Y-coordinate in the image
 * @param value Array of channel values to write
 */
inline void gsph_image_write_double(GSPHImage * img, int x, int y, double * value) {
    int c;  /* Channel index */
    char * ptr = img->data;  /* Pointer to the image data */
    
    /* Calculate the offset to the pixel position using strides */
    ptr += x * img->strides[0] + y * img->strides[1];
           
    /* For each channel in the image */
    for(c = 0; c < img->size[2]; c++, ptr += img->strides[2]) {
#pragma omp atomic
        /* Atomically add the value to the pixel (allows thread-safe updates) */
        *((double*) ptr) += value[c];
    }
}

/**
 * Generic image write function that chooses the appropriate precision
 *
 * This function dispatches to either the single or double precision
 * write function based on the image's itemsize.
 *
 * @param img   Pointer to the image structure
 * @param x     X-coordinate in the image
 * @param y     Y-coordinate in the image
 * @param value Array of channel values to write
 */
inline void gsph_image_write(GSPHImage * img, int x, int y, double * value) 
{
    if (img->itemsize == 8) {
        /* Use double precision for 8-byte items */
        gsph_image_write_double(img, x, y, value);
    } else {
        /* Use single precision for 4-byte items */
        gsph_image_write_single(img, x, y, value);
    }
}

/**
 * Initialize an image structure with the given parameters
 *
 * This function sets up the image structure with the provided dimensions,
 * data pointer, and other parameters.
 *
 * @param image   Pointer to the image structure to initialize
 * @param dtype   Data type string ("f4" for float, "f8" for double)
 * @param size    Array of dimensions [width, height, channels]
 * @param strides Array of strides for each dimension or NULL for contiguous
 * @param data    Pointer to the image data buffer
 */
void 
gsph_image_init(GSPHImage * image, 
        char * dtype, 
        int size[3], 
        ptrdiff_t * strides, 
        void * data) 
{
    /* Determine itemsize from dtype string */
    if(dtype[strlen(dtype) - 1] == '8') {
        image->itemsize = 8;  /* 8 bytes for double precision */
    } else {
        image->itemsize = 4;  /* 4 bytes for single precision */
    }
    
    /* Copy the dimensions */
    image->size[0] = size[0];  /* Width */
    image->size[1] = size[1];  /* Height */
    image->size[2] = size[2];  /* Number of channels */
    
    /* Set up strides (offsets between elements) */
    if(strides) {
        /* Use provided strides if available */
        image->strides[0] = strides[0];
        image->strides[1] = strides[1];
        image->strides[2] = strides[2];
    } else {
        /* Otherwise calculate contiguous strides */
        image->strides[2] = image->itemsize;  /* Channel stride = itemsize */
        image->strides[1] = image->strides[2] * size[2];  /* Row stride */
        image->strides[0] = image->strides[1] * size[1];  /* Column stride */
    }
    
    /* Store the data pointer */
    image->data = data;
}

/**
 * Rasterize an SPH particle onto an image
 *
 * This is the core function that renders a single SPH particle to an image
 * by distributing its values according to the SPH kernel.
 *
 * @param image     Pointer to the output image
 * @param sphkernel Function pointer to the SPH kernel to use
 * @param pos       Position of the particle [x, y]
 * @param sml       Smoothing length of the particle
 * @param mvalue    Array of channel values for the particle
 */
void 
gsph_rasterize(GSPHImage * image, GSPHKernel sphkernel,
        double pos[2], double sml, double * mvalue) 
{
    /* sml here is half of Gadget's cubic spline sml (gadget uses support). */
    int * size = image->size;
    int nc = image->size[2];  /* Number of channels */

    /* Check for empty image */
    if (size[0] == 0) return;
    if (size[1] == 0) return;

    double bit = 0;  /* Normalization factor */
    int x, y;        /* Pixel coordinates */
    double r;        /* Distance */
    int k;           /* Loop counter */

    /* Handle very small particles (smaller than a pixel) */
    if(sml < 1.0) {
        /* For very small particles, use a simple point sample */
        /* This is equivalent to CIC (Cloud-In-Cell) interpolation */
        x = pos[1];  /* Note x/y swap: pos[0] is y and pos[1] is x */
        y = pos[0];
        
        /* Check bounds */
        if (x < 0) return;
        if (y < 0) return;
        if (x >= size[1]) return;
        if (y >= size[0]) return;
        
        /* Write the unmodified value to the single pixel */
        gsph_image_write(image, x, y, mvalue);
    } else {
        /* For regular-sized particles, use kernel interpolation */
        double save[128 * 128];  /* Buffer for kernel values */
        int s = 0;               /* Index into save buffer */
        int usekernel = 0;       /* Flag to indicate whether to use precise kernel */

        /* Calculate the particle's bounding box in the image */
        int min[2]; 
        int max[2]; 
        for(k = 0; k < 2; k ++) {
            min[k] = pos[k] - sml;
            max[k] = pos[k] + sml;
            
            /* Check if the particle is fully outside the image */
            if (max[k] < 0) return;
            if (min[k] >= size[k]) return;
            
            /* Clamp the bounds to the image dimensions */
            if (min[k] < 0) min[k] = 0;
            if (max[k] < 0) max[k] = 0;
            if (min[k] >= size[k]) min[k] = size[k] - 1;
            if (max[k] >= size[k]) max[k] = size[k] - 1;
        }
        
        /* For particles of reasonable size, use the precise kernel */
        if(sml < 60) {
            /* Loop over all pixels potentially affected by the particle */
            for(y = pos[0] - sml; y <= pos[0] + sml; y++) {
            for(x = pos[1] - sml; x <= pos[1] + sml; x++) {
                /* Calculate the distance from the pixel to the particle center */
                double dx = x - pos[1];
                double dy = y - pos[0];
                double r = sqrt(dx * dx + dy * dy) / (sml);
                
                /* Evaluate the kernel at this distance */
                r = sphkernel(r);
                
                /* Accumulate for normalization */
                bit += r;
                
                /* If this pixel is within the image bounds, save its kernel value */
                if(x >= min[1] && x <= max[1] 
                        && y >= min[0] && y <= max[0]) {
                    save[s] = r;
                    s++;
                }
            }
            }
            usekernel = 1;  /* Flag to use the precise kernel */
        } else {
            /* For very large particles, use a rectangular approximation */
            /* This is an optimization for efficiency */
            bit = (4 * sml * sml);
        }

        /* Calculate normalization factor */
        bit = 1.0 / bit;
        double sml2 = sml * sml;
        
        /* Reset the buffer index */
        s = 0;
        
        /* Loop over all pixels in the particle's bounding box */
        for(y = min[0]; y <= max[0]; y++) {
        for(x = min[1]; x <= max[1]; x++) {
            double w;
            
            /* Determine the weight for this pixel */
            if(usekernel) {
                /* Use the pre-calculated kernel value */
                w = save[s] * bit;
            } else {
                /* Use constant weight for rectangular approximation */
                w = 1.0 * bit;
            }
            s ++;
            
            /* If the weight is non-zero, write to the image */
            if(w) {
                double tmp[nc];
                int i;
                
                /* Scale the particle values by the kernel weight */
                for(i = 0; i < nc; i ++) {
                    tmp[i] = mvalue[i] * w;
                }
                
                /* Write the weighted values to the image */
                gsph_image_write(image, y, x, tmp);
            }
        }
        }
    }
}