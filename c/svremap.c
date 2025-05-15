/**
 * svremap.c - Survey Volume Remapping Implementation
 *
 * This file implements the Survey Volume Remapping (SVR) algorithm, which transforms 
 * a cubic volume (typically used in cosmological simulations) into a different shape
 * for better visualization or analysis.
 *
 * The main data structure is SVRemap, which holds the transformation matrix and related data.
 * The main functions are:
 * - svremap_init: Initialize an SVRemap structure with a transformation matrix
 * - svremap_apply: Apply the remapping to coordinates
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "gaepsi.h"

/**
 * Initialize an SVRemap structure with a given transformation matrix.
 *
 * This function takes a 3x3 integer transformation matrix and initializes
 * the SVRemap structure, calculating an orthonormal basis and the size of
 * the remapped volume.
 *
 * @param r     Pointer to the SVRemap structure to initialize
 * @param remap 3x3 integer transformation matrix as a flattened array
 * @return      0 on success
 */
int svremap_init(SVRemap * r, int remap[3][3]) {
    /* BAS will hold our orthonormal basis */
    double BAS[3][3] = {0};
    int i, j;
    double uv, uu, uvp;

    /* Copy the remap matrix */
    for(i = 0; i < 3; i++) {
        for(j = 0; j < 3; j++) {
            r->remap[i][j] = remap[i][j];
        } 
    }

    /* 
     * The Gram-Schmidt orthogonalization process:
     * We convert the input matrix into an orthonormal basis.
     */
    
    /* u1 = v1 (first basis vector is same as first input vector) */
    for(j = 0; j < 3; j++) {
        BAS[0][j] = remap[0][j];
    } 

    /* 
     * u2 = v2 - proj(u1, v2) 
     * u3' = v3 - proj(u1, v3)
     * (subtract the projection onto the first basis vector)
     */
    uv = 0;   /* Dot product of u1 and v2 */
    uvp = 0;  /* Dot product of u1 and v3 */
    uu = 0;   /* Dot product of u1 with itself (squared length) */
    
    /* Calculate dot products */
    for(j = 0; j < 3; j ++) {
        uv += remap[1][j] * BAS[0][j];
        uu += BAS[0][j] * BAS[0][j];
        uvp += remap[2][j] * BAS[0][j];
    }
    
    /* Subtract projections to get orthogonal vectors */
    for(j = 0; j < 3; j++) {
        BAS[1][j] = remap[1][j] - uv / uu * BAS[0][j];
        BAS[2][j] = remap[2][j] - uvp / uu * BAS[0][j];
    } 

    /* 
     * u3 = u3' - proj(u2, u3')
     * (make the third vector orthogonal to the second as well)
     */
    uv = 0;
    uu = 0;
    
    /* Calculate dot products */
    for(j = 0; j < 3; j ++) {
        uv += BAS[1][j] * BAS[2][j];
        uu += BAS[1][j] * BAS[1][j];
    }
    
    /* Subtract projection */
    for(j = 0; j < 3; j++) {
        BAS[2][j] = BAS[2][j] - uv / uu * BAS[1][j];
    } 

    /* 
     * Now normalize the basis vectors to get an orthonormal basis,
     * and store it in the transformation matrix T.
     */
    memset(r->T, 0, sizeof(double) * 16);
    for(i = 0; i < 3; i++) {
        uu = 0;
        /* Calculate vector length */
        for(j = 0; j < 3; j++) {
            uu += BAS[i][j] * BAS[i][j];
        }
        uu = sqrt(uu);
        
        /* Normalize and store in T */
        for(j = 0; j < 3; j++) {
            r->T[i][j] = BAS[i][j] / uu;
        }
    }
    r->T[3][3] = 1.0;  /* Set homogeneous coordinate scaling */

    /* 
     * Calculate the size of the remapped volume.
     * This is the dot product of each input vector with the
     * corresponding basis vector, giving the size along each axis.
     */
    for(i = 0; i < 3; i++) {
        r->size[i] = 0;
        for(j = 0; j < 3; j++) {
            r->size[i] += remap[i][j] * r->T[i][j];
        }
        
        /* Ensure size is positive (flip basis vector if needed) */
        if (r->size[i] < 0) {
            r->size[i] *= -1;
            for(j = 0; j < 3; j++) {
                r->T[i][j] *= -1;
            }
        }
    }
    
    return 0;
}

/*
 * Helper function to test if a point falls within the remapped volume.
 *
 * This function takes the integer coordinates I and fractional offset x,
 * transforms them to the new coordinate system, and checks if the resulting
 * point y is within the remapped volume.
 *
 * @param r     Pointer to the SVRemap structure
 * @param x     Fractional offset [0,1] in each dimension
 * @param y     Output coordinates in the remapped system
 * @param I     Integer coordinates (cell indices)
 * @return      Badness measure (0 if point is within the volume)
 */
static double svremap_test(SVRemap * r, double x[3], double y[3], int I[3]) {
    double v1[4];
    double v2[4];
    int i;
    
    /* Create homogeneous coordinates from I + x */
    for(i = 0; i < 3; i ++) {
        v1[i] = x[i] + I[i];
    }
    v1[3] = 1.0;
    
    /* Apply the transformation matrix */
    gtmat_apply(r->T, v1, v2);
    
    /* Copy the result to output */
    for(i = 0; i < 3; i ++) {
        y[i] = v2[i];
    }
    
    /* 
     * Calculate "badness" - how far the point is outside the volume.
     * 0 means the point is inside the volume.
     */
    double badness = 0.0;
    for(i = 0; i < 3; i ++) {
        if(v2[i] < 0.0) {
            badness = fmax(badness, - v2[i]);
        }
        if(v2[i] > r->size[i]) {
            badness = fmax(badness, v2[i] - r->size[i]);
        }
    }
    
    return badness;
}

/**
 * Calculate the bounds of integer coordinates that could map into the volume.
 *
 * This is a helper function for svremap_apply that finds the range of integer
 * coordinates I that could potentially map into the remapped volume.
 *
 * @param r     Pointer to the SVRemap structure
 * @param Imin  Output array for minimum bounds
 * @param Imax  Output array for maximum bounds
 */
static void svremap_bounds(SVRemap * r, int Imin[3], int Imax[3]) {
    /* Calculate the inverse transformation */
    GTMatrix inv;
    gtmat_inverse(inv, r->T);
    
    int i;
    int a;
    
    /* 
     * Test all 8 corners of the remapped volume (2^3 = 8)
     * to find the bounds of integer coordinates.
     */
    for(i = 0; i < 27; i ++) {
        double v1[4];
        double v2[3];
        v1[3] = 1.0;
        
        /* Create corner coordinates */
        for(a = 0; a < 3; a ++) {
            v1[a] = ((i >> a) & 1) * r->size[a];
        }
        
        /* Apply inverse transformation */
        gtmat_apply(inv, v1, v2);
        
        /* Update bounds */
        for(a = 0; a < 3; a ++) {
            if(floor(v2[a]) < Imin[a])  Imin[a] = floor(v2[a]);
            if(ceil(v2[a]) > Imax[a])  Imax[a] = ceil(v2[a]);
        }
    }
}

/**
 * Apply the survey volume remapping to coordinates.
 *
 * This function transforms coordinates from the input space to the
 * remapped space, finding the integer cell and fractional offset that
 * best maps to the given point.
 *
 * @param r     Pointer to the SVRemap structure
 * @param x     Input coordinates [0,1] in each dimension
 * @param y     Output coordinates in the remapped system
 * @param I     Output integer cell indices
 * @return      Badness measure (0 if perfect mapping found)
 */
double svremap_apply(SVRemap * r, double x[3], double y[3], int I[3]) {
    /* First try with the provided integer coordinates */
    if(svremap_test(r, x, y, I) == 0.0) return 0.0;
    
    /* If that doesn't work, search for better integer coordinates */
    int Imax[3] = {0}, Imin[3] = {0}, Ibest[3]; 
    double ybest[3];
    double bestbadness = 99.99; 
    int a;
    
    /* Calculate bounds for the search */
    svremap_bounds(r, Imin, Imax);

    /* Start search at the minimum bounds */
    for(a = 0; a < 3; a++) {
        I[a] = Imin[a];
    }
    
    /* Exhaustive search through all possible integer coordinates */
    int done = 0;
    while(!done) {
        /* Test this set of integer coordinates */
        double badness = svremap_test(r, x, y, I);
        
        /* If perfect, return immediately */
        if(badness == 0.0) return 0.0;
        
        /* If better than previous best, update best */
        if(badness < bestbadness) {
            for(a = 0; a < 3; a++) {
                Ibest[a] = I[a];
                ybest[a] = y[a];
            }
            bestbadness = badness;
        }

        /* Move to next set of integer coordinates */
        I[0] ++;
        for(a = 0; a < 3; a++) {
            if(I[a] != Imax[a] + 1) continue;
            if(a + 1 == 3) done = 1;
            I[a + 1] ++;
            I[a] = Imin[a];
        }
    }
    
    /* Use the best found integer coordinates */
    for(a = 0; a < 3; a++) {
        I[a] = Ibest[a];
        y[a] = ybest[a];
    }
    
    /* Return the badness measure */
    return bestbadness;
}