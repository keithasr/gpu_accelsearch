/*
 * bpd_common.h
 *
 *  Created on: September 9, 2016
 *  Author: Keith Azzopardi
 */

#ifndef BPD_COMMON_H_
#define BPD_COMMON_H_

// ACCEL_USELEN must be less than 65536 since we
// use unsigned short ints to index our arrays...
//
// #define ACCEL_USELEN 32000 // This works up to zmax=300 to use 32K FFTs
// #define ACCEL_USELEN 15660 // This works up to zmax=300 to use 16K FFTs
//   The following is probably the best bet for general use given
//   current speeds of FFTs.  However, if you only need to search up
//   to zmax < 100, dropping to 4K FFTs is a few percent faster.  SMR 131110
//#define ACCEL_USELEN 7470 // This works up to zmax=300 to use 8K FFTs
// #define ACCEL_USELEN 7960 // This works up to zmax=100 to use 8K FFTs
// #define ACCEL_USELEN 3850 // This works up to zmax=100 to use 4K FFTs
// #define ACCEL_USELEN 1820 // This works up to zmax=100 to use 2K FFTs

 #define ACCEL_USELEN 32000 // KEITH: This works up to zmax=300 to use 32K FFTs
// #define ACCEL_USELEN 15660 // KEITH: This works up to zmax=300 to use 16K FFTs
//#define ACCEL_USELEN 7470 // KEITH:This works up to zmax=300 to use 8K FFTs
//#define ACCEL_USELEN 3370 // KEITH: This works up to zmax=300 to use 4K FFTs
//#define ACCEL_USELEN 1320 // KEITH: This works up to zmax=300 to use 2K FFTs

#define ACCEL_NUMBETWEEN 2 		// Increment size in Fourier Frequency for each step.
#define ACCEL_DR  0.5 	   	    // Increment size in Fourier Frequency for each step.
#define ACCEL_RDR 2 			// Reciprocal of ACCEL_DR.
#define ACCEL_DZ  2 			// Increment size in Fourier F-dot for each step.
#define ACCEL_RDZ 0.5 			// Reciprocal of ACCEL_DZ
#define ACCEL_PADDING 2000  	// Data Padding size.
#define DBLCORRECT    1e-14

/// Number of bins (total) to average for local power. Must be an even number (1/2 on each side).
#define NUMLOCPOWAVG  20

/// Number of bins next to freq in question to ignore (on each side) when determining local power.
#define DELTAAVGBINS  5

/// Number of bins on each side of the central frequency  to sum for Fourier interpolation (low accuracy)
#define NUMFINTBINS   16

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdio.h>
#include "cufft.h"
#include "app_utilities.h"

#endif /* BPD_COMMON_H_ */
