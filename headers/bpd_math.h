/*
 * bpd_math.h
 *
 *  Created on: September 9, 2016
 *  Author: Keith Azzopardi
 */

#ifndef BPD_MATH_H_
#define BPD_MATH_H_

#include <math.h>

#define NEAREST_INT(x) (int) (x < 0 ? ceil(x - 0.5) : floor(x + 0.5))

#define ELEM_SWAP(a,b) { register float t=(a);(a)=(b);(b)=t; }
#define PI            3.1415926535897932384626433832795028841971693993751
#define SQRT2         1.4142135623730950488016887242096980785696718753769
#define PIBYTWO       1.5707963267948966192313216916397514420985846996876

float fast_median(float arr[], int n);
float inverse_square_root(float x);
int fast_log2(int n);

/// Return rounded value.
int toNearestInt (double x);
/// Return the first value of 2^n >= x
long long get2PowerNGreaterThan(long long x);

#endif /* BPD_MATH_H_ */
