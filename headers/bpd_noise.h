/*
 * bpd_noise.h
 *
 *  Created on: September 9, 2016
 *  Author: Keith Azzopardi
 */

#ifndef BPD_NOISE_H_
#define BPD_NOISE_H_

#include "bpd_common.h"


/// Flunctuations in the receiver and/or data acquisition system often manifest themselves
/// via a significant low-frequency or red noise component when viewed in the Fourier domain.
void deredden(cufftComplex *fft, unsigned numOfBins);

#endif /* BPD_NOISE_H_ */
