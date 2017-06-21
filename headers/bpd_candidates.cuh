/*
 * bpd_subharmonics.h
 *
 *  Created on: January 11, 2017
 *  Author: Keith Azzopardi
 */

#ifndef BPD_CANDIDATES_CUH_
#define BPD_CANDIDATES_CUH_

#include "bpd_cudacommon.cuh"
#include "../headers/bpd_search.cuh"


typedef struct RDERIVS {
    double pow;	     /* Power normalized with local power   */
    double phs;	     /* Signal phase                        */
    double dpow;     /* 1st deriv of power wrt fourier freq */
    double dphs;     /* 1st deriv of phase wrt fourier freq */
    double d2pow;    /* 2nd deriv of power wrt fourier freq */
    double d2phs;    /* 2nd deriv of phase wrt fourier freq */
    double locpow;   /* Local mean power level              */
} rderivs;

typedef struct Candidates
{
	  float power;         /* Summed power level (normalized) */
	  float sigma;         /* Equivalent sigma based on numindep (above) */
	  int numharm;         /* Number of harmonics summed */
	  double r;            /* Fourier freq of first harmonic */
	  double z;            /* Fourier f-dot of first harmonic */
	  double *pows;        /* Optimized powers for the harmonics */
	  double *hirs;        /* Optimized freqs for the harmonics */
	  double *hizs;        /* Optimized fdots for the harmonics */
	  rderivs *derivs;     /* An rderivs structure for each harmonic */
} Candidates;



Candidates *search_ffdot_powers(ffdotInfo *ffdot, int numharm, float powcut, long long numindep, Candidates *cands);

/* =======================================================================
 * Optimize Candidates
 * =======================================================================
 */
void sortCandidates();
void eliminateHarmonics();
void optimizeCandidates();
void calculateProperties();

#endif /* BPD_CANDIDATES_CUH_ */
