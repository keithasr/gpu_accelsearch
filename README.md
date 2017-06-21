# gpu_accelsearch
GPU-based Fourier Domain acceleration searching

This has been optimized for NVIDIA Kepler architecture. Might need additional tweaking for newer architectures.

The Fourier-Domain acceleration search technique is an optimized version of the algorithm described in http://adsabs.harvard.edu/abs/2002AJ....124.1788R. The algorithm described in this paper is also used in PRESTO CPU-accelerated implementation.

You might need additional tools such as PRESTO to perform de-dispersion and noise excision. 

The algorithm involves the following procedures:
1. Generating kernel template responses.
2. Applying real-to-complex DFT of time-series of a specific DM trial.
3. Whitened output of the DFT to remove red-noise.
4. Dividing whitened signal in segments of equal length. Each segment has
an overlapping region from the previous segment (using Overlap-and-Save
method).
    (a) Generating `local Fourier amplitudes'.
    (b) Applying complex-to-complex DFT of segments.
    (c) Fourier Domain convolution of the segment with all kernel templates,
    producing a f - f dot plane.
    (d) Inverse complex-to-complex DFT of each convolved kernel template.
    (e) Calculating Fourier power of each harmonic.
    (f) Adding harmonics to fundamental harmonic (Harmonic summing).
    (g) Searching all folded harmonics powers (NOT IMPLEMENTED - See below for further detail)

NOTE that that acceleration searching requires candidate selection and a procedure to detect pulsars, including binary pulsars, from the folded Fourier power. In this implementation, though the search procedure was considered, the candidate selection was limited to the gener-
ation of an empty list of potential candidates. The simplest method for selecting these candidates is to store all candidates above a certain S/N ratio and sort them accordingly. Several GPU-based sorting algorithms are available but these depends on the desired searching procedure.
