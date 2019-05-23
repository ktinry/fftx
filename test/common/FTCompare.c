/*
 * FFTX Copyright (c) 2019, The Regents of the University of California, through 
 * Lawrence Berkeley National Laboratory (subject to receipt of any required 
 * approvals from the U.S. Dept. of Energy), Carnegie Mellon University and 
 * SpiralGen, Inc.  All rights reserved.
 * 
 * If you have questions about your rights to use or distribute this software, 
 * please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.
 * 
 * NOTICE.  This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government  consequently  retains certain rights. As such, 
 * the U.S. Government has been granted for itself and others acting on its 
 * behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software 
 * to reproduce, distribute copies to the public, prepare derivative works, and 
 * perform publicly and display publicly, and to permit others to do so.
 */

#include <FTCompare.h>
#include <math.h>


double test_compare_r1d(int n, fftx_real *vec1, fftx_real *vec2)
{
	double maxdiff = 0.0;
	double diff;
	double norm;
	
	for (int i = 0; i < n; i++) {
		diff = fabs((double)vec1[i] - (double)vec2[i]);
		norm = fabs((double)vec1[i]);
		if (norm != 0.0) {
			diff /= norm; // relative error
		} 
		if (diff > maxdiff) {
			maxdiff = diff;
		}
	}
	
	return maxdiff;
}

double test_compare_c1d(int n, fftx_complex *vec1, fftx_complex *vec2)
{
	return test_compare_r1d(2 * n, (fftx_real *)vec1, (fftx_real *)vec2);
}
