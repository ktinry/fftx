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

#include <FTCreateData.h>

static double local_rand() {
	return (-50 + 100 * ((double)rand() / RAND_MAX));
}


fftx_real *test_create_realdata_1d(int n)
{
	fftx_real *vec = (fftx_real *)fftx_internal_malloc(n * sizeof(fftx_real));

	for (int i = 0; i < n; i++) {
		vec[i] = (fftx_real)local_rand();
	}

	return vec;
}


fftx_complex *test_create_complexdata_1d(int n)
{
	return (fftx_complex *)test_create_realdata_1d(2 * n);
}


fftx_real *test_copy_realdata_1d(int n, fftx_real *in)
{
	fftx_real *vec = (fftx_real *)fftx_internal_malloc(n * sizeof(fftx_real));

	for (int i = 0; i < n; i++) {
		vec[i] = in[i];
	}

	return vec;
}


fftx_complex *test_copy_complexdata_1d(int n, fftx_complex *in)
{
	return (fftx_complex *)test_copy_realdata_1d(2 * n, (fftx_real *)in);
}

