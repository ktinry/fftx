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

#include <stdio.h>
#include <stdlib.h>
#include <FTCompare.h>
#include <FTCreateData.h>


int main(int argc, char *argv[])
{
	int n = 256;
	unsigned wflags = FFTW_ESTIMATE;
	unsigned xflags = 0;
	double max_diff_ok = .02;

	if (argc > 1) {
		long newN = strtol(argv[1], NULL, 0);
		if (newN > 1) {
			n = newN;
		}
	}

	fftx_init(xflags);

	fftx_real    *in1  = test_create_realdata_1d(n);
	fftx_real    *in2  = test_copy_realdata_1d(n, in1);
	fftx_complex *out1 = fftx_internal_malloc(n * sizeof(fftx_complex));
	fftx_complex *out2 = fftx_internal_malloc(n * sizeof(fftx_complex));

	fftw_plan wplan = fftw_plan_dft_r2c_1d(n, (double *)in1, (fftw_complex *)out1, wflags);
	fftw_execute(wplan);
	fftw_destroy_plan(wplan);

	fftx_plan xplan = fftx_plan_dft_r2c_1d(n, in2, out2, xflags | wflags);
	fftx_execute(xplan);
	fftx_destroy_plan(xplan);

	double maxdiff = test_compare_c1d(n, out1, out2);

	fftx_internal_free(in1);
	fftx_internal_free(in2);
	fftx_internal_free(out1);
	fftx_internal_free(out2);

	fftx_shutdown();

	if (maxdiff > max_diff_ok) {
		fprintf(stderr, "%s, N = %d, maxdiff: %g\n", argv[0], n, maxdiff);
		return EXIT_FAILURE;
	}

	printf("%s, N = %d, maxdiff: %g\n", argv[0], n, maxdiff);

	return EXIT_SUCCESS;
}