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
#include "fftx.h"


// flags for FFTX
#define MY_FFTX_MODE		0
#define MY_FFTX_MODE_TOP	(FFTX_ESTIMATE | MY_FFTX_MODE)
#define MY_FFTX_MODE_SUB	(MY_FFTX_MODE_TOP | FFTX_FLAG_SUBPLAN)


// globals for the convolution
// bad style, but good enough for the example
fftx_temp_real tmp1, tmp4;
fftx_temp_complex tmp2, tmp3;

// pointwise scaling function: complex multiply by symbol
void complex_scaling(fftx_complex *in, fftx_complex *out, fftx_complex *data) {
	FFTX_COMPLEX_TEMP(t);
	FFTX_COMPLEX_MOV(t, in);
	FFTX_COMPLEX_MULT(t, data);
	FFTX_COMPLEX_MOV(out, t);
}


// produces an fftx_plan to perform the whole convolution

fftx_plan pruned_real_convolution_plan(fftx_real *in, fftx_real *out, fftx_complex *symbol, int n, int n_in, int n_out, int n_freq) {
	int rank = 3,			// 3D = rank 3
		batch_rank = 0,		// no batch, thus batch rank 0
		numsubplans = 5;	// need 5 FFTX subplans for pruned convolution
	// FFTX iodim definitions for 3D + pruning
	fftx_iodim padded_dims[] = { { n, 1, 1 }, { n, n, n }, { n, n*n, n*n } },
		freq_dims[] = { { n_freq, 1, 1 }, { n, n_freq, n_freq }, { n, n_freq*n, n_freq*n } },
		batch_dims = { 1, 1, 1 };					// no batching
	fftx_iodimx in_dimx[] = { { n_in, 0, 0, 0, 1, 1, 1 }, { n_in, 0, 0, 0, n_in, n, 1 }, { n_in, 0, 0, 0, n_in*n_in, n*n, 1 } },
		out_dimx[] = { { n_out, n - n_out, 0, 0, 1, 1, 1 }, { n_out, n - n_out, 0, 0, n, n_out, 1 }, { n_out, n - n_out, 0, 0, n*n, n_out*n_out, 1 } },
		freq_dimx[] = { { n, 0, 0, 0, 1, 1, 1 },{ n, 0, 0, 0, n, n, n },{ n_freq, 0, 0, 0, n*n, n*n, n*n } },
		batch_dimx = { 1, 0, 0, 0, 1, 1, 1, };				// no batching
	fftx_plan plans[5];										// intermediate sub-plans
	fftx_plan p;											// top-level plan

	// create zero-initialized rank-dimensional temporary data cube given by padded_dims for zero-padding the input
	tmp1 = fftx_create_zero_temp_real(rank, padded_dims);

	// copy a rank-dimensional data cube given by in_dims into a contiguous rank-dimensional zero-initialized temporary
	plans[0] = fftx_plan_guru_copy_real(rank, in_dimx, in, tmp1, MY_FFTX_MODE_SUB);

	// RDFT on the padded data
	tmp2 = fftx_create_temp_complex(rank, freq_dims);
	plans[1] = fftx_plan_guru_dft_r2c(rank, padded_dims, batch_rank, &batch_dims, tmp1, tmp2, MY_FFTX_MODE_SUB);

	// pointwise operation
	tmp3 = fftx_create_temp_complex(rank, freq_dims);
	plans[2] = fftx_plan_guru_pointwise_c2c(rank, freq_dimx, batch_rank, &batch_dimx,
		tmp2, tmp3, symbol, (fftx_callback)complex_scaling, MY_FFTX_MODE_SUB | FFTX_PW_POINTWISE);

	// iRDFT on the scaled data
	tmp4 = fftx_create_temp_real(rank, padded_dims);
	plans[3] = fftx_plan_guru_dft_c2r(rank, padded_dims, batch_rank, &batch_dims, tmp3, tmp4, MY_FFTX_MODE_SUB);

	// copy out the rank-dimensional data cube given by out_dims of interest
	plans[4] = fftx_plan_guru_copy_real(rank, out_dimx, tmp4, out, MY_FFTX_MODE_SUB);

	// create the top level plan. this copies the sub-plan pointers, so they can be lost
	p = fftx_plan_compose(numsubplans, plans, MY_FFTX_MODE_TOP);

	// plan to be used with fftx_execute()
	return p;
}


// destroys the plan and temporaries to clean up 

void pruned_real_convolution_destroy(fftx_plan p) {
	// cleanup
	fftx_destroy_temp_real(tmp1);
	fftx_destroy_temp_complex(tmp2);
	fftx_destroy_temp_complex(tmp3);
	fftx_destroy_temp_real(tmp4);
	fftx_destroy_plan_recursive(p);
}


// main function
//
// parameterize problem size at the top
// prints input, output and temporaries in FFTX_MODE_OBSERVE mode
// temporaries and sub-plans are protected and need to be
// un-protected if they need to be inspected or executed separately

int main(void) {
	// input, output, FFTX plan
	fftx_real *in,
		*out;
	fftx_complex *symbol;
	fftx_plan p;

	// problem size definition
	int n = 8,
		n_in = 4,
		n_out = 5,
		n_freq = n / 2 + 1;
	fftx_iodim dims_in[] = { { n_in, 1, 1 }, { n_in, 1, 1 }, { n_in, 1, 1 } },
		dims_out[] = { { n_out, 1, 1 }, { n_out, 1, 1 }, { n_out, 1, 1 } },
		dims_freq[] = { { n_freq, 1, 1 }, { n, 1, 1 }, { n, 1, 1 } };
	double symval = 1.0 / ((double)n*n*n);
	double *planiodata[3];	// used for persistent calls

	// initialize FFTX in FFTX_MODE_OBSERVE
	fftx_init(MY_FFTX_MODE);

	// allocate input, symbol, and output
	planiodata[0] = in = (fftx_real*)fftx_create_data_real(3, dims_in);
	planiodata[1] = out = (fftx_real*)fftx_create_data_real(3, dims_out);
	symbol = (fftx_complex*)fftx_create_data_complex(3, dims_freq);
	planiodata[2] = (double*)symbol;

	// initialize input
	for (int i = 0; i < n_in*n_in*n_in; i++)
		in[i] = i + 1;

	// initialize symbol
	for (int i = 0; i < n_freq*n*n; i++) {
		symbol[i][0] = symval;
		symbol[i][1] = 0.0;
	}

	// build FFTX descriptor for trivial convolution
	p = pruned_real_convolution_plan(in, out, symbol, n, n_in, n_out, n_freq);

	// execute the trivial convolution
	fftx_execute(p);

	// print input and output vectors
	printf("in = ");
	fftx_print_vector_real(in, n_in*n_in*n_in);

	printf("out = ");
	fftx_print_vector_real(out, n_out*n_out*n_out);

	// cleanup
	fftx_destroy_data_real(in);
	fftx_destroy_data_real(out);
	fftx_destroy_data_complex(symbol);

	pruned_real_convolution_destroy(p);

	// shut down FFTX
	fftx_shutdown();
}
