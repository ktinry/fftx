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
 
 // Planewave example

#include <stdio.h>
#include "fftx.h"


#define MY_FFTX_MODE		0
#define MY_FFTX_MODE_TOP	(FFTX_ESTIMATE | FFTX_MODE_OBSERVE)
#define MY_FFTX_MODE_SUB	(MY_FFTX_MODE_TOP | FFTX_FLAG_SUBPLAN)

// globals for the convolution
// bad style, but good enough for the example
fftx_temp_complex tmp1;

// produces an fftx_plan to perform the whole convolution

fftx_plan planewave_plan(fftx_complex *in, fftx_complex *out, int n, int n_in, int batch_rank, int batch_size) {
	int rank = 3,			// 3D = rank 3
		numsubplans = 9;
    // FFTX iodim definitions for 3D + pruning
	fftx_iodim padded_dims[] = {{ n, 1, 1 }, { n, n, n }, { n, n*n, n*n }},
		batch_dims = { batch_size, n*n*n, n*n*n };
	// fftx_iodimx in_dimx[] = {{ n_in, 0, (n-n_in)/2, 0, 1, 1, 1 }, // Copy dense cube in the middle
	// 						 { n_in, 0, (n-n_in)/2, 0, n_in, n, 1 }, 
	// 						 { n_in, 0, (n-n_in)/2, 0, n_in*n_in, n*n, 1 }},
	fftx_iodimx 
		in_dimx_000[] = {{ n_in/2, 0, n-n_in/2, 0, 1, 1, 1 }, 
						 { n_in/2, 0, n-n_in/2, 0, n_in, n, 1 }, 
						 { n_in/2, 0, n-n_in/2, 0, n_in*n_in, n*n, 1 }},
		in_dimx_100[] = {{ n_in/2, n_in/2, 0, 0, 1, 1, 1 }, 
						 { n_in/2, 0, n-n_in/2, 0, n_in, n, 1 }, 
						 { n_in/2, 0, n-n_in/2, 0, n_in*n_in, n*n, 1 }},
		in_dimx_010[] = {{ n_in/2, 0, n-n_in/2, 0, 1, 1, 1 }, 
						 { n_in/2, n_in/2, 0, 0, n_in, n, 1 }, 
						 { n_in/2, 0, n-n_in/2, 0, n_in*n_in, n*n, 1 }},
		in_dimx_110[] = {{ n_in/2, n_in/2, 0, 0, 1, 1, 1 }, 
						 { n_in/2, n_in/2, 0, 0, n_in, n, 1 }, 
						 { n_in/2, 0, n-n_in/2, 0, n_in*n_in, n*n, 1 }},
		in_dimx_001[] = {{ n_in/2, 0, n-n_in/2, 0, 1, 1, 1 }, 
						 { n_in/2, 0, n-n_in/2, 0, n_in, n, 1 }, 
						 { n_in/2, n_in/2, 0, 0, n_in*n_in, n*n, 1 }},
		in_dimx_101[] = {{ n_in/2, n_in/2, 0, 0, 1, 1, 1 }, 
						 { n_in/2, 0, n-n_in/2, 0, n_in, n, 1 }, 
						 { n_in/2, n_in/2, 0, 0, n_in*n_in, n*n, 1 }},
		in_dimx_011[] = {{ n_in/2, 0, n-n_in/2, 0, 1, 1, 1 }, 
						 { n_in/2, n_in/2, 0, 0, n_in, n, 1 }, 
						 { n_in/2, n_in/2, 0, 0, n_in*n_in, n*n, 1 }},
		in_dimx_111[] = {{ n_in/2, n_in/2, 0, 0, 1, 1, 1 }, 
						 { n_in/2, n_in/2, 0, 0, n_in, n, 1 }, 
						 { n_in/2, n_in/2, 0, 0, n_in*n_in, n*n, 1 }},
		batch_dims_in = {batch_size, 0, 0, 0, n_in*n_in*n_in, n*n*n, 1};
	fftx_plan plans[9];										// intermediate sub-plans
	fftx_plan p;											// top-level plan

	// create zero-initialized rank-dimensional temporary data cube given by padded_dims for zero-padding the input
	tmp1 = fftx_create_zero_temp_complex_b(rank, padded_dims, batch_rank, &batch_dims);
	// tmp2 = fftx_create_zero_temp_complex_b(rank, padded_dims, batch_rank, &batch_dims);

	// split input cube into 8 corners
	plans[0] = fftx_plan_guru_copy_complex_b(rank, in_dimx_000, batch_rank, &batch_dims_in, in, tmp1, MY_FFTX_MODE_SUB);
	plans[1] = fftx_plan_guru_copy_complex_b(rank, in_dimx_100, batch_rank, &batch_dims_in, in, tmp1, MY_FFTX_MODE_SUB);
	plans[2] = fftx_plan_guru_copy_complex_b(rank, in_dimx_010, batch_rank, &batch_dims_in, in, tmp1, MY_FFTX_MODE_SUB);
	plans[3] = fftx_plan_guru_copy_complex_b(rank, in_dimx_110, batch_rank, &batch_dims_in, in, tmp1, MY_FFTX_MODE_SUB);
	plans[4] = fftx_plan_guru_copy_complex_b(rank, in_dimx_001, batch_rank, &batch_dims_in, in, tmp1, MY_FFTX_MODE_SUB);
	plans[5] = fftx_plan_guru_copy_complex_b(rank, in_dimx_101, batch_rank, &batch_dims_in, in, tmp1, MY_FFTX_MODE_SUB);
	plans[6] = fftx_plan_guru_copy_complex_b(rank, in_dimx_011, batch_rank, &batch_dims_in, in, tmp1, MY_FFTX_MODE_SUB);
	plans[7] = fftx_plan_guru_copy_complex_b(rank, in_dimx_111, batch_rank, &batch_dims_in, in, tmp1, MY_FFTX_MODE_SUB);

	// // copy input cube in the middle
	// plans[8] = fftx_plan_guru_copy_complex_b(rank, in_dimx, batch_rank, &batch_dims_in, in, tmp2, MY_FFTX_MODE_SUB);

	// DFT on the padded data
	plans[8] = fftx_plan_guru_dft(rank, padded_dims, batch_rank, &batch_dims, tmp1, out, FFTX_BACKWARD, MY_FFTX_MODE_SUB);

	// create the top level plan. this copies the sub-plan pointers, so they can be lost
	p = fftx_plan_compose(numsubplans, plans, MY_FFTX_MODE_TOP);

	// plan to be used with fftx_execute()
	return p;
}


// destroys the plan and temporaries to clean up 

void planewave_destroy(fftx_plan p) {
    // cleanup
	fftx_destroy_temp_complex(tmp1);
	// fftx_destroy_temp_complex(tmp2);
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
	fftx_complex *in, *out;
	fftx_plan p;
	
	// problem size definition
	int n = 256,
		n_in = 128,
		batch_rank = 1,
		batch_size = 10;
	fftx_iodim dims_in[] = {{ n_in, 1, 1 }, { n_in, 1, 1 }, { n_in, 1, 1 }},
		dims_out[] = {{ n, 1, 1 }, { n, 1, 1 }, { n, 1, 1 }},
		batch_dim_in = {batch_size, 1, 1},
		batch_dim_out = {batch_size, 1, 1};
	fftw_complex *planiodata[2];	// used for persistent calls

	// initialize FFTX in FFTX_MODE_OBSERVE
	fftx_init(MY_FFTX_MODE);

	// allocate input, symbol, and output
planiodata[0] = in = (fftx_complex*)fftx_create_data_complex_b(3, dims_in, batch_rank, &batch_dim_in);

	fftx_complex *out_val;
	fftx_plan p_val;
	planiodata[1] = out_val = (fftx_complex*)fftx_create_data_complex_b(3, dims_out, batch_rank, &batch_dim_out);
	out = (fftx_complex*)fftx_create_data_complex_b(3, dims_out, batch_rank, &batch_dim_out);


	// initialize input
	for(fftx_complex *initin = in; initin < in + batch_size*(n_in*n_in*n_in); initin += (n_in*n_in*n_in))
		for (int i = 0; i < n_in*n_in*n_in; i++) {
			initin[i][0] = 1;
			initin[i][1] = 0.;
		}


	p = planewave_plan(in, out, n, n_in, batch_rank, batch_size);


	fftx_execute(p);


	// cleanup
	fftx_destroy_data_complex(in);
	fftx_destroy_data_complex(out);

	planewave_destroy(p);

	// shut down FFTX
	fftx_shutdown();

	return 0;
}
