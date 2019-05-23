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
 

// input pruning: assume only first n_s elements are non-zero
// output pruning: only the last n_d elements are requested
//
// convolution kernel F[G](k) = 1/(4*pi*||k - un||^2_2 )

#include <stdio.h>
#include "fftx.h"


// flags for FFTX
#define MY_FFTX_MODE		0
#define MY_FFTX_MODE_TOP	(FFTX_ESTIMATE | MY_FFTX_MODE)
#define MY_FFTX_MODE_SUB	(MY_FFTX_MODE_TOP | FFTX_FLAG_SUBPLAN)


// globals for the convolution
// bad style, but good enough for the example
fftx_temp_real tmp1, tmp4;
fftx_temp_complex tmp2, tmp3, tmpSymb;

// pointwise scaling function: complex multiply by symbol
void complex_scaling(fftx_complex *in, fftx_complex *out, fftx_complex *data) {
	FFTX_COMPLEX_TEMP(t);
	FFTX_COMPLEX_MOV(t, in);
	FFTX_COMPLEX_MULT(t, data);
	FFTX_COMPLEX_MOV(out, t);
}

// produces an fftx_plan to perform the whole convolution

fftx_plan hockney_plan(fftx_real *in, fftx_real *out, fftx_complex *symbol, int n, int n_in, int n_out, int n_freq, int batch_rank, int batch_size) {
	int rank = 3,			// 3D = rank 3
		numsubplans = 9;	// need 5 FFTX subplans for pruned convolution + 4 for Symbol symmetry

#ifdef DEFAULT_3D_ORDER
#undef DEFAULT_3D_ORDER
#endif
#define DEFAULT_3D_ORDER FFTX_ORDER_ZYX

	FFTX_CUBE(IN_CUBE, n_in);
	FFTX_CUBE(OUT_CUBE, n_out);
	FFTX_CUBE(PADDED_CUBE, 2*n);
	FFTX_3D_BOX(FREQ_CUBE, 2*n, 2*n, n_freq);
	FFTX_CUBE(SYM, n_freq);

	FFTX_IO_BOX(t_padded_dimx, 3, PADDED_CUBE);
	FFTX_IOD_BOX(freq_dimx, 3, FREQ_CUBE);
	FFTX_IO_BATCH(t_batch_dimx, 1, batch_size, 3, PADDED_CUBE, FREQ_CUBE);
	FFTX_IO_BATCH(t_batch_dimx_inv, 1, batch_size, 3, FREQ_CUBE, PADDED_CUBE);

	fftx_iodim  
		*padded_dims      = iodimx2iodim_vec(rank, t_padded_dimx),
		*freq_dims        = iodimx2iodim_vec(rank, freq_dimx), 
		*batch_dims       = iodimx2iodim_vec(batch_rank, t_batch_dimx), 
		*batch_dims_inv   = iodimx2iodim_vec(batch_rank, t_batch_dimx_inv);

	FFTX_EMBED_BOX_BOX(in_dimx, 3, IN_CUBE, PADDED_CUBE, FFTX_VEC(0, 0, 0));
	FFTX_EXTRACT_BOX_BOX(out_dimx, 3, PADDED_CUBE, OUT_CUBE, FFTX_VEC(2*n-n_out, 2*n-n_out, 2*n-n_out));
	FFTX_IO_BATCH(batch_dimx, 1, batch_size, 3, FREQ_CUBE, FREQ_CUBE);
	FFTX_IO_BATCH(batch_dims_in, 1, batch_size, 3, IN_CUBE, PADDED_CUBE);
	FFTX_IO_BATCH(batch_dims_out, 1, batch_size, 3, PADDED_CUBE, OUT_CUBE);

	FFTX_EMBED_BOX_BOX(sym_nn, 3, SYM, FREQ_CUBE, FFTX_VEC(0, 0, 0));
	FFTX_EMBED_BOX_BOX_EXTDEF(sym_ns, 3, SYM, FFTX_VEC(n-1, n_freq, n_freq), FFTX_VEC(n-1, 0, 0), FFTX_VEC(-1, 1, 1),
		FREQ_CUBE, FFTX_VEC(n_freq, 0, 0), FFTX_VEC(1, 1, 1));
	FFTX_EMBED_BOX_BOX_EXTDEF(sym_sn, 3, SYM, FFTX_VEC(n_freq, n-1, n_freq), FFTX_VEC(0, n-1, 0), FFTX_VEC(1, -1, 1),
		FREQ_CUBE, FFTX_VEC(0, n_freq, 0), FFTX_VEC(1, 1, 1));
	FFTX_EMBED_BOX_BOX_EXTDEF(sym_ss, 3, SYM, FFTX_VEC(n-1, n-1, n_freq), FFTX_VEC(n-1, n-1, 0), FFTX_VEC(-1, -1, 1), 
		FREQ_CUBE, FFTX_VEC(n_freq, n_freq, 0), FFTX_VEC(1, 1, 1));

	fftx_plan plans[9];										// intermediate sub-plans
	fftx_plan p;											// top-level plan

	// create zero-initialized rank-dimensional temporary data cube given by padded_dims for zero-padding the input
	tmp1 = fftx_create_zero_temp_real_b(rank, padded_dims, batch_rank, batch_dims);

	// copy a rank-dimensional data cube given by in_dims into a contiguous rank-dimensional zero-initialized temporary
	plans[0] = fftx_plan_guru_copy_real_b(rank, in_dimx, batch_rank, batch_dims_in, in, tmp1, MY_FFTX_MODE_SUB);
	
	// RDFT on the padded data
	tmp2 = fftx_create_temp_complex_b(rank, freq_dims, batch_rank, batch_dims);
	plans[1] = fftx_plan_guru_dft_r2c(rank, padded_dims, batch_rank, batch_dims, tmp1, tmp2, MY_FFTX_MODE_SUB);

	// Symbol symmetry
	tmpSymb = fftx_create_zero_temp_complex(rank, freq_dims);
	plans[2] = fftx_plan_guru_copy_complex(rank, sym_nn, symbol, tmpSymb, MY_FFTX_MODE_SUB);
	plans[3] = fftx_plan_guru_copy_complex(rank, sym_ns, symbol, tmpSymb, MY_FFTX_MODE_SUB);
	plans[4] = fftx_plan_guru_copy_complex(rank, sym_sn, symbol, tmpSymb, MY_FFTX_MODE_SUB);
	plans[5] = fftx_plan_guru_copy_complex(rank, sym_ss, symbol, tmpSymb, MY_FFTX_MODE_SUB);

	// pointwise operation
	tmp3 = fftx_create_temp_complex_b(rank, freq_dims, batch_rank, batch_dims);
	plans[6] = fftx_plan_guru_pointwise_c2c(rank, freq_dimx, batch_rank, batch_dimx,
		tmp2, tmp3, tmpSymb, (fftx_callback)complex_scaling, MY_FFTX_MODE_SUB | FFTX_PW_POINTWISE);

	// iRDFT on the scaled data
	tmp4 = fftx_create_temp_real_b(rank, padded_dims, batch_rank, batch_dims);
	plans[7] = fftx_plan_guru_dft_c2r(rank, padded_dims, batch_rank, batch_dims_inv, tmp3, tmp4, MY_FFTX_MODE_SUB);

	// copy out the rank-dimensional data cube given by out_dims of interest
	plans[8] = fftx_plan_guru_copy_real_b(rank, out_dimx, batch_rank, batch_dims_out, tmp4, out, MY_FFTX_MODE_SUB);

	// create the top level plan. this copies the sub-plan pointers, so they can be lost
	p = fftx_plan_compose(numsubplans, plans, MY_FFTX_MODE_TOP);

	// plan to be used with fftx_execute()
	return p;
}


// destroys the plan and temporaries to clean up 

void hockney_destroy(fftx_plan p) {
    // cleanup
	fftx_destroy_temp_real(tmp1);
	fftx_destroy_temp_complex(tmp2);
	fftx_destroy_temp_complex(tmp3);
	fftx_destroy_temp_real(tmp4);
	fftx_destroy_temp_complex(tmpSymb);
	fftx_destroy_plan_recursive(p);
}


// main function
//
// parameterize problem size at the top
// temporaries and sub-plans are protected and need to be
// un-protected if they need to be inspected or executed separately

int main(void) {
	// input, output, FFTX plan
	fftx_real *in,
		*out;
	fftx_complex *symbol;
	fftx_plan p;
	const double pi = 3.14159265358979323846;
	
	// problem size definition
	int n = 16, //130,
		n_in = 4, //33,
		n_out = 8, //96,
		n_freq = n/2 + 1,
		batch_rank = 1,
		batch_size = 2; //60;
	fftx_iodim dims_in[] = {{ n_in, 1, 1 }, { n_in, 1, 1 }, { n_in, 1, 1 }},
		dims_out[] = {{ n_out, 1, 1 }, { n_out, 1, 1 }, { n_out, 1, 1 }},
		dims_sym[] = {{ n_freq, 1, 1 }, { n_freq, 1, 1 }, { n_freq, 1, 1 }},
		batch_dim_in = {batch_size, 1, 1},
		batch_dim_out = {batch_size, 1, 1};
	double *planiodata[3];	// used for persistent calls

	// initialize FFTX in FFTX_MODE_OBSERVE
	fftx_init(MY_FFTX_MODE);

	// allocate input, symbol, and output
	planiodata[0] = in = (fftx_real*)fftx_create_data_real_b(3, dims_in, batch_rank, &batch_dim_in);
	planiodata[1] = out = (fftx_real*)fftx_create_data_real_b(3, dims_out, batch_rank, &batch_dim_out);

	symbol = (fftx_complex*)fftx_create_data_complex(3, dims_sym);
	planiodata[2] = (double*)symbol;

	// initialize input
	for(fftx_real *initin = in; initin < in + batch_size*(n_in*n_in*n_in); initin += (n_in*n_in*n_in))
		for (int i = 0; i < n_in*n_in*n_in; i++)
			initin[i] = i + 1;

	// initialize symbol
	for (int i = 0; i < n_freq; i++) {
		for (int j = 0; j < n_freq; j++) {
			for (int k = 0; k < n_freq; k++) {
				if(i < n/2 || j < n/2 || k < n/2)
					symbol[i*n_freq*n_freq + j*n_freq + k][0] = 1./(4.*pi*((n/2-i)*(n/2-i)+(n/2-j)*(n/2-j)+(n/2-k)*(n/2-k)));
				else
					symbol[i*n_freq*n_freq + j*n_freq + k][0] = 0.0;
				symbol[i*n_freq*n_freq + j*n_freq + k][1] = 0.0;
			}
		}
	}

	// build FFTX descriptor for trivial convolution
	p = hockney_plan(in, out, symbol, n/2, n_in, n_out, n_freq, batch_rank, batch_size);

	// execute the trivial convolution
	fftx_execute(p);
	   	 
	// cleanup
	fftx_destroy_data_real(in);
	fftx_destroy_data_real(out);
	fftx_destroy_data_complex(symbol);
	hockney_destroy(p);

	// shut down FFTX
	fftx_shutdown();

	return 0;
}
