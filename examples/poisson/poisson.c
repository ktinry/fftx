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

// Poisson example

#include <stdio.h>
#include <stdlib.h>
#include "fftx.h"

#define MY_FFTX_MODE		FFTX_MODE_OBSERVE
#define MY_FFTX_MODE_TOP	(FFTX_ESTIMATE | FFTX_MODE_OBSERVE)
#define MY_FFTX_MODE_SUB	(MY_FFTX_MODE_TOP | FFTX_FLAG_SUBPLAN)


// not all compilers support VLAs
#define MY_RANK_NUM 3

// globals for the convolution
// bad style, but good enough for the example
fftx_temp_complex tmp1;

// pointwise scaling function: complex multiply by symbol
void complex_scaling(fftx_complex *in, fftx_complex *out, fftx_complex *data) {
  FFTX_COMPLEX_TEMP(t);
  FFTX_COMPLEX_MOV(t, in);
  FFTX_COMPLEX_MULT(t, data);
  FFTX_COMPLEX_MOV(out, t);
}



// produces an fftx_plan to perform the whole Poisson operation
fftx_plan poisson_plan(fftx_complex *in, fftx_complex *out, fftx_complex *symbol, int rank, int *sizes) {
  int numsubplans = 3;

  // FFTX iodim definitions for 3D
  fftx_iodim forward[MY_RANK_NUM], symbol_dims[MY_RANK_NUM], inverse[MY_RANK_NUM];

  int is = 1;
  int os = 1;

  for(int i = 0; i != rank; ++i) {
    symbol_dims[i].n = sizes[i];
    symbol_dims[i].is = is;
    symbol_dims[i].os = os;

    forward[i].n = sizes[i];
    forward[i].is = is;
    forward[i].os = os;

    inverse[i].n = sizes[i];
    inverse[i].is = is;
    inverse[i].os = os;
    
    is *= sizes[i];
    os *= sizes[i];
  }

  fftx_iodimx batch;
  batch.n = 1;
  batch.is = is;
  batch.os = os;

  printf("%d %d %d\n", (forward[1]).n, (forward[1]).is, (forward[1]).os);
  
  fftx_plan plans[3], p;
  
  // create zero-initialized rank-dimensional temporary data cube given by padded_dims for zero-padding the input
  tmp1 = (fftx_temp_complex)fftx_create_temp_complex_b(rank, symbol_dims, 1, (fftx_iodim *)&batch);

  // forward 3D FFT
  plans[0] = fftx_plan_guru_dft(rank, forward, 1, (fftx_iodim *)&batch, in, tmp1, FFTX_FORWARD, MY_FFTX_MODE_SUB);

  // pointwise operation
  plans[1] = fftx_plan_guru_pointwise_c2c(rank, (fftx_iodimx *)symbol_dims,  1, NULL, tmp1, tmp1, symbol, (fftx_callback)complex_scaling, MY_FFTX_MODE_SUB | FFTX_PW_POINTWISE);

  // inverse 3D FFT
  plans[2] = fftx_plan_guru_dft(rank, inverse, 1, (fftx_iodim *)&batch, tmp1, out, FFTX_BACKWARD, MY_FFTX_MODE_SUB);
  
  // create the top level plan. this copies the sub-plan pointers, so they can be lost
  p = fftx_plan_compose(numsubplans, plans, MY_FFTX_MODE_TOP);

  // plan to be used with fftx_execute()
  return p;
}


// destroys the plan and temporaries to clean up 

void poisson_destroy(fftx_plan p) {
  // cleanup
  fftx_destroy_temp_complex(tmp1);
  fftx_destroy_plan_recursive(p);
}

// #endif

// main function
//
int main(void) {
  // input, output, FFTX plan
  fftx_complex *in, *out;
  fftx_complex *symbol;
  fftx_plan p;

  const double pi = 3.14159265358979323846;
  
  // problem size definition
  int n = 32, m = 16, k = 32;

  in = (fftx_complex*)malloc(n * m * k * sizeof(fftx_complex));
  out = (fftx_complex*)malloc(n * m * k * sizeof(fftx_complex));
  symbol = (fftx_complex*)malloc(n * m * k * sizeof(fftx_complex));
  
  // initialize FFTX in FFTX_MODE_OBSERVE
  fftx_init(MY_FFTX_MODE);

  // initialize input
  double *input = (double *) in;
  double *output = (double *) out;
  for(int i = 0; i != n; ++i) {
    for(int j = 0; j != m; ++j) {
      for(int l = 0; l != k; ++l) {
	int offset = l + k * j + k * m * i;

	*(input + 2 * offset + 0) = (offset + 1) * 1.0;
	*(input + 2 * offset + 1) = (offset + 1) * 1.0;

	*(output + 2 * offset + 0) = 0.0;
	*(output + 2 * offset + 1) = 0.0;
      }
    }
  }
  
  // initialize symbol
  double *pSymbol = (double*) symbol;
  for (int i = 0; i != n; ++i) {
    for (int j = 0; j != m; ++j) {
      for (int l = 0; l != k; ++l) {
	int offset = l + k * j + k * m * i;
	
	if((i == 0) && (j == 0) && (l == 0)) {
	  *(pSymbol + 2 * offset + 0) = 1.0;
	  *(pSymbol + 2 * offset + 1) = 0.0;
	} else {
	  *(pSymbol + 2 * offset + 0) = (1 / ((k * k + j * j + i * i) * 1.0));
	  *(pSymbol + 2 * offset + 1) = 0.0;
	}
      }
    }
  }

  int rank = MY_RANK_NUM;
  int sizes[MY_RANK_NUM];

  sizes[0] = n;
  sizes[1] = m;
  sizes[2] = k;
  
  // build FFTX descriptor for trivial convolution
  p = poisson_plan(in, out, symbol, rank, sizes);

  // execute the trivial convolution
  fftx_execute(p);

  // cleanup
  free(in);
  free(out);
  free(symbol);

  poisson_destroy(p);

  // shut down FFTX
  fftx_shutdown();

  return 0;
}
