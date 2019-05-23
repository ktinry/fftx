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
#include <string.h>
#include "fftx.h"
#include <limits.h>
#include <complex.h>


#include <math.h>


// persistent label for top level plan
#define MY_PLAN_LABEL	0x1234

#if defined(USE_PERSISTENT_PLAN) && !defined(VALIDATE)
#define MY_FFTX_MODE		FFTX_HIGH_PERFORMANCE
#else
// flags for FFTX
#define MY_FFTX_MODE		FFTX_HIGH_PERFORMANCE // was: FFTX_MODE_OBSERVE
#define MY_FFTX_MODE_TOP	(FFTX_ESTIMATE | FFTX_MODE_OBSERVE)
#define MY_FFTX_MODE_SUB	(MY_FFTX_MODE_TOP | FFTX_FLAG_SUBPLAN)

#define NUMSUBPLANS 3

#define WRITE_INITIAL 0
#define WRITE_INITIAL_FOURIER 0
#define WRITE_UPDATED_FOURIER 0
#define WRITE_INVERSE_FOURIER 0
#define WRITE_BEFORE_STEP 0
#define WRITE_AFTER_STEP 0

// New implementation has rho before J, to make indexing variables
// consistent between input and output, so RHO_BEFORE_J == 1.
// Earlier implementation had RHO_BEFORE_J == 0.
#define RHO_BEFORE_J 1

#define DIM 3

#define X 0
#define Y 1
#define Z 2

// If you omit the parentheses around these, then trouble may result!
#define EBASE 0
#define EX (EBASE+X)
#define EY (EBASE+Y)
#define EZ (EBASE+Z)
#define BBASE (EBASE+DIM) // B after E's DIM components
#define BX (BBASE+X)
#define BY (BBASE+Y)
#define BZ (BBASE+Z)
#if RHO_BEFORE_J // new, RHO before J
#define RHO (BBASE+DIM) // RHO after B's DIM components
#define JBASE (RHO+1) // J after RHO's 1 component
#define JX (JBASE+X)
#define JY (JBASE+Y)
#define JZ (JBASE+Z)
#define NCOMPS (JBASE+DIM) // after J's DIM components
#else // old, RHO after J
#define JBASE (BBASE+DIM) // J after B's DIM components
#define JX (JBASE+X)
#define JY (JBASE+Y)
#define JZ (JBASE+Z)
#define RHO (JBASE+DIM) // RHO after J's DIM components
#define NCOMPS (RHO+1) // after RHO's 1 component
#endif

 const double clight = 299792458.e8;
//const double clight = 1.;

int verbosity = 0;

// NOTE that you need to set (*var)[0] and (*var)[1], not var[0] and var[1].

#define SET_REAL(var, re) (*var)[0] = re; (*var)[1] = 0.;

#define SET_COMPLEX(var, re, im) (*var)[0] = re; (*var)[1] = im;

#define DEFINE_DOUBLE_COMPLEX(dc, fc) double complex dc = (fc)[0] + (fc)[1]*I;

// globals for the convolution
// bad style, but good enough for the example
fftx_temp_real physicalInitial; // WAS tmp1
fftx_temp_complex fourierInitial;
fftx_temp_complex fourierUpdated;
fftx_temp_real physicalUpdated; // WAS tmp4

void fillk1D(fftx_complex* a_k1d,
             double* a_kamp2,
             int a_length)
{
  double my_PI = 4.*atan(1.);
  double my_PI_2 = 2.*atan(1.);
  for (int i = 0; i < a_length; i++)
    {
      int kint = i;
      if (i >= (a_length+1)/2)
        {
          kint -= a_length;
        }

      int mint = i;
      if (i >= a_length/2 + 1)
        {
          mint += a_length;
        }

      double kamp = kint * my_PI_2;
      double th = mint * my_PI / (a_length * 1.);
      (a_k1d[i])[0] = kamp * cos(th);
      (a_k1d[i])[1] = kamp * sin(th);
      a_kamp2[i] = kamp * kamp;
    }
}

void fillkArrays(fftx_complex* a_ikf,
                 fftx_complex* a_ikb,
                 double* a_kabs,
                 int a_rank,
                 int a_length0,
                 int a_length1,
                 int a_length2,
                 fftx_complex* a_k1D,
                 double* a_kamp2)
{
  fftx_complex* ikfpt = a_ikf;
  fftx_complex* ikbpt = a_ikb;
  double* kabspt = a_kabs;
  FFTX_COMPLEX_VAL(ii, 0., 1.);
  for (int i = 0; i < a_length0; i++)
    for (int j = 0; j < a_length1; j++)
      for (int k = 0; k < a_length2; k++)
        {
          // i*(a + i*b) == -b + i*a
          // i*(a - i*b) == b + i*a

          // X
          (*ikbpt)[0] = -(a_k1D[i])[1];
          (*ikbpt)[1] = (a_k1D[i])[0];
          ikbpt++;
          (*ikfpt)[0] = (a_k1D[i])[1];
          (*ikfpt)[1] = (a_k1D[i])[0];
          ikfpt++;

          // Y
          (*ikbpt)[0] = -(a_k1D[j])[1];
          (*ikbpt)[1] = (a_k1D[j])[0];
          ikbpt++;
          (*ikfpt)[0] = (a_k1D[j])[1];
          (*ikfpt)[1] = (a_k1D[j])[0];
          ikfpt++;

          // Z
          (*ikbpt)[0] = -(a_k1D[k])[1];
          (*ikbpt)[1] = (a_k1D[k])[0];
          ikbpt++;
          (*ikfpt)[0] = (a_k1D[k])[1];
          (*ikfpt)[1] = (a_k1D[k])[0];
          ikfpt++;

          double val2 = a_kamp2[i] + a_kamp2[j] + a_kamp2[k];
          *kabspt = sqrt(val2);
          kabspt++;
        }
}

// produces an fftx_plan to perform the whole convolution
fftx_plan warpx_plan(fftx_real *physicalInitial,
                     fftx_real *physicalUpdated,
                     fftx_complex *data, 
                     int rank,
                     int n,
                     int iv,
                     int ov,
                     int batch)
{
  int nReduced = n/2 + 1; // rounded down

  size_t nAll = n;
  nAll *= n;
  nAll *= n;
  
  size_t nAllReduced = nReduced;
  nAllReduced *= n;
  nAllReduced *= n;
  
  // struct fftx_iodim components: {size, input stride, output stride}
  // This is the SAME as fftw_iodim

  // FFTX iodim definitions for 3D + pruning:
  // struct of {loop length n, input stride is, output stride os}
  // as in fftw_iodim.
  // A row-major multidimensional array with dimensions n[rank] corresponds
  // to dims[i].n = n[i], and the recurrence dims[i].is = n[i+1] * dims[i+1].is
  // (similarly for os).
  // The stride of the last (i=rank-1) dimension is the overall stride of
  // the array. e.g. to be equivalent to the advanced complex-DFT interface,
  // you would have dims[rank-1].is = istride and dims[rank-1].os = ostride. 
  fftx_iodim
    // dft_dims specifies innermost loops
    // dft_dims[] = {{ n, iv, iv }, // innermost loop: n iterations, stride iv in both input and output
    // { n, n*iv, n*iv }, // next inner: loop size n, stride = previous loop size * previous stride
    // { n, n*n*iv, n*n*iv }}, // loop size n, stride = previous loop size * previous stride
    //    dft_dims[] = {{ n, n*n*iv, n*n*iv }, // outermost loop
    //                  { n, n*iv, n*iv },
    //                  { n, iv, iv }},  // innermost loop: length, input stride, output stride
    dft_dims[] = {{ n, n*n*iv, n*nReduced*iv }, // outermost loop
                  { n, n*iv, nReduced*iv },
                  { n, iv, iv }},  // innermost loop: length, input stride, output stride
    // dft_howmany specifies outermost loops
    dft_howmany[] = {{ batch, nAll*iv, nAll*iv }, //Interleaved batch // outermost; batch many times with stride nAll*iv in both input and output
                     { iv, 1, 1 }}, // loop over iv input variables
    dft_r2c_howmany[] = {{ batch, nAll*iv, nAllReduced*iv }, //Interleaved batch // outermost; batch many times with input stride nAll*iv, and output stride nAllReduced*iv
                         { iv, 1, 1 }}, // loop over iv input variables
    // dft_r2c_dims is same as dft_dims except for last dimension
    //    dft_r2c_dims[] = {{1, 1, 1}, // innermost loop
    //                      {n, 1, 1}, // next inner
    //                      {nReduced, 1, 1}}, // loop nReduced times with stride 1 in both input and output
    // dft_r2c_dims[] = {{nReduced, iv, iv}, // innermost loop
    //                   {n, nReduced*iv, nReduced*iv}, // next inner: size n, stride = previous loop size * previous stride
    //                   {n, n*nReduced*iv, n*nReduced*iv}}, // loop size n, stride = previous loop size * previous stride
    //    dft_r2c_dims[] = {{n, iv, iv}, // innermost loop
    //                      {n, n*iv, n*iv}, // next inner: size n, stride = previous loop size * previous stride
    //                      {nReduced, n*n*iv, n*n*iv}}, // loop size n, stride = previous loop size * previous stride
    dft_r2c_dims[] = {{ n, n*nReduced*iv, n*nReduced*iv }, // outermost loop
                      { n, nReduced*iv, nReduced*iv },
                      { nReduced, iv, iv }}, // innermost loop
    //    idft_dims[] = {{ n, ov, ov },
    //                   { n, n*ov, n*ov }, // loop size n, stride = previous loop size * previous stride
    //                   { n, n*n*ov, n*n*ov }}, // loop size n, stride = previous loop size * previous stride
      idft_dims[] = {{ n, n*nReduced*ov, n*n*ov },
                     { n, nReduced*ov, n*ov },
                     { n, ov, ov }},
      idft_c2r_dims[] = {{ n, n*nReduced*ov, n*nReduced*ov },
                         { n, nReduced*ov, nReduced*ov },
                         { nReduced, ov, ov }},
      idft_howmany[] = {{ batch, nAll*ov, nAll*ov }, //Interleaved batch
                        { ov, 1, 1 }},
    idft_c2r_howmany[] = {{ batch, nAllReduced*ov, nAll*ov }, //Interleaved batch // outermost; batch many times with input stride nAllReduced*iv, and output stride nAll*iv
                          { ov, 1, 1 }}; // loop over iv input variables
                      

  int in_rank = 1,
    out_rank = 1,
    data_rank = 2; 

  // Inherits from FFTW guru interface:
  // fftx_iodimx is fftx_iodim (same as fftw_iodim) with OFFSETS.
  // struct fftx_iodimx components: {size,
  //                                 input offset, output offset, data offset (or 0),
  //                                 input stride, output stride, data stride (or 0)}
  fftx_iodimx //      size     offsets         strides
    outer_tc_dimx[] = {{n,  0, 0, 0,  iv, ov, iv*ov},
                       {n,  0, 0, 0,  n*iv, n*ov, n*iv*ov},
                       {n,  0, 0, 0,  n*n*iv, n*n*ov, n*n*iv*ov}},
    outer_tc_r2c_dimx[] = {{n,  0, 0, 0,  n*nReduced*iv, n*nReduced*ov, n*nReduced*iv*ov},
                           {n,  0, 0, 0,  nReduced*iv, nReduced*ov, nReduced*iv*ov},
                           {nReduced,  0, 0, 0,  iv, ov, iv*ov}},
    outer_tc_howmany = {batch,  0, 0, 0,  nAll*iv, nAll*ov, 0},
    outer_tc_r2c_howmany = {batch,  0, 0, 0,  nAllReduced*iv, nAllReduced*ov, 1},
    in_tc_dimx[] = {{iv,  0, 0, 0,  1, 0, 0}},
    out_tc_dimx[] = {{ov,  0, 0, 0,  0, 1, 0}},
    data_tc_dimx[] = {{ov,  0, 0, 0,  0, 0, iv},
                      {iv,  0, 0, 0,  0, 0, 1}};

  fftx_plan plans[NUMSUBPLANS];
  fftx_plan p; // top-level plan

  // create zero-initialized rank-dimensional temporary data cube given by padded_dims for zero-padding the input
  // "For an out-of-place transform, the real data is simply an array with
  // physical dimensions n[0] x n[1] x n[2] x ... x n[d-1] in row-major order."
  // physicalInitial = fftx_create_zero_temp_real_b(rank, dft_dims, 2, dft_howmany);
  // physicalUpdated = fftx_create_temp_real_b(rank, idft_dims, 2, idft_howmany);

  // fourierInitial = fftx_create_zero_temp_complex_b(rank, padded_dims, batch_rank, &batch_dims);

  int NUMPLAN = 0;

  // DFT on the padded data
  // TIP: See Hockney for another example of r2c.
  // fourierInitial = fftx_create_temp_complex_b(rank, dft_dims, 2, dft_howmany); // change these for r2c
  // See documentation for FFTW guru real-data DFTs.
  // The last dimension of dft_dims is interpreted specially:
  // last dimension of the real input array has that size, but
  // last dimension of the complex output array has that size halved plus 1.
  // Hence tmp1 has dimensions dft_dims.n[0], dft_dims.n[1], dft_dims.n[2];
  // and fourierInitial has dimensions dft_dims.n[0], dft_dims.n[1], dft_dims.n[2]/2+1.
  fourierInitial = fftx_create_temp_complex_b(rank, dft_r2c_dims, 2, dft_r2c_howmany);
  plans[NUMPLAN++] =
    fftx_plan_guru_dft_r2c(rank, // rank 3 for 3D FFT
                           dft_dims, // dimensions: same for r2c as for c2r
                           2, // rank of dft_howmany
                           dft_r2c_howmany, // same for r2c as for c2r
                           physicalInitial, // input, WAS tmp1
                           fourierInitial, // output
                           MY_FFTX_MODE_SUB //flags
                           );

  // same size as fourierInitial, the result of r2c
  // fourierUpdated = fftx_create_zero_temp_complex_b(rank, idft_dims, 2, idft_howmany);
  fourierUpdated = fftx_create_zero_temp_complex_b(rank, idft_c2r_dims, 2, idft_c2r_howmany);
  plans[NUMPLAN++] =
    fftx_plan_tc_c2c(rank, // rank 3
                     outer_tc_r2c_dimx, // WAS outer_tc_dimx,
                     1, // rank of next arg, howmany
                     &outer_tc_r2c_howmany, // WAS &outer_tc_howmany,
                     fourierInitial, // input
                     in_rank,
                     in_tc_dimx,
                     fourierUpdated, // output
                     out_rank,
                     out_tc_dimx, 
                     data,
                     data_rank,
                     data_tc_dimx,
                     MY_FFTX_MODE_SUB | FFTX_TC);

  // 
  // iDFT on the padded data
  // tmp4 = fftx_create_temp_complex_b(rank, idft_dims, 2, idft_howmany);
  // plans[NUMPLAN++] = fftx_plan_guru_dft(rank, idft_dims, 2, idft_howmany, fourierUpdated, tmp4, FFTX_BACKWARD, MY_FFTX_MODE_SUB);
  // tmp4 = fftx_create_temp_real_b(rank, idft_dims, 2, idft_howmany);
  // physicalUpdated = fftx_create_temp_real_b(rank, idft_dims, 2, idft_howmany);
  plans[NUMPLAN++] =
    fftx_plan_guru_dft_c2r(rank, // rank 3 for 3D FFT
                           idft_dims, // dimensions: same for c2r as for r2c
                           2, // rank of idft_howmany
                           idft_c2r_howmany,
                           fourierUpdated, // input
                           physicalUpdated, // output, WAS tmp4
                           MY_FFTX_MODE_SUB // flags
                           );
    
  if (NUMPLAN != NUMSUBPLANS)
    {
      printf("Should have %d plans but you have %d plans!\n", NUMSUBPLANS, NUMPLAN);
      abort();
    }
  
  // create the top level plan. this copies the sub-plan pointers, so they can be lost
  p = fftx_plan_compose(NUMSUBPLANS, plans, MY_FFTX_MODE_TOP);

  // plan to be used with fftx_execute()
  return p;
}


// destroys the plan and temporaries to clean up 

void warpx_destroy(fftx_plan p)
{
  // cleanup
  fftx_destroy_temp_real(physicalInitial);
  fftx_destroy_temp_complex(fourierInitial);
  fftx_destroy_temp_complex(fourierUpdated);
  fftx_destroy_temp_real(physicalUpdated);
  fftx_destroy_plan_recursive(p);
}

#endif

void parseCommandLine(int* a_size1D, int* a_nsteps, double* a_dt,
                      int argc, char* argv[])
{
  *a_size1D = -1;
  *a_nsteps = -1;
  *a_dt = -1.0;

  for (int iarg = 0; iarg < argc; iarg++)
  {
      if (strcmp(argv[iarg], "-l") == 0)
	  {
          *a_size1D = atoi(argv[iarg+1]);
	  }
      else if (strcmp(argv[iarg], "-s") == 0)
	  {
          *a_nsteps = atoi(argv[iarg+1]);
	  }
	  else if (strcmp(argv[iarg], "-t") == 0)
	  {
		  *a_dt = atof(argv[iarg+1]);
	  }
      else if (strcmp(argv[iarg], "-v") == 0)
	  {
          verbosity = atoi(argv[iarg+1]);
	  }
	  else if (strcmp(argv[iarg], "-h") == 0)
	  {
		  printf("Usage: %s -l size1D -s nsteps -t dt -h [-v verbosity]\n", argv[0]);
		  printf("\n");
		  printf("size1D is 1D length of domain\n");
		  printf("nsteps is number of steps\n");
		  printf("verbosity (optional) is 0 for minimal output, 2 for the most\n");
		  exit(-1);
	  }
  }

  if ((*a_size1D < 0) || (*a_nsteps < 0) || (*a_dt < 0.0) || (verbosity < 0))
  {
	  if (*a_size1D < 0) *a_size1D = 15; 
	  if (*a_nsteps < 0) *a_nsteps = 20;
	  if (*a_dt < 0.0) *a_dt = 10.0;
	  if (verbosity < 0) verbosity = 0;
	  printf("%s: defaulting values: size1D = %d, nsteps = %d, dt = %.2f, verbosity = %d\n",
			 argv[0], *a_size1D, *a_nsteps, *a_dt, verbosity);
	  return;
  }
}

// main function
//
// parameterize problem size at the top
// prints input, output and temporaries in FFTX_MODE_OBSERVE mode
// temporaries and sub-plans are protected and need to be
// un-protected if they need to be inspected or executed separately

int main(int argc, char* argv[])
{
  // input, output, FFTX plan
  fftx_complex *in, *out, *data;
  fftx_plan p;
  fftx_plan_label h = MY_PLAN_LABEL;
  
  int size1D, nsteps;
  double dt;
  parseCommandLine(&size1D, &nsteps, &dt, argc, argv);

  // problem size definition
  int batch = 1, // 30,
    rank = 3,
    n = size1D, // 30,
    ns = n, // 16,
    nd = n, // 16,
    iv = NCOMPS, // 6,
    ov = DIM + DIM + 1; // 3;

  int nReduced = n/2 + 1; // rounded down
  fftx_iodim 
    dims_in[] = {{ ns, 1, 1 },
                 { ns, 1, 1 },
                 { ns, 1, 1 }},
    dims_out[] = {{ nd, 1, 1 },
                  { nd, 1, 1 },
                  { nd, 1, 1 }},
    dims_data[] = {{ n*iv*ov, 1, 1 },
                   { n, 1, 1 },
                   { nReduced, 1, 1 }},
    in_howmany[] = {{iv, 1, 1},
                    {batch, 1, 1}},
    out_howmany[] = {{ov, 1, 1},
                     {batch, 1, 1}},
    data_howmany = {1, 0, 0};

  fftx_complex * planiodata[3]; // used for persistent calls: in, out, data

  // initialize FFTX in FFTX_MODE_OBSERVE
  fftx_init(MY_FFTX_MODE);

  
  // allocate input, symbol, and output
  planiodata[0] = in = (fftx_real*)fftx_create_data_real_b(rank, dims_in, 2, in_howmany);


  printf("defining out\n");
  fftx_plan p_val;
  planiodata[1] = (fftx_real*)fftx_create_data_real_b(rank, dims_out, 2, out_howmany);
  out = (fftx_real *)fftx_create_data_complex_b(rank, dims_out, 2, out_howmany);
  planiodata[2] = data = (fftx_real *)fftx_create_data_complex_b(rank, dims_data, 1, &data_howmany);

  double dx = 2e-5 / (double) (size1D);
  double scaling = dx * dx * dx;
  /*
    initialize input
  */
  if (verbosity >= 1) 
    {
      printf("initializing input\n");
    }
  //  int j = 1;
  int iThis = 1, jThis = 2, kThis = 3;
  printf("iv*ns*ns*ns = %d\n", iv*ns*ns*ns);
  {
    fftx_real* initin = in;
    double multer[3] = {3., 5., 7.};
    // Holdover from when I had rho after J.  Bogus data anyway.
#if RHO_BEFORE_J
    // new implementation with rho before J
    double starter[NCOMPS] = {2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 11.5, 8.5, 9.5, 10.5};
#else
    // earlier implementation with rho after J
    double starter[NCOMPS] = {2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5};
#endif
    // for (fftx_real* batchin = in; batchin < in + batch*iv*ns*ns*ns; batchin+=iv*ns*ns*ns)
    for (int countBatch = 0; countBatch < batch; countBatch++)
      {
        printf("countBatch = %d\n", countBatch);
        // Remember that this version uses C ORDER, and Proto uses FORTRAN ORDER.
        for (int i = 0; i < ns; i++)
          for (int j = 0; j < ns; j++)
            for (int k = 0; k < ns; k++)
              {
                double x = i*dx;
                double y = j*dx;
                double z = k*dx;
                double dist2 = multer[0]*x*x + multer[1]*y*y + multer[2]*z*z;
                for (int comp = 0; comp < iv; comp++)
                  {
                    *initin = sqrt(starter[comp] + dist2);
                    if ((iThis == i) && (jThis == j) && (kThis == k))
                      {
                        printf("init(%d,%d,%d)[%d] = %f\n", i, j, k, comp, *initin);
                      }
                    initin++;
                  }
              }
      }
  }

  
  /*
    initialize data: entries in the tensor contraction matrix
  */
  if (verbosity >= 1) 
    {
      printf("initializing k\n");
    }
  fftx_complex* k1D = (fftx_complex*) malloc(size1D * sizeof(fftx_complex));
  double* kamp2 = (double*) malloc(size1D * sizeof(double));
  if (verbosity >= 1) 
    {
      printf("fillk1D\n");
    }
  fillk1D(k1D, kamp2, size1D);

  size_t nAll = 1;
  for (int dir = 0; dir < rank; dir++)
    {
      nAll *= size1D;
    }
  size_t nAllReduced = nReduced;
  nAllReduced *= size1D;
  nAllReduced *= size1D;
  // fftx_complex* ikfAll = (fftx_complex*) malloc(nAll*(rank*sizeof(fftx_complex)));
  fftx_complex* ikfAll = (fftx_complex*) malloc(nAllReduced*(rank*sizeof(fftx_complex)));
  // fftx_complex* ikbAll = (fftx_complex*) malloc(nAll*(rank*sizeof(fftx_complex)));
  fftx_complex* ikbAll = (fftx_complex*) malloc(nAllReduced*(rank*sizeof(fftx_complex)));
  // double* kabsAll = (double*) malloc(nAll*sizeof(double));
  double* kabsAll = (double*) malloc(nAllReduced*sizeof(double));

  fillkArrays(ikfAll, ikbAll, kabsAll, rank, size1D, size1D, nReduced, k1D, kamp2);

  free(k1D);
  free(kamp2);

  if (verbosity >= 1) 
    {
      printf("initializing data\n");
    }
  {
    double cdt = clight * dt;
    fftx_complex* dataPtr = (fftx_complex *) data;
    double* kabsd = kabsAll;
    fftx_complex* ikfd = ikfAll;
    fftx_complex* ikbd = ikbAll;
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        for (int k = 0; k < nReduced; k++)
          {
            double k1 = *kabsd;
            double theta = k1 * cdt;
            double COS = cos(theta);
            double A1, A2, A3, A4, A5;
            A2 = COS;
            if (fabs(theta) < 1.e-5)
              { // These approximations O(theta^4), better numerically for small theta.
                double cdt2 = cdt * cdt;
                double theta2 = theta * theta;
                A1 = cdt * (1. - theta2/6.);
                A3 = cdt2 * (0.5 - theta2/24.);
                A4 = -cdt2 * (10. + theta2)/30.;
                A5 = cdt2 * (theta2 - 20.) / 120.;
              }
            else
              {
                double SIN = sin(theta);
                A1 = SIN / k1;
                double k2 = k1 * k1;
                A3 = (1. - COS) / k2;
                A4 = (A2 - A1 / cdt) / k2;
                A5 = (A1 / cdt - 1.) / k2;
              }
            // Every row except the last contains exactly one of these in
            // every term.
            A1 *= scaling;
            A2 *= scaling;
            A3 *= scaling;
            A4 *= scaling;
            A5 *= scaling;

            FFTX_COMPLEX_VAL(A1complex, A1, 0.0);
            // FFTX_COMPLEX_VAL(A2complex, A2, 0.);
            // FFTX_COMPLEX_VAL(A3complex, A3, 0.);
            // FFTX_COMPLEX_VAL(A4complex, A4, 0.);
            // FFTX_COMPLEX_VAL(A5complex, A5, 0.);

            DEFINE_DOUBLE_COMPLEX(ikfdx, ikfd[X]);
            DEFINE_DOUBLE_COMPLEX(ikfdy, ikfd[Y]);
            DEFINE_DOUBLE_COMPLEX(ikfdz, ikfd[Z]);
            // double complex ikfdx = (k1D[i])[1] + (k1D[i])[0]*I;
            // double complex ikfdy = (k1D[j])[1] + (k1D[j])[0]*I;
            // double complex ikfdz = (k1D[k])[1] + (k1D[k])[0]*I;
            
            DEFINE_DOUBLE_COMPLEX(ikbdx, ikbd[X]);
            DEFINE_DOUBLE_COMPLEX(ikbdy, ikbd[Y]);
            DEFINE_DOUBLE_COMPLEX(ikbdz, ikbd[Z]);

            // double complex ikbdx = -(k1D[i])[1] + (k1D[i])[0]*I;
            // double complex ikbdy = -(k1D[j])[1] + (k1D[j])[0]*I;
            // double complex ikbdz = -(k1D[k])[1] + (k1D[k])[0]*I;

            double complex mat[DIM+DIM+1][NCOMPS] =
              {
                // row for Ex:  coefficients for E, B, rho, J
                {A2, 0., 0.,
                 0., -A1*ikbdz, A1*ikbdy,
#if RHO_BEFORE_J
                 (A4+A5)*ikfdx,
                 -A5*dt*ikfdx*ikfdx - A1, -A5*dt*ikfdx*ikfdy, -A5*dt*ikfdx*ikfdz
#else
                 -A5*dt*ikfdx*ikfdx - A1, -A5*dt*ikfdx*ikfdy, -A5*dt*ikfdx*ikfdz,
                 (A4+A5)*ikfdx
#endif
                },
                // row for Ey:  coefficients for E, B, rho, J
                {0., A2, 0.,
                 A1*ikbdz, 0., -A1*ikbdx,
#if RHO_BEFORE_J
                 (A4+A5)*ikfdy,
                 -A5*dt*ikfdy*ikfdx, -A5*dt*ikfdy*ikfdy - A1, -A5*dt*ikfdy*ikfdz
#else
                 -A5*dt*ikfdy*ikfdx, -A5*dt*ikfdy*ikfdy - A1, -A5*dt*ikfdy*ikfdz,
                 (A4+A5)*ikfdy
#endif
                },
                // row for Ez:  coefficients for E, B, rho, J
                {0., 0., A2,
                 -A1*ikbdy, A1*ikbdx, 0.,
#if RHO_BEFORE_J
                 (A4+A5)*ikfdz,
                 -A5*dt*ikfdz*ikfdx, -A5*dt*ikfdz*ikfdy, -A5*dt*ikfdz*ikfdz - A1
#else
                 -A5*dt*ikfdz*ikfdx, -A5*dt*ikfdz*ikfdy, -A5*dt*ikfdz*ikfdz - A1,
                 (A4+A5)*ikfdz
#endif
                },
                // row for Bx:  coefficients for E, B, rho, J
                {0., A1*ikfdz, -A1*ikfdy,
                 A2, 0., 0.,
#if RHO_BEFORE_J
                 0.,
                 0., -A3*ikfdz, A3*ikfdy
#else
                 0., -A3*ikfdz, A3*ikfdy,
                 0.
#endif
                },
                // row for By:  coefficients for E, B, rho, J
                {-A1*ikfdz, 0., A1*ikfdx,
                 0., A2, 0.,
#if RHO_BEFORE_J
                 0.,
                 A3*ikfdz, 0., -A3*ikfdx
#else
                 A3*ikfdz, 0., -A3*ikfdx,
                 0.
#endif
                },
                // row for Bz:  coefficients for E, B, rho, J
                {A1*ikfdy, -A1*ikfdx, 0.,
                 0., 0., A2,
#if RHO_BEFORE_J
                 0.,
                 -A3*ikfdy, A3*ikfdx, 0.
#else
                 -A3*ikfdy, A3*ikfdx, 0.,
                 0.,
#endif
                },
                // row for rho:  coefficients for E, B, rho, J
                {0., 0., 0.,
                 0., 0., 0.,
#if RHO_BEFORE_J
                 scaling,
                 -dt*ikfdx*scaling, -dt*ikfdy*scaling, -dt*ikfdz*scaling
#else
                 -dt*ikfdx*scaling, -dt*ikfdy*scaling, -dt*ikfdz*scaling,
                 scaling
#endif
                }
              };
            for (int row = 0; row < DIM+DIM+1; row++)
              for (int col = 0; col < NCOMPS; col++)
                {
                  double complex element = mat[row][col];
                  SET_COMPLEX(dataPtr, creal(element), cimag(element));
                  //(*dataPtr)[0] = creal(element);
                  // (*dataPtr)[1] = cimag(element);
                  // printf("data[%d][%d] = (%e, %e) (%e, %e)\n", row, col, creal(element), cimag(element), *dataPtr[0], *dataPtr[1]);
                  dataPtr++;
                }

 

            // Increment pointers to next point.
            kabsd++;
            ikfd += DIM;
            ikbd += DIM;
        }
  }
  //  for (fftx_complex *initd = data; initd < data + iv*ov*(n*n*n); initd += (iv*ov))
    //    for (int i = 0; i < iv*ov; i++)
    //      {
    //        initd[i][0] = i;
    //        initd[i][1] = 0.;
    //      }

  free(ikfAll);
  free(ikbAll);
  free(kabsAll);

  fftx_iodimx //      size     offsets         strides
    copy_in_dimx[] = {{ns,  0, (n-ns)/2, 0,  iv, iv, 0 }, // Copy dense cube in the middle
                      {ns,  0, (n-ns)/2, 0,  ns*iv, n*iv, 0 }, 
                      {ns,  0, (n-ns)/2, 0,  ns*ns*iv, n*n*iv, 0 }},
    copy_in_howmany[] = {{iv,  0, 0, 0,  1, 1, 0}, // loop over iv variables
                         {batch,  0, 0, 0,  ns*ns*ns*iv, nAll*iv, 0}}, // loop over batch
    copy_out_dimx[] = {{ nd,  (n-nd)/2, 0, 0,  ov, ov, 0 }, // Copy dense cube in the middle
                       { nd,  (n-nd)/2, 0, 0,  n*ov, nd*ov, 0 }, 
                       { nd,  (n-nd)/2, 0, 0,  n*n*ov, nd*nd*ov, 0 }},
    copy_out_howmany[] = {{ov,  0, 0, 0,  1, 1, 0},
                          {batch,  0, 0, 0,  nAll*ov, nd*nd*nd*ov, 0}};

  // Copying all ov components of physicalUpdated
  // to physicalInitial, which has iv > ov components.
  fftx_iodimx // length, offsets (in, out, data), strides (in, out, data)
    copy_phys_dimx[] = {{ n,  0, 0, 0, ov, iv, 0 },
                        { n,  0, 0, 0,  n*ov, n*iv, 0 }, 
                        { n,  0, 0, 0,  n*n*ov, n*n*iv, 0 }},
    copy_phys_howmany[] = {{ov,  0, 0, 0,  1, 1, 0},
                           {batch,  0, 0, 0,  nAll*ov, nAll*iv, 0}};

  fftx_iodim
    dft_dims[] = {{ n, n*n*iv, n*nReduced*iv }, // outermost loop
                  { n, n*iv, nReduced*iv },
                  { n, iv, iv }},  // innermost loop: length, input stride, output stride
    // dft_howmany specifies outermost loops
    dft_howmany[] = {{ batch, nAll*iv, nAll*iv }, //Interleaved batch // outermost; batch many times with stride nAll*iv in both input and output
                     { iv, 1, 1 }}; // loop over iv input variables
    
  physicalInitial = fftx_create_zero_temp_real_b(rank, dft_dims, 2, dft_howmany);

  // copy input cube in the middle
  fftx_plan plan_copy_in = 
    fftx_plan_guru_copy_real_b(rank, // rank=3 for 3D
                               copy_in_dimx, // size, input & output dimensions
                               2, // rank of copy_in_howmany
                               copy_in_howmany, // what is this?
                               in, // input
                               physicalInitial, // output, WAS tmp1
                               MY_FFTX_MODE_SUB // flags
                               );

  fftx_iodim
    idft_dims[] = {{ n, n*nReduced*ov, n*n*ov },
                   { n, nReduced*ov, n*ov },
                   { n, ov, ov }},
    idft_howmany[] = {{ batch, nAll*ov, nAll*ov }, //Interleaved batch
                      { ov, 1, 1 }};

  physicalUpdated = fftx_create_temp_real_b(rank, idft_dims, 2, idft_howmany);

  // copy output cube to the middle
  fftx_plan plan_copy_out =
    fftx_plan_guru_copy_real_b(rank, // rank 3 for 3D copy
                               copy_out_dimx, // dimensions
                               2, // rank of copy_out_howmany
                               copy_out_howmany,
                               physicalUpdated, // input, WAS tmp4
                               out, // output
                               MY_FFTX_MODE_SUB // flags
                               );

  fftx_plan plan_copy_physical =
    fftx_plan_guru_copy_real_b(rank, // rank 3 for 3D copy
                               copy_phys_dimx, // dimensions
                               2, // rank of copy_out_howmany
                               copy_phys_howmany,
                               physicalUpdated, // input
                               physicalInitial, // output
                               MY_FFTX_MODE_SUB // flags
                               );
    
#if defined(USE_PERSISTENT_PLAN) && !defined(VALIDATE)
  // create plan from persistent handle
  printf("create plan from persistent handle\n");
  p = fftx_plan_from_persistent_label(h, planiodata, MY_FFTX_MODE);
#else
  printf("create plan from warpx_plan\n");
  p = warpx_plan(physicalInitial, physicalUpdated, (fftx_complex *)data, rank, n, iv, ov, batch);

  // declare a handle to the plan for future use
  h = fftx_make_persistent_plan(h, p);
  printf("FFTX persistent plan label 0x%X\n", h);
#endif

  // Copy from in to physical.
  fftx_execute(plan_copy_in);
  for (int step = 0; step < nsteps; step++)
    {
#if WRITE_BEFORE_STEP
      { // BEGIN DEBUG
        printf("physical before step %d\n", step);
        fftx_real* physicalInitialPtr = physicalInitial; // out;
        // Order i, j, k for output.
        for (int i = 0; i < n; i++)
          for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
              {
                for (int comp = 0; comp < NCOMPS; comp++)
                  {
                    printf("before phys(%d,%d,%d)[%d] = %f\n", i, j, k, comp,
                           *physicalInitialPtr);
                    physicalInitialPtr++;
                  }
              }
      } // END DEBUG
#endif

      if (verbosity >= 1) 
        {
          printf("calling fftx_execute(p) at step %d\n", step);
        }
      // Start with physicalInitial on iv components.
      // fourierInitial = fft(physicalInitial);
      // fourierUpdated = tensorcontraction(fourierInitial) 
      // physicalUpdated = ifft(fourierUpdated);
      // End with physicalUpdated on ov components.
      fftx_execute(p);
      if (verbosity >= 1) 
        {
          printf("called fftx_execute(p) at step %d\n", step);
        }
      // Now copy physicalUpdated to physicalInitial,
      // overwriting components E, B, and rho;
      // components J will be unchanged in physicalInitial.
      // In the real application, J and rho in physicalInitial would be
      // updated by interpolation from particles.
      fftx_execute(plan_copy_physical);

#if WRITE_AFTER_STEP
      { // BEGIN DEBUG
        printf("physical after step %d\n", step);
        fftx_real* physicalInitialPtr = physicalInitial; // out;
        // Order i, j, k for output.
        for (int i = 0; i < n; i++)
          for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
              {
                for (int comp = 0; comp < NCOMPS; comp++)
                  {
                    printf("after phys(%d,%d,%d)[%d] = %f\n", i, j, k, comp,
                           *physicalInitialPtr);
                    physicalInitialPtr++;
                  }
              }
      } // END DEBUG
#endif

    }
  // Copy from physicalUpdated to out.
  fftx_execute(plan_copy_out);
  if (verbosity >= 1) 
    {
      printf("done fftx_execute(p)\n");
    }

  /*
  {
    printf("tmp1, physical(%d,%d,%d) on %d components:\n",
           iThis, jThis, kThis, iv);
    fftx_real* tmp1ThisStart = tmp1 + ((iThis*n + jThis)*n + kThis)*iv;
    for (int comp = 0; comp < iv; comp++)
      {
        printf("%3d : %f\n", comp, tmp1ThisStart[comp]);
      }
  }
  */

#if WRITE_INITIAL
  { // BEGIN DEBUG: this AGREES with the other implementation.
    fftx_real* physicalInitialPtr = (fftx_real*) physicalInitial;
    // Order i, j, k for output.
    for (int i = 0; i < size1D; i++)
      for (int j = 0; j < size1D; j++)
        for (int k = 0; k < size1D; k++)
          {
            for (int comp = 0; comp < NCOMPS; comp++)
              {
                printf("initial(%d,%d,%d)[%d] = %f\n", i, j, k, comp, *physicalInitialPtr);
                physicalInitialPtr++;
              }
          }
  } // END DEBUG
#endif
  
  {
    printf("fourierInitial, initial fourier(%d,%d,%d) on %d components:\n",
           iThis, jThis, kThis, iv);
    fftx_complex* fourierInitialThisStart = fourierInitial + ((iThis*n + jThis)*nReduced + kThis)*iv;
    for (int comp = 0; comp < iv; comp++)
      {
        fftx_real* fourierInitialhere = fourierInitialThisStart[comp];
        printf("%3d : (%f, %f)\n", comp, fourierInitialhere[0], fourierInitialhere[1]);
      }
  }

#if WRITE_INITIAL_FOURIER
  { // BEGIN DEBUG: this AGREES with octave, even for correct NCOMPS.
    printf("Write fourierInitial\n");
    fftx_complex* fourierInitialPtr = (fftx_complex*) fourierInitial;
    // Order i, j, k for output.
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        for (int k = 0; k < nReduced; k++)
          {
            for (int comp = 0; comp < NCOMPS; comp++)
              {
                fftx_real* fourierInitialhere = (fftx_real*) fourierInitialPtr;
                printf("initial fourier(%d,%d,%d)[%d] = (%f, %f)\n", i, j, k, comp,
                       fourierInitialhere[0], fourierInitialhere[1]);
                fourierInitialPtr++;
              }
          }
  } // END DEBUG
#endif

  {
    printf("fourierUpdated, updated fourier(%d,%d,%d) on %d components:\n",
           iThis, jThis, kThis, iv);
    fftx_complex* fourierUpdatedThisStart = fourierUpdated + ((iThis*n + jThis)*nReduced + kThis)*ov;
    for (int comp = 0; comp < ov; comp++)
      {
        fftx_real* fourierUpdatedhere = fourierUpdatedThisStart[comp];
        printf("%3d : (%f, %f)\n", comp, fourierUpdatedhere[0], fourierUpdatedhere[1]);
      }
  }

#if WRITE_UPDATED_FOURIER
  { // BEGIN DEBUG
    printf("Write fourierUpdated\n");
    fftx_complex* fourierUpdatedPtr = (fftx_complex*) fourierUpdated;
    // Order i, j, k for output.
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        for (int k = 0; k < nReduced; k++)
          {
            for (int comp = 0; comp < DIM+DIM+1; comp++)
              {
                fftx_real* fourierUpdatedhere = (fftx_real*) fourierUpdatedPtr;
                printf("updated fourier(%d,%d,%d)[%d] = (%f, %f)\n", i, j, k, comp,
                       fourierUpdatedhere[0], fourierUpdatedhere[1]);
                fourierUpdatedPtr++;
              }
          }
  } // END DEBUG
#endif

#if WRITE_INVERSE_FOURIER
  { // BEGIN DEBUG: agrees with the other one if NCOMPS=2 and no tensor contraction
    printf("Write physicalUpdated\n");
    fftx_real* physicalUpdatedPtr = physicalUpdated; // out;
    // Order i, j, k for output.
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
          for (int k = 0; k < n; k++)
            {
              for (int comp = 0; comp < ov; comp++)
                {
                  printf("back(%d,%d,%d)[%d] = %f\n", i, j, k, comp,
                         *physicalUpdatedPtr);
                  physicalUpdatedPtr++;
                }
            }
  } // END DEBUG
#endif

  {
    printf("Final physical(%d,%d,%d) on %d components:\n",
           iThis, jThis, kThis, ov);
    fftx_complex* outThisStart = (fftx_complex *) (out + ((iThis*nd + jThis)*nd + kThis)*ov);
    for (int comp = 0; comp < ov; comp++)
      {
        printf("%3d : %f\n", comp, outThisStart[comp][0]);
      }
  }


  // cleanup
  fftx_destroy_data_complex((fftx_complex *)in);
  fftx_destroy_data_complex((fftx_complex *)out);
  fftx_destroy_data_complex((fftx_complex *)data);


  warpx_destroy(p);


  // shut down FFTX
  fftx_shutdown();
  
  return 0;
}
