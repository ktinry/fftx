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


#ifndef __FFTX_H__
#define __FFTX_H__

#include <fftx_build_info.h>

#ifdef USE_CUFFTW
#include <cufftw.h>
#else
#include "fftw3.h"
#endif

#include "fftx_helper.h"

// FFTX data types
typedef double fftx_real;
typedef fftw_complex fftx_complex;
typedef fftw_iodim fftx_iodim;

typedef unsigned fftx_plan_label;

#define FFTX_UNUSED_INT		0
#define FFTX_UNUSED_PTR		NULL

// maximal buffer size for iodim printing
# define FFTX_MAX_BUF		1024

typedef struct {
	int n,
		iofs,	// add in offset; allows shift or reversal
		oofs,	// add out offset; allows shift or reversal
		dofs,	// add data offset; allows shift or reversal
		is,
		os,
		ds;		// independent data stride, used to capture symmetries
} fftx_iodimx;


// data type for temporaries
typedef double *fftx_temp_real;
typedef fftw_complex *fftx_temp_complex;

// generic callback function
typedef void(*fftx_callback)(void);

// true pointwise callback function
typedef void(*fftx_callback_pw)(fftx_complex *, fftx_complex *, fftx_complex *);
// guru callback function that also provides the array base pointers, geometry, and loop variables
typedef void(*fftx_callback_guru_pw)(fftx_complex *, fftx_complex *, fftx_complex *,
	fftx_complex *, fftx_complex *, fftx_complex *,
	int, fftx_iodimx *, int*, int, fftx_iodimx *, int *);

// plan structures
// tensor contraction plans 
typedef struct {
	unsigned flags;
	int n, 
		rank, howmany_rank,
		in_rank, out_rank, data_rank;
	fftx_iodimx *dims, *howmany_dims,
				*in_dims, *out_dims, *data_dims;
	void *in,
		*out,
		*data;
} fftx_plan_tc_s;
typedef fftx_plan_tc_s *fftx_plan_tc;

typedef struct {
	unsigned flags;
	int rank;
	fftx_iodimx *dims;
	void *in,
		*out;
} fftx_plan_copy_s;
typedef fftx_plan_copy_s *fftx_plan_copy;

// plan structures
// pointwise plans 
typedef struct {
	unsigned flags;
	int n, 
		rank, 
		howmany_rank;
	fftx_iodimx *dims, *howmany_dims;
	fftx_callback func;
	void *in,
		*out,
		*data;
} fftx_plan_pw_s;
typedef fftx_plan_pw_s *fftx_plan_pw;

// compose plans
typedef struct {
	unsigned flags;
	int howmany;
	void *plans; // this should be fftx_plan*
} fftx_plan_comp_s;
typedef fftx_plan_comp_s *fftx_plan_comp;

// persistent handle plans
typedef struct {
	unsigned flags;
	fftx_plan_label persistent_handle;
	fftx_real **ptrs; // in, out, and data, in order of being created 
} fftx_plan_persist_s;
typedef fftx_plan_persist_s *fftx_plan_persist;

// fftx polymorphic plan
typedef struct {
	unsigned type;
	union {
		fftw_plan fftw;
		fftx_plan_pw_s pw;
		fftx_plan_comp_s compose;
		fftx_plan_persist_s persist;
		fftx_plan_tc_s tc;
		fftx_plan_copy_s cp;
	} plan;
} fftx_plan_s;
typedef fftx_plan_s *fftx_plan;

typedef struct {
	unsigned fftw_only;
	unsigned values;
} fftx_flags;

// dispatch on plan types in fftx_execute
#define FFTX_PLAN_FFTW					1
#define FFTX_PLAN_POINTWISE_C2C_1D		2
#define FFTX_PLAN_POINTWISE_C2C_GURU	3
#define FFTX_PLAN_COMPOSE				4
#define FFFTX_PLAN_PERSISTENT_HANDLE	5
#define FFTX_PLAN_TC_C2C				6
#define FFTX_PLAN_COPY_C				7
#define FFTX_PLAN_COPY_R				8

// FFTX options
// FFTW 3 uses up to (1U << 21) for its flags
#define FFTW_FLAGS_ONLY	0x1FFFFF

// FFTW compatible options
#define FFTX_ESTIMATE	FFTW_ESTIMATE
#define FFTX_FORWARD	FFTW_FORWARD
#define FFTX_BACKWARD	FFTW_BACKWARD

// types of pointwise functions
#define FFTX_PW_FLAGS			((1U << 25) | (1U << 26))
#define FFTX_PW_POINTWISE		(1U << 25)
#define FFTX_PW_GURU			(1U << 26)

// FFTX modes
#define FFTX_MODE_OBSERVE		(1U << 27)
#define FFTX_HIGH_PERFORMANCE	(1U << 28)	

// FFTX flags
#define FFTX_FLAG_SUBPLAN		(1U << 29)

// types of tensor contraction functions
#define FFTX_TC_FLAGS			(1U << 30)
#define FFTX_TC				(1U << 30)

// pointwise specification helper macros and functions
// defines a simple SSA 2-operand language similar to x86 ASM
// C makes it hard to have full-fledged expressions or 3 address code in this style

void trace_complex_temp(fftx_complex *p);

#define FFTX_COMPLEX_TEMP(_t) fftx_complex _t##v_ = {0.0, 0.0}, *_t = &_t##v_; trace_complex_temp(_t);

#define FFTX_COMPLEX_VAL(_t, _re, _im) fftx_complex _t##v_ = {_re, _im}, *_t = &_t##v_; trace_complex_temp(_t);
#define FFTX_COMPLEX_MULT(_a, _b) _fftx_complex_mult((_a), (_b))
#define FFTX_COMPLEX_ADD(_a, _b) _fftx_complex_add((_a), (_b))
#define FFTX_COMPLEX_SUB(_a, _b) _fftx_complex_sub((_a), (_b))
#define FFTX_COMPLEX_MOV(_a, _b) _fftx_complex_mov((_a), (_b))

#define FFTX_REAL_TEMP(_t) fftx_real _t##v_ = 0.0, *_t = &_t##v_; trace_real_temp(_t);
#define FFTX_REAL_VAL(_t, _val) fftx_real _t##v_ = _val, *_t = &_t##v_; trace_real_temp(_t);
#define FFTX_REAL_MULT(_a, _b) _fftx_real_mult((_a), (_b))
#define FFTX_REAL_ADD(_a, _b) _fftx_real_add((_a), (_b))
#define FFTX_REAL_SUB(_a, _b) _fftx_real_sub((_a), (_b))
#define FFTX_REAL_MOV(_a, _b) _fftx_real_mov((_a), (_b))

#define FFTX_INT_TEMP(_t) int _t##v_ = 0, *_t = &_t##v_; trace_int_temp(_t);
#define FFTX_INT_VAL(_t, _v) int _t##v_ = _v, *_t = &_t##v_; trace_int_temp(_t);
#define FFTX_INT_MULT(_a, _b) _int_mult((_a), (_b))
#define FFTX_INT_ADD(_a, _b) _int_add((_a), (_b))
#define FFTX_INT_SUB(_a, _b) _int_sub((_a), (_b))
#define FFTX_INT_MOV(_a, _b) _int_mov((_a), (_b))

#define FFTX_BOOL_TEMP(_t) unsigned _t##v_ = 0, *_t = &_t##v_; trace_bool_temp(_t);
#define FFTX_BOOL_VAL(_t, _v) unsigned _t##v_ = _v, *_t = &_t##v_; trace_bool_temp(_t);
#define FFTX_BOOL_AND(_a, _b) _bool_and((_a), (_b))
#define FFTX_BOOL_OR(_a, _b) _bool_or((_a), (_b))
#define FFTX_BOOL_NOT(_a, _b) _bool_not((_a), (_b))
#define FFTX_BOOL_XOR(_a, _b) _bool_xor((_a), (_b))
#define FFTX_BOOL_MOV(_a, _b) _bool_mov((_a), (_b))
#define FFTX_BOOL_CMOV(_a, _b, _c) _bool_cmov((_a), (_b), (_c))

// indexing in guru pointwise functions
#define FFTX_IODIMX_NTH(rank, iodimx, vars) perror("not yet implemented\n"); exit(1);

// backend functions for macros above
void _fftx_complex_mult(fftx_complex *a, fftx_complex *b);
void _fftx_complex_add(fftx_complex *a, fftx_complex *b);
void _fftx_complex_sub(fftx_complex *a, fftx_complex *b);
void _fftx_complex_mov(fftx_complex *a, fftx_complex *b);

void _fftx_real_mult(fftx_real *a, fftx_real *b);
void _fftx_real_add(fftx_real *a, fftx_real *b);
void _fftx_real_sub(fftx_real *a, fftx_real *b);
void _fftx_real_mov(fftx_real *a, fftx_real *b);

// FFTX memory and object mamagement
void *fftx_malloc(unsigned size);
void fftx_free(void *p);

void *fftx_internal_malloc(unsigned size);
void fftx_internal_free(void *p);

// init and shutdown
void fftx_init(unsigned flags);
void fftx_shutdown(void);

// FFTX pointer to handle conversions to disable access to temps
// debug use only
fftx_real *fftx_pointer2handle_real(fftx_real *ptr);
fftx_real *fftx_handle2pointer_real(fftx_real *ptr);
fftx_complex *fftx_pointer2handle_complex(fftx_complex *ptr);
fftx_complex *fftx_handle2pointer_complex(fftx_complex *ptr);
fftx_plan fftx_pointer2handle_plan(fftx_plan p);
fftx_plan fftx_handle2pointer_plan(fftx_plan p);
fftx_plan fftx_pointer2persistent_handle_plan(fftx_plan p);

// debug functions
void fftx_print_vector_real(fftx_real *vec, int n);
void fftx_print_vector_complex(fftx_complex *vec, int n);
void fftx_print_csv_real(fftx_real *vec, int n);
void fftx_print_csv_complex(fftx_complex *vec, int n);
void fftx_print_matlab_real(char const *name, fftx_real * vec, int const *size);
void fftx_print_matlab_complex(char const *name, fftx_complex * vec, int const *size);

void fftx_cmp_real_guru(int rank, fftx_iodimx *dims1, fftx_iodimx *dims2, int howmany_rank, fftx_iodimx *howmany_dims1, fftx_iodimx *howmany_dims2,
		fftx_real *in1, fftx_real *in2, unsigned flags);
void howmany_rank_cmp_real(int rank, fftx_iodimx *dims1, fftx_iodimx *dims2, int howmany_rank, fftx_iodimx *howmany_dims1, fftx_iodimx *howmany_dims2,
		fftx_real *in1, fftx_real *in2, unsigned flags);
void rank_cmp_real(int rank, fftx_iodimx *dims1,	fftx_iodimx *dims2, fftx_real *in1, fftx_real *in2, unsigned flags);

void fftx_cmp_complex_guru(int rank, fftx_iodimx *dims1, fftx_iodimx *dims2, int howmany_rank, fftx_iodimx *howmany_dims1, fftx_iodimx *howmany_dims2,
		fftx_complex *in1, fftx_complex *in2, unsigned flags);
void howmany_rank_cmp_complex(int rank, fftx_iodimx *dims1, fftx_iodimx *dims2, int howmany_rank, fftx_iodimx *howmany_dims1, fftx_iodimx *howmany_dims2,
		fftx_complex *in1, fftx_complex *in2, unsigned flags);
void rank_cmp_complex(int rank, fftx_iodimx *dims1,	fftx_iodimx *dims2, fftx_complex *in1, fftx_complex *in2, unsigned flags);

// input/output data creation and cleanup
fftx_real *fftx_create_data_real(int rank, fftx_iodim *zero_dims);
fftx_complex *fftx_create_data_complex(int rank, fftx_iodim *zero_dims);
void fftx_destroy_data_real(fftx_real *tmp);
void fftx_destroy_data_complex(fftx_complex *tmp);

// temporary creation and cleanup
fftx_real *fftx_create_temp_real(int rank, fftx_iodim *zero_dims);
fftx_real *fftx_create_zero_temp_real(int rank, fftx_iodim *zero_dims);
fftx_complex *fftx_create_temp_complex(int rank, fftx_iodim *zero_dims);
fftx_complex *fftx_create_zero_temp_complex(int rank, fftx_iodim *zero_dims);
void fftx_destroy_temp_real(fftx_real *tmp);
void fftx_destroy_temp_complex(fftx_complex *tmp);

//data/temp creation with batch info
//Should replace the one without batch info in all data/temp creation and drop the _b suffix.
fftx_real *fftx_create_data_real_b(int rank, fftx_iodim *zero_dims, int howmany_rank, fftx_iodim *howmany_dims);
fftx_complex *fftx_create_data_complex_b(int rank, fftx_iodim *zero_dims, int howmany_rank, fftx_iodim *howmany_dims);

fftx_real *fftx_create_temp_real_b(int rank, fftx_iodim *zero_dims, int howmany_rank, fftx_iodim *howmany_dims);
fftx_real *fftx_create_zero_temp_real_b(int rank, fftx_iodim *zero_dims, int howmany_rank, fftx_iodim *howmany_zero_dims);
fftx_complex *fftx_create_temp_complex_b(int rank, fftx_iodim *zero_dims, int howmany_rank, fftx_iodim *howmany_dims);
fftx_complex *fftx_create_zero_temp_complex_b(int rank, fftx_iodim *zero_dims, int howmany_rank, fftx_iodim *howmany_zero_dims);

// FFTX  executor
void fftx_execute(fftx_plan x);

// internal executors
void fftx_execute_pointwise_c2c_1d(fftx_plan_pw p);
void fftx_execute_compose(fftx_plan_comp p);
void fftx_execute_pointwise_c2c_guru(fftx_plan_pw p);
void fftx_execute_tc_c2c(fftx_plan_tc p);
void fftx_execute_copy_real(fftx_plan_copy p);
void fftx_execute_copy_complex(fftx_plan_copy p);

// FFTX cleanup
void fftx_destroy_plan(fftx_plan p);
void fftx_destroy_plan_recursive(fftx_plan p);

// 1D plans
// for debugging only
fftx_plan fftx_plan_dft_r2c_1d(int n, fftx_real *in, fftx_complex *out, unsigned flags);
fftx_plan fftx_plan_dft_c2r_1d(int n, fftx_complex *in, fftx_real *out, unsigned flags);
fftx_plan fftx_plan_pointwise_c2c_1d(int n, fftx_complex *in, fftx_complex *out, fftx_complex *data,
	fftx_callback func, unsigned flags);

// guru helpers
fftx_iodim *copy_iodim_vec(int rank, fftx_iodim *dims);
fftx_iodimx *copy_iodimx_vec(int rank, fftx_iodimx *dims);
fftx_iodim *iodimx2iodim_vec(int rank, fftx_iodimx *dims);
int iodimx_in_ofs(int rank, fftx_iodimx *dims);
int iodimx_out_ofs(int rank, fftx_iodimx *dims);
int iodimx_data_ofs(int rank, fftx_iodimx *dims);

void howmany_rank_loop_complex(int rank, fftx_iodimx *dims, int howmany_rank, fftx_iodimx *howmany_dims,
	fftx_complex *in, fftx_complex *out, fftx_complex *data,
	fftx_complex *in_base, fftx_complex *out_base, fftx_complex *data_base,
	int * rankvars, int *rankvars_base, int *howmanyvars, int *howmanyvars_base,
	fftx_callback func, unsigned flags);
void rank_loop_complex(int rank, fftx_iodimx *dims, int howmany_rank, fftx_iodimx *howmany_dims,
	fftx_complex *in, fftx_complex *out, fftx_complex *data,
	fftx_complex *in_base, fftx_complex *out_base, fftx_complex *data_base,
	int * rankvars, int *rankvars_base, int *howmanyvars, int *howmanyvars_base,
	fftx_callback func, unsigned flags);

void howmany_rank_loop_tc_complex(int rank, fftx_iodimx *dims, int howmany_rank, fftx_iodimx *howmany_dims,
	fftx_complex *in, int in_rank, fftx_iodimx *in_dims, fftx_complex *out, int out_rank, fftx_iodimx *out_dims, 
	fftx_complex *data, int data_rank, fftx_iodimx *data_dims,
	unsigned flags);
void rank_loop_tc_complex(int rank, fftx_iodimx *dims, 
	fftx_complex *in, int in_rank, fftx_iodimx *in_dims, fftx_complex *out, int out_rank, fftx_iodimx *out_dims, 
	fftx_complex *data, int data_rank, fftx_iodimx *data_dims,
	unsigned flags);
void tc_c2c(fftx_complex *in, int in_rank, fftx_iodimx *in_dims, fftx_complex *out, int out_rank, fftx_iodimx *out_dims, fftx_complex *data, 
	int data_rank, fftx_iodimx *data_dims);

// sequence of plans
fftx_plan fftx_plan_compose(int howmany, fftx_plan *plans, unsigned flags);

// guru copy plans
fftx_plan fftx_plan_guru_copy_real(int rank, fftx_iodimx *data_dimsx, fftx_real *in, fftx_real *out, unsigned flags);
fftx_plan fftx_plan_guru_copy_complex(int rank, fftx_iodimx *data_dimsx, fftx_complex *in, fftx_complex *out, unsigned flags);

// guru copy plans - batch
fftx_plan fftx_plan_guru_copy_real_b(int rank, fftx_iodimx *data_dimsx, int howmany_rank, fftx_iodimx *howmany_dims, fftx_real *in, fftx_real *out, unsigned flags);
fftx_plan fftx_plan_guru_copy_complex_b(int rank, fftx_iodimx *data_dimsx, int howmany_rank, fftx_iodimx *howmany_dims, fftx_complex *in, fftx_complex *out, unsigned flags);

// guru DFT/RDFT plans
fftx_plan fftx_plan_guru_dft(int rank, fftx_iodim *dims, int howmany_rank, fftx_iodim *howmany_dims,
	fftx_complex *in, fftx_complex *out, int sign, unsigned flags);
fftx_plan fftx_plan_guru_dft_r2c(int rank, fftx_iodim *dims, int howmany_rank, fftx_iodim *howmany_dims,
	fftx_real *in, fftx_complex *out, unsigned flags);
fftx_plan fftx_plan_guru_dft_c2r(int rank, fftx_iodim *dims, int howmany_rank, fftx_iodim *howmany_dims,
	fftx_complex *in, fftx_real *out, unsigned flags);
fftx_plan fftx_plan_guru_pointwise_c2c(int rank, fftx_iodimx *dims, int howmany_rank, fftx_iodimx *howmany_dims,
	fftx_complex *in, fftx_complex *out, fftx_complex *data, fftx_callback func, unsigned flags);
fftx_plan fftx_plan_tc_c2c(int rank, fftx_iodimx *dims, int howmany_rank, fftx_iodimx *howmany_dims,
	fftx_complex *in, int in_rank, fftx_iodimx *in_dims, fftx_complex *out, int out_rank, fftx_iodimx *out_dims, fftx_complex *data, int data_rank, fftx_iodimx *data_dims,
	unsigned flags);



// FFTX handle functions
fftx_plan_label fftx_make_persistent_plan(fftx_plan_label h, fftx_plan p);
fftx_plan fftx_plan_from_persistent_label(fftx_plan_label h, fftx_real **ptrs, unsigned flags);

// helper print functions
char *iodim_print(int rank, fftx_iodim *dims);
char *iodimx_print(int rank, fftx_iodimx *dimsx);

#endif
