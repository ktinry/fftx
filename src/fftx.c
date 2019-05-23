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
 
// FFTX Reference Implementation
//
// defines the FFTX API and implements it via FFTW 3.6+

#include <stdlib.h>
#include <string.h>
//#include <malloc.h>
#include <math.h>
#include "fftx.h"

#ifdef USE_PERSISTENT_PLAN
#include "spiral-dispatch.h"
#endif

fftx_flags fftx_setup_flags() {
	fftx_flags f = {0, 0};
	return f;
}

fftx_flags fftx_set_flags(fftx_flags flags, unsigned choice) {
	flags.fftw_only |= (choice & FFTW_FLAGS_ONLY);
	flags.fftw_only |= (choice & ~FFTW_FLAGS_ONLY);
	return flags;
}

fftx_flags fftx_unset_flags(fftx_flags flags, unsigned choice) {
	flags.fftw_only &= ~choice;
	flags.fftw_only &= ~choice;
	return flags;
}

int fftx_is_set_flags(fftx_flags flags, unsigned choice) {
	return (flags.fftw_only & choice) || (flags.values & choice);
}

void _fftx_destroy_plan(fftx_plan p);

// FFTX->SPIRAL print buffer
static char fftx_print_buf[FFTX_MAX_BUF];
static unsigned callback_print_flag = 0;

// global flag from init
unsigned fftx_global_flags = 0;

// if doesn't fit, do nothing
static void safe_strcat(char *dst, int dest_max, char* src)
{
	int dlen = (int)strlen(dst);
	int slen = (int)strlen(src);
	if ((dlen + slen) < (dest_max - 1))
	{
		strcat(dst, src);
	}
}

#define TPRBUFF_SZ 256

char *iodim_print(int rank, fftx_iodim *dims) {
	char tbuff[TPRBUFF_SZ];
	if(rank == 0) {
		snprintf(fftx_print_buf, FFTX_MAX_BUF, "[ ]");
		return fftx_print_buf;
	}
	snprintf(fftx_print_buf, FFTX_MAX_BUF, "[ rec(n := %d, is := %d, os := %d)", dims[0].n, dims[0].is, dims[0].os);
	for (int i = 1; i < rank; i++)
	{
		snprintf(tbuff, TPRBUFF_SZ, ", rec(n := %d, is := %d, os := %d)", dims[i].n, dims[i].is, dims[i].os);
		safe_strcat(fftx_print_buf, FFTX_MAX_BUF, tbuff);
	}
	safe_strcat(fftx_print_buf, FFTX_MAX_BUF, " ]");
	return fftx_print_buf;
}

char *iodimx_print(int rank, fftx_iodimx *dimsx) {
	char tbuff[TPRBUFF_SZ];
	if(rank == 0) {
		snprintf(fftx_print_buf, FFTX_MAX_BUF, "[ ]");
		return fftx_print_buf;
	}
	snprintf(fftx_print_buf, FFTX_MAX_BUF, "[ rec(n := %d, iofs := %d, oofs := %d, dofs := %d, is := %d, os := %d, ds := %d)",
		dimsx[0].n, dimsx[0].iofs, dimsx[0].oofs, dimsx[0].dofs, dimsx[0].is, dimsx[0].os, dimsx[0].ds);
	for (int i = 1; i < rank; i++)
	{
		snprintf(tbuff, TPRBUFF_SZ, ", rec(n := %d, iofs := %d, oofs := %d, dofs := %d, is := %d, os := %d, ds := %d)",
			dimsx[i].n, dimsx[i].iofs, dimsx[i].oofs, dimsx[i].dofs, dimsx[i].is, dimsx[i].os, dimsx[i].ds);
		safe_strcat(fftx_print_buf, FFTX_MAX_BUF, tbuff);
	}
	safe_strcat(fftx_print_buf, FFTX_MAX_BUF, " ]");
	return fftx_print_buf;
}

// pointwise helper functions
void trace_complex_temp(fftx_complex *p) {
	fftx_real *pr = (fftx_real *)p;
	if ((fftx_global_flags & FFTX_MODE_OBSERVE) && callback_print_flag)
		fprintf(stderr, "\t\t\trec(op := \"FFTX_COMPLEX_VAR\", var := IntHexString(\"%p\"), re := %e, im := %e),\n", p, pr[0], pr[1]);
}

void trace_real_temp(fftx_real *p) {
	if ((fftx_global_flags & FFTX_MODE_OBSERVE) && callback_print_flag)
		fprintf(stderr, "\t\t\trec(op := \"FFTX_REAL_VAR\", var := IntHexString(\"%p\"), val := %e),\n", p, p[0]);
}

void trace_int_temp(int *p) {
	if ((fftx_global_flags & FFTX_MODE_OBSERVE) && callback_print_flag)
		fprintf(stderr, "\t\t\trec(op := \"FFTX_INT_VAR\", var := IntHexString(\"%p\"), val := %d),\n", p, p[0]);
}

void trace_bool_temp(unsigned *p) {
	if ((fftx_global_flags & FFTX_MODE_OBSERVE) && callback_print_flag)
		fprintf(stderr, "\t\t\trec(op := \"FFTX_BOOL_VAR\", var := IntHexString(\"%p\"), val := %u),\n", p, p[0]);
}



// this defines just rudimentary arithmetic for now
void _fftx_complex_mult(fftx_complex *a, fftx_complex *b) {
	fftx_real *ap = (fftx_real*)a,
		*bp = (fftx_real*)b;
	if (callback_print_flag) {
		if (fftx_global_flags & FFTX_MODE_OBSERVE)
			fprintf(stderr, "\t\t\trec(op := \"FFTX_COMPLEX_MUL\", target := IntHexString(\"%p\"), source := IntHexString(\"%p\")),\n", a, b);
	} else {
		fftx_real a_re = ap[0],
			a_im = ap[1],
			b_re = bp[0],
			b_im = bp[1];
		ap[0] = a_re * b_re - a_im * b_im;
		ap[1] = a_re * b_im + a_im * b_re;
	}
}

void _fftx_complex_add(fftx_complex *a, fftx_complex *b) {
	fftx_real *ap = (fftx_real*)a,
		*bp = (fftx_real*)b;
	if (callback_print_flag) {
		if (fftx_global_flags & FFTX_MODE_OBSERVE)
			fprintf(stderr, "\t\t\trec(op := \"FFTX_COMPLEX_ADD\", target := IntHexString(\"%p\"), source := IntHexString(\"%p\")),\n", a, b);
	}
	else {
		ap[0] += bp[0];
		ap[1] += bp[1];
	}
}

void _fftx_complex_sub(fftx_complex *a, fftx_complex *b) {
	fftx_real *ap = (fftx_real*)a,
		*bp = (fftx_real*)b;
	if (callback_print_flag) {
		if (fftx_global_flags & FFTX_MODE_OBSERVE)
			fprintf(stderr, "\t\t\trec(op := \"FFTX_COMPLEX_SUB\", target := IntHexString(\"%p\"), source := IntHexString(\"%p\")),\n", a, b);
	} 	else {
		ap[0] -= bp[0];
		ap[1] -= bp[1];
	}
}

void _fftx_complex_mov(fftx_complex *a, fftx_complex *b) {
	fftx_real *ap = (fftx_real*)a,
		*bp = (fftx_real*)b;
	if (callback_print_flag) {
		if (fftx_global_flags & FFTX_MODE_OBSERVE)
			fprintf(stderr, "\t\t\trec(op := \"FFTX_COMPLEX_MOV\", target := IntHexString(\"%p\"), source := IntHexString(\"%p\")),\n", a, b);
	} else {
		ap[0] = bp[0];
		ap[1] = bp[1];
	}
}

void _fftx_real_mult(fftx_real *a, fftx_real *b) {
	if (callback_print_flag) {
		if (fftx_global_flags & FFTX_MODE_OBSERVE)
			fprintf(stderr, "\t\t\trec(op := \"FFTX_REAL_MULT\", target := IntHexString(\"%p\"), source := IntHexString(\"%p\")),\n", a, b);
	} else 
		*a *= *b;
}

void _fftx_real_add(fftx_real *a, fftx_real *b) {
	if (callback_print_flag) {
		if (fftx_global_flags & FFTX_MODE_OBSERVE)
			fprintf(stderr, "\t\t\trec(op := \"FFTX_REAL_ADD\", target := IntHexString(\"%p\"), source := IntHexString(\"%p\")),\n", a, b);
	} else
		*a += *b;
}

void _fftx_real_sub(fftx_real *a, fftx_real *b) {
	if (callback_print_flag) {
		if (fftx_global_flags & FFTX_MODE_OBSERVE)
			fprintf(stderr, "\t\t\trec(op := \"FFTX_REAL_SUB\", target := IntHexString(\"%p\"), source := IntHexString(\"%p\")),\n", a, b);
	} else
		*a -= *b;
}

void _fftx_real_mov(fftx_real *a, fftx_real *b) {
	if (callback_print_flag) {
		if (fftx_global_flags & FFTX_MODE_OBSERVE)
			fprintf(stderr, "\t\t\trec(op := \"FFTX_REAL_MOV\", target := IntHexString(\"%p\"), source := IntHexString(\"%p\")),\n", a, b);
	} else
		*a = *b;
}

void _int_mult(int *a, int *b) {
	if (callback_print_flag) {
		if (fftx_global_flags & FFTX_MODE_OBSERVE)
			fprintf(stderr, "\t\t\trec(op := \"FFTX_INT_MUL\", target := IntHexString(\"%p\"), source := IntHexString(\"%p\")),\n", a, b);
	} else
		*a *= *b;
}

void _int_add(int *a, int *b) {
	if (callback_print_flag) {
		if (fftx_global_flags & FFTX_MODE_OBSERVE)
			fprintf(stderr, "\t\t\trec(op := \"FFTX_INT_ADD\", target := IntHexString(\"%p\"), source := IntHexString(\"%p\")),\n", a, b);
	} else
		*a += *b;
}

void _int_sub(int *a, int *b) {
	if (callback_print_flag) {
		if (fftx_global_flags & FFTX_MODE_OBSERVE)
			fprintf(stderr, "\t\t\trec(op := \"FFTX_INT_SUB\", target := IntHexString(\"%p\"), source := IntHexString(\"%p\")),\n", a, b);
	} else
		*a -= *b;
}

void _int_mov(int *a, int *b) {
	if (callback_print_flag) {
		if (fftx_global_flags & FFTX_MODE_OBSERVE)
			fprintf(stderr, "\t\t\trec(op := \"FFTX_INT_MOV\", target := IntHexString(\"%p\"), source := IntHexString(\"%p\")),\n", a, b);
	} else
		*a = *b;
}

void _bool_and(unsigned *a, unsigned *b) {
	if (callback_print_flag) {
		if (fftx_global_flags & FFTX_MODE_OBSERVE)
			fprintf(stderr, "\t\t\trec(op := \"FFTX_BOOL_AND\", target := IntHexString(\"%p\"), source := IntHexString(\"%p\")),\n", a, b);
	} else
		*a &= *b;
}

void _bool_or(unsigned *a, unsigned *b) {
	if (callback_print_flag) {
		if (fftx_global_flags & FFTX_MODE_OBSERVE)
			fprintf(stderr, "\t\t\trec(op := \"FFTX_BOOL_OR\", target := IntHexString(\"%p\"), source := IntHexString(\"%p\")),\n", a, b);
	} else
		*a |= *b;
}

void _bool_xor(unsigned *a, unsigned *b) {
	if (callback_print_flag) {
		if (fftx_global_flags & FFTX_MODE_OBSERVE)
			fprintf(stderr, "\t\t\trec(op := \"FFTX_BOOL_XOR\", target := IntHexString(\"%p\"), source := IntHexString(\"%p\")),\n", a, b);
	} else
		*a ^= *b;
}

void _bool_not(unsigned *a, unsigned *b) {
	if (callback_print_flag) {
		if (fftx_global_flags & FFTX_MODE_OBSERVE)
			fprintf(stderr, "\t\t\trec(op := \"FFTX_BOOL_NOT\", target := IntHexString(\"%p\"), source := IntHexString(\"%p\")),\n", a, b);
	} else
		*a = !*b;
}

void _bool_mov(unsigned *a, unsigned *b) {
	if (callback_print_flag) {
		if (fftx_global_flags & FFTX_MODE_OBSERVE)
			fprintf(stderr, "\t\t\trec(op := \"FFTX_BOOL_MOV\", target := IntHexString(\"%p\"), source := IntHexString(\"%p\")),\n", a, b);
	} else
		*a = *b;
}

void _bool_cmov(unsigned *a, unsigned *b, unsigned *c) {
	if (callback_print_flag) {
		if (fftx_global_flags & FFTX_MODE_OBSERVE)
			fprintf(stderr, "\t\t\trec(op := \"FFTX_BOOL_CMOV\", target := IntHexString(\"%p\"), source := IntHexString(\"%p\"), cond := IntHexString(\"%p\")),\n", a, b, c);
	} else
		*a = *c ? *b : *a;
}


// init and shutdown
void fftx_init(unsigned flags) {
	fftx_global_flags = flags;
#ifdef USE_PERSISTENT_PLAN
#include "spiral-init.c"
#endif
	if (fftx_global_flags & FFTX_MODE_OBSERVE)
		fprintf(stderr, "fftx_session := [\n\trec(op := \"fftx_init\", flags := IntHexString(\"%X\")),\n", flags);
}

void fftx_shutdown(void) {
	if (fftx_global_flags & FFTX_MODE_OBSERVE)
		fprintf(stderr, "\trec(op := \"fftx_shutdown\")\n];\n");
	fftx_global_flags = 0;
}

// FFTX executor
void _fftx_execute(fftx_plan p) {
	switch (p->type) {
		case FFTX_PLAN_FFTW:
			fftw_execute(p->plan.fftw);
			break;
		case FFTX_PLAN_COPY_C:
#ifdef USE_CUFFTW
			cudaDeviceSynchronize();
#endif
			fftx_execute_copy_complex(&p->plan.cp);
			break;
		case FFTX_PLAN_COPY_R:
#ifdef USE_CUFFTW
			cudaDeviceSynchronize();
#endif
			fftx_execute_copy_real(&p->plan.cp);
			break;
		case FFTX_PLAN_POINTWISE_C2C_1D:
#ifdef USE_CUFFTW
			cudaDeviceSynchronize();
#endif
			fftx_execute_pointwise_c2c_1d(&p->plan.pw);
			break;
		case FFTX_PLAN_POINTWISE_C2C_GURU:
#ifdef USE_CUFFTW
			cudaDeviceSynchronize();
#endif
			fftx_execute_pointwise_c2c_guru(&p->plan.pw);
			break;
		case FFTX_PLAN_TC_C2C:
#ifdef USE_CUFFTW
			cudaDeviceSynchronize();
#endif
			fftx_execute_tc_c2c(&p->plan.tc);
			break;
		case FFTX_PLAN_COMPOSE:
			fftx_execute_compose(&p->plan.compose);
			break;
		case FFFTX_PLAN_PERSISTENT_HANDLE:
			switch (p->plan.persist.persistent_handle) {
#ifdef USE_PERSISTENT_PLAN
#include "spiral-dispatch.c"
#endif
				default:
					perror("not yet implemented");
					exit(1);
			}
			break;
		default:
			perror("not yet implemented");
			exit(1);
	}
}

void fftx_execute(fftx_plan p) {
	if (fftx_global_flags & FFTX_MODE_OBSERVE)
		fprintf(stderr, "\trec(op := \"fftx_execute\", plan := IntHexString(\"%p\")),\n", p);
	_fftx_execute(p);
#ifdef USE_CUFFTW
	cudaDeviceSynchronize();
#endif
}


// recursive destroy all plans and free all data
void _fftx_destroy_plan_recursive(fftx_plan p) {
	switch (p->type) {
		case FFTX_PLAN_COMPOSE:
			for (int i = 0; i < p->plan.compose.howmany; i++) {
				_fftx_destroy_plan(fftx_handle2pointer_plan(((fftx_plan*)p->plan.compose.plans)[i]));
			}
			fftx_internal_free(p->plan.compose.plans);
			break;
	}
}

// FFTX destroy plan
void _fftx_destroy_plan(fftx_plan p) {

	switch (p->type) {
		case FFTX_PLAN_FFTW:
			fftw_destroy_plan(p->plan.fftw);
			break;
		case FFTX_PLAN_COMPOSE:
			_fftx_destroy_plan_recursive(p);
			break;
		case FFTX_PLAN_COPY_C:
			fftx_internal_free(p->plan.cp.dims);
			break;
		case FFTX_PLAN_COPY_R:
			fftx_internal_free(p->plan.cp.dims);
			break;
		case FFTX_PLAN_POINTWISE_C2C_GURU:
			fftx_internal_free(p->plan.pw.dims);
			fftx_internal_free(p->plan.pw.howmany_dims);
			break;
		case FFTX_PLAN_TC_C2C:
			fftx_internal_free(p->plan.tc.dims);
			fftx_internal_free(p->plan.tc.howmany_dims);
			fftx_internal_free(p->plan.tc.in_dims);
			fftx_internal_free(p->plan.tc.out_dims);
			fftx_internal_free(p->plan.tc.data_dims);
			break;
	}
	fftx_internal_free(p);
}


void fftx_destroy_plan(fftx_plan p) {
	if (fftx_global_flags & FFTX_MODE_OBSERVE)
		fprintf(stderr, "\trec(op := \"fftx_destroy_plan\", ptr := IntHexString(\"%p\")),\n", p);
	_fftx_destroy_plan(p);
}


// recursively destroy all plans and free all data
void fftx_destroy_plan_recursive(fftx_plan p) {

	_fftx_destroy_plan_recursive(p);
//	switch (p->type) {
//		case FFTX_PLAN_COMPOSE:
//			for (int i = 0; i < p->plan.compose.howmany; i++) {
//				_fftx_destroy_plan(fftx_handle2pointer_plan(((fftx_plan*)p->plan.compose.plans)[i]));
//			}
//			break;
//	}
//	_fftx_destroy_plan(p);
	if (fftx_global_flags & FFTX_MODE_OBSERVE)
		fprintf(stderr, "\trec(op := \"fftx_destroy_plan_recursive\", ptr := IntHexString(\"%p\")),\n", p);
}


// 1D functions

fftx_plan fftx_plan_dft_r2c_1d(int n, fftx_real *in, fftx_complex *out, unsigned flags) {
	fftx_plan p = (fftx_plan)fftx_internal_malloc(sizeof(fftx_plan_s));
	p->type = FFTX_PLAN_FFTW;
	p->plan.fftw = fftw_plan_dft_r2c_1d(n, fftx_handle2pointer_real(in), fftx_handle2pointer_complex(out), flags & FFTW_FLAGS_ONLY);
	return flags & FFTX_FLAG_SUBPLAN ? fftx_pointer2handle_plan(p) : p;
}

fftx_plan fftx_plan_dft_c2r_1d(int n, fftx_complex *in, fftx_real *out, unsigned flags) {
	fftx_plan p = (fftx_plan)fftx_internal_malloc(sizeof(fftx_plan_s));
	p->type = FFTX_PLAN_FFTW;
	p->plan.fftw = fftw_plan_dft_c2r_1d(n, fftx_handle2pointer_complex(in), fftx_handle2pointer_real(out), flags & FFTW_FLAGS_ONLY);
	return flags & FFTX_FLAG_SUBPLAN ? fftx_pointer2handle_plan(p) : p;
}

fftx_plan fftx_plan_guru_dft(int rank, fftx_iodim *dims, int howmany_rank, fftx_iodim *howmany_dims,
	fftx_complex *in, fftx_complex *out, int sign, unsigned flags) {

	fftx_plan p = (fftx_plan)fftx_internal_malloc(sizeof(fftx_plan_s));
	p->type = FFTX_PLAN_FFTW;
	p->plan.fftw = fftw_plan_guru_dft(rank, dims, howmany_rank, howmany_dims,
		fftx_handle2pointer_complex(in), fftx_handle2pointer_complex(out), sign, flags & FFTW_FLAGS_ONLY);
	p = flags & FFTX_FLAG_SUBPLAN ? fftx_pointer2handle_plan(p) : p;

	if (fftx_global_flags & FFTX_MODE_OBSERVE) {
		fprintf(stderr, "\trec(op := \"fftx_plan_guru_dft\", ptr := IntHexString(\"%p\"), rank := %d, dims := %s, ",
			p, rank, iodim_print(rank, dims));
		fprintf(stderr, "howmany_rank := %d, howmany_dims := %s,\n\t\tinp := IntHexString(\"%p\"), outp := IntHexString(\"%p\"), sign := %d, flags := IntHexString(\"%X\")),\n",
			howmany_rank, iodim_print(howmany_rank, howmany_dims), in, out, sign, flags);
	}
	return p;
}

// DFT/RDFT guru functions

fftx_plan fftx_plan_guru_dft_r2c(int rank, fftx_iodim *dims, int howmany_rank, fftx_iodim *howmany_dims,
	fftx_real *in, fftx_complex *out, unsigned flags) {

	fftx_plan p = (fftx_plan)fftx_internal_malloc(sizeof(fftx_plan_s));
	p->type = FFTX_PLAN_FFTW;
	p->plan.fftw = fftw_plan_guru_dft_r2c(rank, dims, howmany_rank, howmany_dims,
		fftx_handle2pointer_real(in), fftx_handle2pointer_complex(out), flags & FFTW_FLAGS_ONLY);
	p = flags & FFTX_FLAG_SUBPLAN ? fftx_pointer2handle_plan(p) : p;

	if (fftx_global_flags & FFTX_MODE_OBSERVE) {
		fprintf(stderr, "\trec(op := \"fftx_plan_guru_dft_r2c\", ptr := IntHexString(\"%p\"), rank := %d, dims := %s, ",
			p, rank, iodim_print(rank, dims));
		fprintf(stderr, "howmany_rank := %d, howmany_dims := %s,\n\t\tinp := IntHexString(\"%p\"), outp := IntHexString(\"%p\"), flags := IntHexString(\"%X\")),\n",
			howmany_rank, iodim_print(howmany_rank, howmany_dims), in, out, flags);
	}
	return p;
}

fftx_plan fftx_plan_guru_dft_c2r(int rank, fftx_iodim *dims, int howmany_rank, fftx_iodim *howmany_dims,
	fftx_complex *in, fftx_real *out, unsigned flags) {

	fftx_plan p = (fftx_plan)fftx_internal_malloc(sizeof(fftx_plan_s));
	p->type = FFTX_PLAN_FFTW;
	p->plan.fftw = fftw_plan_guru_dft_c2r(rank, dims, howmany_rank, howmany_dims,
		fftx_handle2pointer_complex(in), fftx_handle2pointer_real(out), flags & FFTW_FLAGS_ONLY);
	p = flags & FFTX_FLAG_SUBPLAN ? fftx_pointer2handle_plan(p) : p;

	if (fftx_global_flags & FFTX_MODE_OBSERVE) {
		fprintf(stderr, "\trec(op := \"fftx_plan_guru_dft_c2r\", ptr := IntHexString(\"%p\"), rank := %d, dims := %s, ",
			p, rank, iodim_print(rank, dims));
		fprintf(stderr, "howmany_rank := %d, howmany_dims := %s,\n\t\tinp := IntHexString(\"%p\"), outp := IntHexString(\"%p\"), flags := IntHexString(\"%X\")),\n",
			howmany_rank, iodim_print(howmany_rank, howmany_dims), in, out, flags);
	}
	return p;
}


int iodimx_in_ofs(int rank, fftx_iodimx *dims) {
	int ofs = 0;
	for (int i = 0; i < rank; i++)
		ofs += (dims[i].is<0 ? -dims[i].is : dims[i].is) * dims[i].iofs;
	return ofs;
}

int iodimx_out_ofs(int rank, fftx_iodimx *dims) {
	int ofs = 0;
	for (int i = 0; i < rank; i++)
		ofs += (dims[i].os<0 ? -dims[i].os : dims[i].os) * dims[i].oofs;
	return ofs;
}

int iodimx_data_ofs(int rank, fftx_iodimx *dims) {
	int ofs = 0;
	for (int i = 0; i < rank; i++)
		ofs += dims[i].ds * dims[i].dofs;
	return ofs;
}

fftx_iodim *copy_iodim_vec(int rank, fftx_iodim *dims) {
	fftx_iodim *rdims = (fftx_iodim *)fftx_internal_malloc(rank * sizeof(fftx_iodim));
	for (int i = 0; i < rank; i++) {
		rdims[i].n = dims[i].n;
		rdims[i].is = dims[i].is;
		rdims[i].os = dims[i].os;
	}
	return rdims;
}

fftx_iodimx *copy_iodimx_vec(int rank, fftx_iodimx *dims) {
	fftx_iodimx *rdims = (fftx_iodimx *)fftx_internal_malloc(rank * sizeof(fftx_iodimx));
	for (int i = 0; i < rank; i++) {
		rdims[i].n = dims[i].n;
		rdims[i].iofs = dims[i].iofs;
		rdims[i].oofs = dims[i].oofs;
		rdims[i].dofs = dims[i].dofs;
		rdims[i].is = dims[i].is;
		rdims[i].os = dims[i].os;
		rdims[i].ds = dims[i].ds;
	}
	return rdims;
}

fftx_iodim *iodimx2iodim_vec(int rank, fftx_iodimx *dims) {
	fftx_iodim *rdims = (fftx_iodim *)fftx_internal_malloc(rank * sizeof(fftx_iodim));
	for (int i = 0; i < rank; i++) {
		rdims[i].n = dims[i].n;
		rdims[i].is = dims[i].is;
		rdims[i].os = dims[i].os;
	}
	return rdims;
}

int iodimx_in_fullstride(int rank, fftx_iodimx *dims) {
	int s = 1, t;
	for (int i = 0; i < rank; i++) {
		t = dims[i].is * dims[i].n;
		s = t>s ? t : s;
	}

	return s;
}

int iodimx_out_fullstride(int rank, fftx_iodimx *dims) {
	int s = 1, t;
	for (int i = 0; i < rank; i++) {
		t = dims[i].os * dims[i].n;
		s = t>s ? t : s;
	}

	return s;
}

int iodimx_data_fullstride(int rank, fftx_iodimx *dims) {
	int s = 1, t;
	for (int i = 0; i < rank; i++) {
		t = dims[i].ds * dims[i].n;
		s = t>s ? t : s;
	}

	return s;
}

char has_neg_stride(int rank, fftx_iodimx *dims) {
	for (int i = 0; i < rank; ++i)
		if((dims[i].is < 0) || (dims[i].os < 0))
			return 1;
	return 0;

}

/*
 * copy a real rank-dimensional data cube given by data-dims into a contiguous rank-dimensional temporary
 * to zero-pad, create a new temporary with create_zero_temp_real
 */
fftx_plan fftx_plan_guru_copy_real(int rank, fftx_iodimx *data_dimsx, fftx_real *in, fftx_real *out, unsigned flags) {
	fftx_plan p = (fftx_plan)fftx_internal_malloc(sizeof(fftx_plan_s));
	if (has_neg_stride(rank, data_dimsx)) { 
		p->type = FFTX_PLAN_COPY_R;
		p->plan.cp.rank = rank;
		p->plan.cp.dims = copy_iodimx_vec(rank, data_dimsx);
		p->plan.cp.in = fftx_handle2pointer_real(in) + iodimx_in_ofs(rank, data_dimsx);
		p->plan.cp.out = fftx_handle2pointer_real(out) + iodimx_out_ofs(rank, data_dimsx);
	} else { // Map to FFTW plan
#ifdef USE_CUFFTW
		p->type = FFTX_PLAN_COPY_R;
		p->plan.cp.rank = rank;
		p->plan.cp.dims = copy_iodimx_vec(rank, data_dimsx);
		p->plan.cp.in = fftx_handle2pointer_real(in);
		p->plan.cp.out = fftx_handle2pointer_real(out);
#else			
		int copy_rank = 0;
		fftx_iodim copy_dims = { 1, 0, 0 },
			*data_dims = iodimx2iodim_vec(rank, data_dimsx);
		fftw_r2r_kind kind = FFTW_DHT;
		p->type = FFTX_PLAN_FFTW;
		p->plan.fftw = fftw_plan_guru_r2r(copy_rank, &copy_dims, rank, 
			data_dims,
			fftx_handle2pointer_real(in) + iodimx_in_ofs(rank, data_dimsx), 
			fftx_handle2pointer_real(out) + iodimx_out_ofs(rank, data_dimsx),
			&kind, flags & FFTW_FLAGS_ONLY);
		// fftx_internal_free(data_dims);
#endif
	}
	p = flags & FFTX_FLAG_SUBPLAN ? fftx_pointer2handle_plan(p) : p;

	if (fftx_global_flags & FFTX_MODE_OBSERVE)
		fprintf(stderr, "\trec(op := \"fftx_plan_guru_copy_real\", ptr := IntHexString(\"%p\"), rank := %d, data_dimsx := %s,\n\t\tinp := IntHexString(\"%p\"), outp := IntHexString(\"%p\"), flags := IntHexString(\"%X\")),\n",
			p, rank, iodimx_print(rank, data_dimsx), in, out, flags);
	return p;
}

int iodimx_size(int rank, fftx_iodimx *dims) {
	int s = 1;
	for (int i = 0; i < rank; i++) {
		s *= dims[i].n;
	}
	return s;
}

// loop over the howmany_ranks and plan recursively

void howmany_rank_loop_copy_real(int rank, fftx_iodimx *dims, int howmany_rank, fftx_iodimx *howmany_dims,
	fftx_real *in, fftx_real *out, 
	fftx_plan *plans, unsigned flags) {

	if (howmany_rank == 0) {
		fftx_plan p = (fftx_plan)fftx_internal_malloc(sizeof(fftx_plan_s));
		if (has_neg_stride(rank, dims)) { 
			p->type = FFTX_PLAN_COPY_R;
			p->plan.cp.rank = rank;
			p->plan.cp.dims = copy_iodimx_vec(rank, dims);
			p->plan.cp.in = fftx_handle2pointer_real(in);
			p->plan.cp.out = fftx_handle2pointer_real(out);
		} else { // Map to FFTW plan
#ifdef USE_CUFFTW
			p->type = FFTX_PLAN_COPY_R;
			p->plan.cp.rank = rank;
			p->plan.cp.dims = copy_iodimx_vec(rank, dims);
			p->plan.cp.in = fftx_handle2pointer_real(in);
			p->plan.cp.out = fftx_handle2pointer_real(out);
#else			
			int copy_rank = 0;
			fftx_iodim copy_dims = { 1, 0, 0 },
				*data_dims = iodimx2iodim_vec(rank, dims);
			fftw_r2r_kind kind = FFTW_DHT;
			p->type = FFTX_PLAN_FFTW;
			p->plan.fftw = fftw_plan_guru_r2r(copy_rank, &copy_dims, rank, 
				data_dims,
				fftx_handle2pointer_real(in), 
				fftx_handle2pointer_real(out),
				&kind, flags & FFTW_FLAGS_ONLY);
			// fftx_internal_free(data_dims);
#endif
		}
		*plans = p;
	} else
		for (int i = 0; i < howmany_dims->n; i++) {
			int howmany = iodimx_size(howmany_rank - 1, howmany_dims + 1);
			howmany_rank_loop_copy_real(rank, dims, howmany_rank - 1, howmany_dims + 1,
				in + i * howmany_dims->is, out + i * howmany_dims->os,
				plans + i*howmany, flags);
		}
}

fftx_plan fftx_plan_guru_copy_real_b(int rank, fftx_iodimx *data_dimsx, int howmany_rank, fftx_iodimx *howmany_dims, 
	fftx_real *in, fftx_real *out, unsigned flags) {

	fftx_plan p = (fftx_plan)fftx_internal_malloc(sizeof(fftx_plan_s));
	p->type = FFTX_PLAN_COMPOSE;
	int howmany = iodimx_size(howmany_rank, howmany_dims);
	p->plan.compose.howmany = howmany;
	p->plan.compose.plans = fftx_internal_malloc(howmany * sizeof(fftx_plan));

	howmany_rank_loop_copy_real(rank, data_dimsx, howmany_rank, howmany_dims,
			in + iodimx_in_ofs(rank, data_dimsx) + iodimx_in_ofs(howmany_rank, howmany_dims),
			out + iodimx_out_ofs(rank, data_dimsx) + iodimx_out_ofs(howmany_rank, howmany_dims),
			(fftx_plan *)p->plan.compose.plans, flags);

	p->plan.compose.flags = flags;
	p = flags & FFTX_FLAG_SUBPLAN ? fftx_pointer2handle_plan(p) : p;

	if (fftx_global_flags & FFTX_MODE_OBSERVE)
		fprintf(stderr, "\trec(op := \"fftx_plan_guru_copy_real\", ptr := IntHexString(\"%p\"), rank := %d, data_dimsx := %s,\n\t\tinp := IntHexString(\"%p\"), outp := IntHexString(\"%p\"), flags := IntHexString(\"%X\")),\n",
			p, rank, iodimx_print(rank, data_dimsx), in, out, flags);
	return p;
}

// copy a complex rank-dimensional data cube given by data-dims into a contiguous rank-dimensional temporary
// to zero-pad, create a new temporary with create_zero_temp_complex
// If copying with negative stride use a different plan type as FFTW does not support this kind of data access.

fftx_plan fftx_plan_guru_copy_complex(int rank, fftx_iodimx *data_dimsx, fftx_complex *in, fftx_complex *out, unsigned flags) {
	fftx_plan p = (fftx_plan)fftx_internal_malloc(sizeof(fftx_plan_s));
	if (has_neg_stride(rank, data_dimsx)) { 
		p->type = FFTX_PLAN_COPY_C;
		p->plan.cp.rank = rank;
		p->plan.cp.dims = copy_iodimx_vec(rank, data_dimsx);
		p->plan.cp.in = fftx_handle2pointer_complex(in) + iodimx_in_ofs(rank, data_dimsx);
		p->plan.cp.out = fftx_handle2pointer_complex(out) + iodimx_out_ofs(rank, data_dimsx);
	} else { // Map to FFTW plan
#ifdef USE_CUFFTW
			p->type = FFTX_PLAN_COPY_C;
			p->plan.cp.rank = rank;
			p->plan.cp.dims = copy_iodimx_vec(rank, data_dimsx);
			p->plan.cp.in = fftx_handle2pointer_complex(in);
			p->plan.cp.out = fftx_handle2pointer_complex(out);
#else			
		int copy_rank = 0;
		fftx_iodim copy_dims = { 1, 0, 0 }, 
			*data_dims = iodimx2iodim_vec(rank, data_dimsx);
		p->type = FFTX_PLAN_FFTW;
		p->plan.fftw = fftw_plan_guru_dft(copy_rank, &copy_dims, rank, data_dims,
			fftx_handle2pointer_complex(in) + iodimx_in_ofs(rank, data_dimsx), 
			fftx_handle2pointer_complex(out) + iodimx_out_ofs(rank, data_dimsx), 
			FFTW_FORWARD, flags & FFTW_FLAGS_ONLY);
#endif
	}
	p =  flags & FFTX_FLAG_SUBPLAN ? fftx_pointer2handle_plan(p) : p;
	if (fftx_global_flags & FFTX_MODE_OBSERVE)
		fprintf(stderr, "\trec(op := \"fftx_plan_guru_copy_complex\", ptr := IntHexString(\"%p\"), rank := %d, data_dimsx := %s,\n\t\tinp := IntHexString(\"%p\"), outp := IntHexString(\"%p\"), flags := IntHexString(\"%X\")),\n",
			p, rank, iodimx_print(rank, data_dimsx), in, out, flags);
	return p;
}

// loop over the howmany_ranks and plan recursively
void howmany_rank_loop_copy_complex(int rank, fftx_iodimx *dims, int howmany_rank, fftx_iodimx *howmany_dims,
	fftx_complex *in, fftx_complex *out, 
	fftx_plan *plans, unsigned flags) {

	if (howmany_rank == 0) {
		fftx_plan p = (fftx_plan)fftx_internal_malloc(sizeof(fftx_plan_s));
		if (has_neg_stride(rank, dims)) { 
			p->type = FFTX_PLAN_COPY_C;
			p->plan.cp.rank = rank;
			p->plan.cp.dims = copy_iodimx_vec(rank, dims);
			p->plan.cp.in = fftx_handle2pointer_complex(in);
			p->plan.cp.out = fftx_handle2pointer_complex(out);
		} else { // Map to FFTW plan
#ifdef USE_CUFFTW
			p->type = FFTX_PLAN_COPY_C;
			p->plan.cp.rank = rank;
			p->plan.cp.dims = copy_iodimx_vec(rank, dims);
			p->plan.cp.in = fftx_handle2pointer_complex(in);
			p->plan.cp.out = fftx_handle2pointer_complex(out);
#else			
			int copy_rank = 0;
			fftx_iodim copy_dims = { 1, 0, 0 }, 
				*data_dims = iodimx2iodim_vec(rank, dims);
			// p = fftx_internal_malloc(sizeof(fftx_plan_s));
			p->type = FFTX_PLAN_FFTW;
			p->plan.fftw = fftw_plan_guru_dft(copy_rank, &copy_dims, rank, data_dims,
				fftx_handle2pointer_complex(in), 
				fftx_handle2pointer_complex(out), 
				FFTW_FORWARD, flags & FFTW_FLAGS_ONLY);
#endif
		}
		*plans = p;
	} else
		for (int i = 0; i < howmany_dims->n; i++) {
			int howmany = iodimx_size(howmany_rank - 1, howmany_dims + 1);
			howmany_rank_loop_copy_complex(rank, dims, howmany_rank - 1, howmany_dims + 1,
				in + i * howmany_dims->is, out + i * howmany_dims->os,
				plans + i*howmany, flags);
		}
}

fftx_plan fftx_plan_guru_copy_complex_b(int rank, fftx_iodimx *data_dimsx, int howmany_rank, fftx_iodimx *howmany_dims, 
	fftx_complex *in, fftx_complex *out, unsigned flags) {

	fftx_plan p = (fftx_plan)fftx_internal_malloc(sizeof(fftx_plan_s));
	p->type = FFTX_PLAN_COMPOSE;
	int howmany = iodimx_size(howmany_rank, howmany_dims);
	p->plan.compose.howmany = howmany;
	p->plan.compose.plans = fftx_internal_malloc(howmany * sizeof(fftx_plan));
	howmany_rank_loop_copy_complex(rank, data_dimsx, howmany_rank, howmany_dims,
			in + iodimx_in_ofs(rank, data_dimsx) + iodimx_in_ofs(howmany_rank, howmany_dims),
			out + iodimx_out_ofs(rank, data_dimsx) + iodimx_out_ofs(howmany_rank, howmany_dims),
			(fftx_plan *)p->plan.compose.plans, flags);

	p->plan.compose.flags = flags;
	p =  flags & FFTX_FLAG_SUBPLAN ? fftx_pointer2handle_plan(p) : p;
	if (fftx_global_flags & FFTX_MODE_OBSERVE) {
		fprintf(stderr, "\trec(op := \"fftx_plan_guru_copy_complex\", ptr := IntHexString(\"%p\"), rank := %d, data_dimsx := %s,\n",
			p, rank, iodimx_print(rank, data_dimsx));
		fprintf(stderr, "\t\thowmany_rank := %d, howmany_dimsx := %s,\n",
			howmany_rank, iodimx_print(howmany_rank, howmany_dims));
		fprintf(stderr, "\t\tinp := IntHexString(\"%p\"), outp := IntHexString(\"%p\"), flags := IntHexString(\"%X\")),\n", 
			in, out, flags);
	}
	return p;
}


// loop over the ranks, when rank == 0 apply copy
void rank_loop_cp_real(int rank, fftx_iodimx *dims, 
	fftx_real *in, fftx_real *out, 
	unsigned flags) {

	int i;

	if (rank == 0) {
		*out = *in;
	}
	else 
		for (i = 0; i < dims->n; i++) {
			rank_loop_cp_real(rank - 1, dims + 1, 
				in + i * dims->is, out + i * dims->os,
				flags);
		}
}

void rank_loop_cp_complex(int rank, fftx_iodimx *dims, 
	fftx_complex *in, fftx_complex *out, 
	unsigned flags) {

	int i;

	if (rank == 0) {
		(*out)[0] = (*in)[0];
		(*out)[1] = (*in)[1];
	}
	else
		for (i = 0; i < dims->n; i++) {
			rank_loop_cp_complex(rank - 1, dims + 1, 
				in + i * dims->is, out + i * dims->os,
				flags);
		}
}

// Copy with negative stride: executor
void fftx_execute_copy_real(fftx_plan_copy p) {
	fftx_real *in = (fftx_real *)p->in,
		*out = (fftx_real *)p->out;
	int rank = p->rank;
	fftx_iodimx *dims = copy_iodimx_vec(rank, p->dims); 

	rank_loop_cp_real(rank, dims, in, out, p->flags);
}

void fftx_execute_copy_complex(fftx_plan_copy p) {
	fftx_complex *in = (fftx_complex *)p->in,
		*out = (fftx_complex *)p->out;
	int rank = p->rank;
	fftx_iodimx *dims = copy_iodimx_vec(rank, p->dims); 

	rank_loop_cp_complex(rank, dims, in, out, p->flags);
}


// 1D pointwise function application: planner

fftx_plan fftx_plan_pointwise_c2c_1d(int n, fftx_complex *in, fftx_complex *out, fftx_complex *data,
	fftx_callback func, unsigned flags) {
	
	fftx_plan p = (fftx_plan)fftx_internal_malloc(sizeof(fftx_plan_s));
	p->type = FFTX_PLAN_POINTWISE_C2C_1D;
	p->plan.pw.n = n;
	p->plan.pw.flags = flags & FFTX_PW_FLAGS;
	p->plan.pw.func = func;
	p->plan.pw.in = fftx_handle2pointer_complex(in);
	p->plan.pw.out = fftx_handle2pointer_complex(out);
	p->plan.pw.data = fftx_handle2pointer_complex(data);

	return flags & FFTX_FLAG_SUBPLAN ? fftx_pointer2handle_plan(p) : p;
}

// 1D pointwise function application: executor

void fftx_execute_pointwise_c2c_1d(fftx_plan_pw p) {
	fftx_callback_pw func = (fftx_callback_pw)p->func;
	fftx_complex *in = (fftx_complex *)p->in, 
		*out = (fftx_complex *)p->out, 
		*data = (fftx_complex *)p->data;
	switch (p->flags) {
		case FFTX_PW_POINTWISE:
			for (int i = 0; i < p->n; i++)
				func(out+i, in+i, data+i);
		}
}

// tensor contraction application: planner
fftx_plan fftx_plan_tc_c2c(int rank, fftx_iodimx *dims, int howmany_rank, fftx_iodimx *howmany_dims,
	fftx_complex *in, int in_rank, fftx_iodimx *in_dims, 
	fftx_complex *out, int out_rank, fftx_iodimx *out_dims, 
	fftx_complex *data, int data_rank, fftx_iodimx *data_dims,
	unsigned flags) {
	
	fftx_plan p = (fftx_plan)fftx_internal_malloc(sizeof(fftx_plan_s));
	p->type = FFTX_PLAN_TC_C2C;
	p->plan.tc.rank = rank;
	p->plan.tc.dims = copy_iodimx_vec(rank, dims);
	p->plan.tc.howmany_rank = howmany_rank;
	p->plan.tc.howmany_dims = copy_iodimx_vec(howmany_rank, howmany_dims);
	p->plan.tc.in = fftx_handle2pointer_complex(in);
	p->plan.tc.out = fftx_handle2pointer_complex(out);
	p->plan.tc.data = fftx_handle2pointer_complex(data);
	p->plan.tc.in_rank = in_rank;
	p->plan.tc.in_dims = copy_iodimx_vec(in_rank, in_dims);
	p->plan.tc.out_rank = out_rank;
	p->plan.tc.out_dims = copy_iodimx_vec(out_rank, out_dims);
	p->plan.tc.data_rank = data_rank;
	p->plan.tc.data_dims = copy_iodimx_vec(data_rank, data_dims);
	p->plan.tc.flags = flags & FFTX_TC_FLAGS;

	p = flags & FFTX_FLAG_SUBPLAN ? fftx_pointer2handle_plan(p) : p;

	if (fftx_global_flags & FFTX_MODE_OBSERVE) {
		fprintf(stderr, "\trec(op := \"fftx_plan_tc_c2c\", ptr := IntHexString(\"%p\"), rank := %d, dimsx := %s,\n\t\t",
			p, rank, iodimx_print(rank, dims));
		fprintf(stderr, "howmany_rank := %d, howmany_dimsx := %s,\n\t\t",
			howmany_rank, iodimx_print(howmany_rank, howmany_dims));
		fprintf(stderr, "inp := IntHexString(\"%p\"), in_rank := %d, in_dimsx := %s,\n\t\t",
			in, in_rank, iodimx_print(in_rank, in_dims));
		fprintf(stderr, "outp := IntHexString(\"%p\"), out_rank := %d, out_dimsx := %s,\n\t\t",
			out, out_rank, iodimx_print(out_rank, out_dims));
		fprintf(stderr, "datap := IntHexString(\"%p\"), data_rank := %d, data_dimsx := %s,\n\t\t",
			data, data_rank, iodimx_print(data_rank, data_dims));
		fprintf(stderr, "flags := IntHexString(\"%X\")), \n", flags);
	}

	return p;
}

// tensor contraction application: executor

void fftx_execute_tc_c2c(fftx_plan_tc p) {
	fftx_complex *in = (fftx_complex *)p->in,
		*out = (fftx_complex *)p->out,
		*data = (fftx_complex *)p->data;
	unsigned flags = p->flags;
	int rank = p->rank,
		howmany_rank = p->howmany_rank;
	fftx_iodimx *dims = copy_iodimx_vec(rank, p->dims), 
		*howmany_dims = copy_iodimx_vec(rank, p->howmany_dims);
	int in_rank = p->in_rank;
	fftx_iodimx *in_dims = copy_iodimx_vec(in_rank, p->in_dims);
	int out_rank = p->out_rank;
	fftx_iodimx *out_dims = copy_iodimx_vec(out_rank, p->out_dims);
	int data_rank = p->data_rank;
	fftx_iodimx *data_dims = copy_iodimx_vec(data_rank, p->data_dims);

	// first loop over howmany_ranks, then ranks to apply tc
	howmany_rank_loop_tc_complex(rank, dims, howmany_rank, howmany_dims,
		in + iodimx_in_ofs(rank, dims) + iodimx_in_ofs(howmany_rank, howmany_dims) + iodimx_in_ofs(in_rank, in_dims), 
		in_rank, in_dims, 
		out + iodimx_out_ofs(rank, dims) + iodimx_out_ofs(howmany_rank, howmany_dims) + iodimx_out_ofs(out_rank, out_dims), 
		out_rank, out_dims, 
		data + iodimx_data_ofs(rank, dims) + iodimx_data_ofs(howmany_rank, howmany_dims) + iodimx_data_ofs(data_rank, data_dims), 
		data_rank, data_dims,
		flags);
}


// loop over the howmany_ranks recursively, when consumed switch to rank loop

void howmany_rank_loop_tc_complex(int rank, fftx_iodimx *dims, int howmany_rank, fftx_iodimx *howmany_dims,
	fftx_complex *in, int in_rank, fftx_iodimx *in_dims,
	fftx_complex *out, int out_rank, fftx_iodimx *out_dims, 
	fftx_complex *data, int data_rank, fftx_iodimx *data_dims,
	unsigned flags) {

	if (howmany_rank == 0)
		rank_loop_tc_complex(rank, dims, 
			in, in_rank, in_dims, out, out_rank, out_dims, data, data_rank, data_dims,
			flags);
	else
		for (int i = 0; i < howmany_dims->n; i++) {
			howmany_rank_loop_tc_complex(rank, dims, howmany_rank - 1, howmany_dims + 1,
				in + i * howmany_dims->is, in_rank, in_dims, 
				out + i * howmany_dims->os, out_rank, out_dims, 
				data + i * howmany_dims->ds, data_rank, data_dims,
				flags);
		}
}


// loop over the ranks, when rank == 0 apply TC
void rank_loop_tc_complex(int rank, fftx_iodimx *dims, 
	fftx_complex *in, int in_rank, fftx_iodimx *in_dims, fftx_complex *out, int out_rank, fftx_iodimx *out_dims, 
	fftx_complex *data, int data_rank, fftx_iodimx *data_dims,
	unsigned flags) {

	int i;

	if (rank == 0) {
		if (flags & FFTX_TC)
			tc_c2c(in, in_rank, in_dims, out, out_rank, out_dims, data, data_rank, data_dims);
	}
	else 
		for (i = 0; i < dims->n; i++) {
			rank_loop_tc_complex(rank - 1, dims + 1, 
				in + i * dims->is, in_rank, in_dims, 
				out + i * dims->os, out_rank, out_dims,
				data + i * dims->ds, data_rank, data_dims,
				flags);
		}
}

void tc_c2c(fftx_complex *in, int in_rank, fftx_iodimx *in_dims, fftx_complex *out, int out_rank, fftx_iodimx *out_dims, fftx_complex *data, 
	int data_rank, fftx_iodimx *data_dims) {

	int i;

	if(in_rank == 0 && out_rank == 0) {
		(*out)[0] += (*data)[0] * (*in)[0] - (*data)[1] * (*in)[1];
		(*out)[1] += (*data)[0] * (*in)[1] + (*data)[1] * (*in)[0];
	} else if(out_rank > 0) {
		for (i = 0; i < out_dims->n; i++) {
			tc_c2c(in, in_rank, in_dims, out + i * out_dims->os, out_rank - 1, out_dims + 1, data + i * data_dims->ds, data_rank - 1, data_dims + 1);
		}
	} else {
		for (i = 0; i < in_dims->n; i++) {
			tc_c2c(in + i * in_dims->is, in_rank - 1, in_dims + 1, out, out_rank, out_dims, data + i * data_dims->ds, data_rank - 1, data_dims + 1);
		}
	}
}

void tc_c2c_tmp(fftx_complex *in, int in_rank, fftx_iodimx *in_dims, fftx_complex *out, int out_rank, fftx_iodimx *out_dims, fftx_complex *data, 
	int data_rank, fftx_iodimx *data_dims) {

	int i;

	if(out_rank == 0) { //Dot product 
		fftx_complex *ti, *td, ot;
		ot[0] = ot[1] = 0.;
		for (i = 0; i < data_dims->n; i++) {
			ti = (in + i * in_dims->is); td = (data + i * data_dims->ds);
			ot[0] += (*td)[0] * (*ti)[0] - (*td)[1] * (*ti)[1];
			ot[1] += (*td)[0] * (*ti)[1] + (*td)[1] * (*ti)[0];
		}
		(*out)[0] = ot[0]; (*out)[1] = ot[1];
	} else if(out_rank == 1) {
		for (i = 0; i < out_dims->n; i++) {
			tc_c2c(in, in_rank, in_dims, out + i * out_dims->os, out_rank - 1, out_dims + 1, data + i * data_dims->ds, data_rank - 1, data_dims + 1);
		}
	} else {
		perror("TC not implemented");
		exit(-1);
	}
}

// guru pointwise function application: planner
fftx_plan fftx_plan_guru_pointwise_c2c(int rank, fftx_iodimx *dims, int howmany_rank, fftx_iodimx *howmany_dims,
	fftx_complex *in, fftx_complex *out, fftx_complex *data, fftx_callback func, unsigned flags) {
	
	fftx_plan p = (fftx_plan)fftx_internal_malloc(sizeof(fftx_plan_s));
	fftx_complex *sym_in = (fftx_complex *)fftx_pointer2handle_complex((void *)1), 
		*sym_out = (fftx_complex *)fftx_pointer2handle_complex((void *)2), 
		*sym_data = (fftx_complex *)fftx_pointer2handle_complex((void *)3);
	p->type = FFTX_PLAN_POINTWISE_C2C_GURU;
	p->plan.pw.rank = rank;
	p->plan.pw.dims = copy_iodimx_vec(rank, dims);
	p->plan.pw.howmany_rank = howmany_rank;
	p->plan.pw.howmany_dims = copy_iodimx_vec(howmany_rank, howmany_dims);
	p->plan.pw.in = fftx_handle2pointer_complex(in);
	p->plan.pw.out = fftx_handle2pointer_complex(out);
	p->plan.pw.data = fftx_handle2pointer_complex(data);
	p->plan.pw.func = func;
	p->plan.pw.flags = flags & FFTX_PW_FLAGS;

	p = flags & FFTX_FLAG_SUBPLAN ? fftx_pointer2handle_plan(p) : p;

	if (fftx_global_flags & FFTX_MODE_OBSERVE) {
		fprintf(stderr, "\trec(op := \"fftx_plan_guru_pointwise_c2c\", ptr := IntHexString(\"%p\"), rank := %d, dimsx := %s,\n\t\t",
			p, rank, iodimx_print(rank, dims));
		fprintf(stderr, "howmany_rank := %d, howmany_dimsx := %s,\n\t\tinp := IntHexString(\"%p\"), outp := IntHexString(\"%p\"), data := IntHexString(\"%p\"), flags := IntHexString(\"%X\"), \n\t\tcallback := [\n",
			howmany_rank, iodimx_print(howmany_rank, howmany_dims), in, out, data, flags);
		callback_print_flag = 1;
		if (flags & FFTX_PW_POINTWISE) {
			fprintf(stderr, "\t\t\trec(op := \"call\", inp := IntHexString(\"%p\"), outp := IntHexString(\"%p\"), data := IntHexString(\"%p\")),\n", sym_in, sym_out, sym_data);
			((fftx_callback_pw)func)(sym_in, sym_out, sym_data);
		} else { 
			perror("not yet implemented");
			exit(1);
		}
		callback_print_flag = 0;
		fprintf(stderr, "\t\t\trec(op := \"return\")\n\t\t]), \n");
	}
	return p;
}


// guru pointwise function application: executor

void fftx_execute_pointwise_c2c_guru(fftx_plan_pw p) {
	fftx_callback func = p->func;
	fftx_complex *in = (fftx_complex *)p->in,
		*out = (fftx_complex *)p->out,
		*data = (fftx_complex *)p->data;
	unsigned flags = p->flags;
	int rank = p->rank,
		howmany_rank = p->howmany_rank;
	fftx_iodimx *dims = copy_iodimx_vec(rank, p->dims), 
		*howmany_dims = copy_iodimx_vec(rank, p->howmany_dims);
	int *rankvars = (int *)fftx_internal_malloc(sizeof(int)*rank);
	int *howmanyvars = (int *)fftx_internal_malloc(sizeof(int)*howmany_rank);

	// first loop over howmany_ranks, then ranks to apply the pointwise function
	howmany_rank_loop_complex(rank, dims, howmany_rank, howmany_dims,
		in + iodimx_in_ofs(rank, dims) + iodimx_in_ofs(howmany_rank, howmany_dims),
		out + iodimx_out_ofs(rank, dims) + iodimx_out_ofs(howmany_rank, howmany_dims),
		data + iodimx_data_ofs(rank, dims) + iodimx_data_ofs(howmany_rank, howmany_dims),
		in, out, data, rankvars, rankvars, howmanyvars, howmanyvars,
		func, flags);
}


// loop over the howmany_ranks recursively, when consumed switch to rank loop

void howmany_rank_loop_complex(int rank, fftx_iodimx *dims, int howmany_rank, fftx_iodimx *howmany_dims,
	fftx_complex *in, fftx_complex *out, fftx_complex *data,
	fftx_complex *in_base, fftx_complex *out_base, fftx_complex *data_base,
	int * rankvars, int *rankvars_base, int *howmanyvars, int *howmanyvars_base,
	fftx_callback func, unsigned flags) {

	if (howmany_rank == 0)
		rank_loop_complex(rank, dims, howmany_rank, howmany_dims, 
			in, out, data, 
			in_base, out_base, data_base, 
			rankvars, rankvars_base, howmanyvars, howmanyvars_base,
			func, flags);
	else
		for (int i = 0; i < howmany_dims->n; i++) {
			*howmanyvars = i;
			howmany_rank_loop_complex(rank, dims, howmany_rank - 1, howmany_dims + 1,
				in + i * howmany_dims->is, out + i * howmany_dims->os, data + i * howmany_dims->ds, 
				in_base, out_base, data_base, 
				rankvars, rankvars_base, howmanyvars + 1, howmanyvars_base,
				func, flags);
		}
}


// loop over the ranks, rank == 1 is special for conjugate even, and rank == 0 is pointwise application

void rank_loop_complex(int rank, fftx_iodimx *dims, int howmany_rank, fftx_iodimx *howmany_dims,
	fftx_complex *in, fftx_complex *out, fftx_complex *data,
	fftx_complex *in_base, fftx_complex *out_base, fftx_complex *data_base,
	int * rankvars, int *rankvars_base, int *howmanyvars, int *howmanyvars_base,
	fftx_callback func, unsigned flags) {

	int i;

	if (rank == 0) {
		if (flags & FFTX_PW_POINTWISE)
			((fftx_callback_pw)func)(in, out, data);
		else if (flags & FFTX_PW_GURU) 
			((fftx_callback_guru_pw)func)(in, out, data, in_base, out_base, data_base, rank, dims, rankvars_base, howmany_rank, howmany_dims, howmanyvars_base);
		else {
			perror("unknown pointwise type");
			exit(-1);
		}
	}
	else 
		for (i = 0; i < dims->n; i++) {
			*rankvars = i;
			rank_loop_complex(rank - 1, dims + 1, howmany_rank, howmany_dims,
				in + i * dims->is, out + i * dims->os, data + i * dims->ds, 
				in_base, out_base, data_base, rankvars + 1, rankvars_base, howmanyvars, howmanyvars_base,
				func, flags);
		}
}

// Compare two cubes elementwise

void fftx_cmp_real_guru(int rank, fftx_iodimx *dims1, fftx_iodimx *dims2, int howmany_rank, fftx_iodimx *howmany_dims1, fftx_iodimx *howmany_dims2,
		fftx_real *in1, fftx_real *in2, unsigned flags) {

	// first loop over howmany_ranks, then ranks to apply the pointwise function
	howmany_rank_cmp_real(rank, dims1, dims2, howmany_rank, howmany_dims1, howmany_dims2,
		in1 + iodimx_in_ofs(rank, dims1) + iodimx_in_ofs(howmany_rank, howmany_dims1),
		in2 + iodimx_in_ofs(rank, dims2) + iodimx_in_ofs(howmany_rank, howmany_dims2),
		flags);
}


void howmany_rank_cmp_real(int rank, fftx_iodimx *dims1, fftx_iodimx *dims2, int howmany_rank, fftx_iodimx *howmany_dims1, fftx_iodimx *howmany_dims2,
		fftx_real *in1, fftx_real *in2, unsigned flags) {

	if (howmany_rank == 0)
		rank_cmp_real(rank, dims1, dims2, in1, in2, flags);
	else
		for (int i = 0; i < howmany_dims1->n; i++) {
			howmany_rank_cmp_real(rank, dims1, dims2, howmany_rank - 1, howmany_dims1 + 1, howmany_dims1 + 2,
				in1 + i * howmany_dims1->is, in2 + i * howmany_dims2->is,
				flags);
		}
}


// loop over the ranks, rank == 1 is special for conjugate even, and rank == 0 is pointwise application

void rank_cmp_real(int rank, fftx_iodimx *dims1, fftx_iodimx *dims2, fftx_real *in1, fftx_real *in2, unsigned flags) {

	int i;
	if (rank == 0) {
		double d = fabs( (*in1) - (*in2) );
		if(d > 1e-6)
			printf("Diff: %f (in1: %f, in2: %f)\n", d, *in1, *in2);
	}
	else
		for (i = 0; i < dims1->n; i++) {
			rank_cmp_real(rank - 1, dims1 + 1, dims2 + 1,
				in1 + i * dims1->is, in2 + i * dims2->is, flags);
		}
}

void fftx_cmp_complex_guru(int rank, fftx_iodimx *dims1, fftx_iodimx *dims2, int howmany_rank, fftx_iodimx *howmany_dims1, fftx_iodimx *howmany_dims2,
		fftx_complex *in1, fftx_complex *in2, unsigned flags) {

	// first loop over howmany_ranks, then ranks to apply the pointwise function
	howmany_rank_cmp_complex(rank, dims1, dims2, howmany_rank, howmany_dims1, howmany_dims2,
		in1 + iodimx_in_ofs(rank, dims1) + iodimx_in_ofs(howmany_rank, howmany_dims1),
		in2 + iodimx_in_ofs(rank, dims2) + iodimx_in_ofs(howmany_rank, howmany_dims2),
		flags);
}


void howmany_rank_cmp_complex(int rank, fftx_iodimx *dims1, fftx_iodimx *dims2, int howmany_rank, fftx_iodimx *howmany_dims1, fftx_iodimx *howmany_dims2,
		fftx_complex *in1, fftx_complex *in2, unsigned flags) {

	if (howmany_rank == 0)
		rank_cmp_complex(rank, dims1, dims2, in1, in2, flags);
	else
		for (int i = 0; i < howmany_dims1->n; i++) {
			howmany_rank_cmp_complex(rank, dims1, dims2, howmany_rank - 1, howmany_dims1 + 1, howmany_dims2 + 1,
				in1 + i * howmany_dims1->is, in2 + i * howmany_dims2->is,
				flags);
		}
}


// loop over the ranks, rank == 1 is special for conjugate even, and rank == 0 is pointwise application

void rank_cmp_complex(int rank, fftx_iodimx *dims1,	fftx_iodimx *dims2, fftx_complex *in1, fftx_complex *in2, unsigned flags) {

	int i;
	if (rank == 0) {
		double dr = fabs( ((*in1)[0]) - ((*in2)[0]) ),
			   dc = fabs( ((*in1)[1]) - ((*in2)[1]) );
		if (dr > 1e-6 || dc > 1e-6)
			printf("Diff: (%f,%f)  -- in1: (%f, %f), in2: (%f, %f)\n", dr, dc, (*in1)[0], (*in1)[1], (*in2)[0], (*in2)[1]);
	}
	else
		for (i = 0; i < dims1->n; i++) {
			rank_cmp_complex(rank - 1, dims1 + 1, dims2 + 1,
				in1 + i * dims1->is, in2 + i * dims2->is, flags);
		}
}

// compose planner
// SSA sequence defining a DAG. dependencies are encoded via vectors

fftx_plan fftx_plan_compose(int howmany, fftx_plan *plans, unsigned flags) {
	fftx_plan p = (fftx_plan)fftx_internal_malloc(sizeof(fftx_plan_s));
    
	p->type = FFTX_PLAN_COMPOSE;
	p->plan.compose.howmany = howmany;
	p->plan.compose.plans = fftx_internal_malloc(howmany * sizeof(fftx_plan));
	for (int i = 0; i < howmany; i++)
		((fftx_plan*)p->plan.compose.plans)[i] = plans[i];
	p->plan.compose.flags = flags;
	p = flags & FFTX_FLAG_SUBPLAN ? fftx_pointer2handle_plan(p) : p;

	if (fftx_global_flags & FFTX_MODE_OBSERVE) {
		fprintf(stderr, "\trec(op := \"fftx_plan_compose\", ptr := IntHexString(\"%p\"),\n\t\tsubplans := [\n",
			p);
		for (int i = 0; i < howmany - 1; i++)
			fprintf(stderr, "\t\t\tIntHexString(\"%p\"),\n", plans[i]);
		fprintf(stderr, "\t\t\tIntHexString(\"%p\")],\n\t\tflags :=  IntHexString(\"%X\")),\n", plans[howmany - 1], flags);
	}
	return p;
}


// compose executor

void fftx_execute_compose(fftx_plan_comp p) {
	for (int i = 0; i < p->howmany; i++) {
		_fftx_execute(fftx_handle2pointer_plan(((fftx_plan*)p->plans)[i]));
	}
}


// debug functions

void fftx_print_vector_real(fftx_real *vec, int n) {
	fftx_real *tmp = fftx_handle2pointer_real(vec);

	for (int i = 0; i < n; i++)
		printf("%e ", (double)tmp[i]);
	printf("\n");
}

void fftx_print_vector_complex(fftx_complex *vec, int n) {
	fftx_complex *tmp = fftx_handle2pointer_complex(vec);

	for (int i = 0; i < n; i++)
		printf("(%e, %e) ", ((double*)tmp)[2*i], ((double*)tmp)[2*i+1]);
	printf("\n");
}

void fftx_print_csv_real(fftx_real *vec, int n) {
	fftx_real *tmp = fftx_handle2pointer_real(vec);

	for (int i = 0; i < n; i++)
		printf("%e\n", (double)tmp[i]);
}

void fftx_print_csv_complex(fftx_complex *vec, int n) {
	fftx_complex *tmp = fftx_handle2pointer_complex(vec);

	for (int i = 0; i < n; i++)
		printf("%e, %e\n", ((double*)tmp)[2*i], ((double*)tmp)[2*i+1]);
}

void fftx_print_matlab_real(char const *name, fftx_real * vec, int const *size) {
	for (int i = 0; i < size[0]; i++) {
		printf("%s(1:%d,1:%d,%d) = [ ", name, size[1], size[2], i+1);
		for (int j = 0; j < size[1]; ++j) {
			for (int k = 0; k < size[2]; ++k)
			{
				printf("%f ", vec[i*size[1]*size[2]+j*size[2] + k]);
			}
			printf("; ");
		}
		printf("]\n");
	}
}

void fftx_print_matlab_complex(char const *name, fftx_complex * vec, int const *size) {
	for (int i = 0; i < size[0]; i++) {
		printf("%s(1:%d,1:%d,%d) = [ ", name, size[1], size[2], i+1);
		for (int j = 0; j < size[1]; ++j) {
			for (int k = 0; k < size[2]; ++k)
			{
				printf("%f+%fi ", vec[i*size[1]*size[2]+j*size[2] + k][0], vec[i*size[1]*size[2]+j*size[2] + k][1]);
			}
			printf("; ");
		}
		printf("]\n");
	}
}

// destroy temporary vectors

void fftx_destroy_temp_real(fftx_real *tmp) {
	if (fftx_global_flags & FFTX_MODE_OBSERVE)
		fprintf(stderr, "\trec(op := \"fftx_destroy_temp_real\", ptr := IntHexString(\"%p\")),\n", tmp);
	fftx_internal_free(fftx_handle2pointer_real(tmp));
}

void fftx_destroy_temp_complex(fftx_complex *tmp) {
	if (fftx_global_flags & FFTX_MODE_OBSERVE)
		fprintf(stderr, "\trec(op := \"fftx_destroy_temp_complex\", ptr := IntHexString(\"%p\")),\n", tmp);
	fftx_internal_free(fftx_handle2pointer_complex(tmp));
}



// create/destroy data vectors

fftx_real *fftx_create_data_real(int rank, fftx_iodim *dims) {

	fftx_real *tmp;
	int tmpsize = 1;
	int i;

	for (i = 0; i < rank; i++)
		tmpsize *= dims[i].n;

	tmp = (fftx_real *)fftx_internal_malloc(tmpsize * sizeof(fftx_real));

	if (fftx_global_flags & FFTX_MODE_OBSERVE)
		fprintf(stderr, "\trec(op := \"fftx_create_data_real\", rank := %d, dims := %s, ptr := IntHexString(\"%p\")),\n",
			rank, iodim_print(rank, dims), tmp);
	
	return tmp;
}

fftx_real *fftx_create_data_real_b(int rank, fftx_iodim *dims, int howmany_rank, fftx_iodim *howmany_dims) {

	fftx_real *tmp;
	int tmpsize = 1;
	int i;

	for (i = 0; i < rank; i++)
		tmpsize *= dims[i].n;

	for (i = 0; i < howmany_rank; i++)
		tmpsize *= howmany_dims[i].n;

	tmp = (fftx_real *)fftx_internal_malloc(tmpsize * sizeof(fftx_real));

	if (fftx_global_flags & FFTX_MODE_OBSERVE)
		fprintf(stderr, "\trec(op := \"fftx_create_data_real\", rank := %d, dims := %s, ptr := IntHexString(\"%p\")),\n",
			rank, iodim_print(rank, dims), tmp);
	
	return tmp;
}


fftx_complex *fftx_create_data_complex(int rank, fftx_iodim *dims) {

	fftx_complex *tmp;
	int tmpsize = 1;
	int i;

	for (i = 0; i < rank; i++)
		tmpsize *= dims[i].n;

	tmp = (fftx_complex *)fftx_internal_malloc(tmpsize * sizeof(fftx_complex));

	if (fftx_global_flags & FFTX_MODE_OBSERVE)
		fprintf(stderr, "\trec(op := \"fftx_create_data_complex\", rank := %d, dims := %s, ptr := IntHexString(\"%p\")),\n",
			rank, iodim_print(rank, dims), tmp);

	return tmp;
}

fftx_complex *fftx_create_data_complex_b(int rank, fftx_iodim *dims, int howmany_rank, fftx_iodim *howmany_dims) {

	fftx_complex *tmp;
	int tmpsize = 1;
	int i;

	for (i = 0; i < rank; i++)
		tmpsize *= dims[i].n;

	for (i = 0; i < howmany_rank; i++)
		tmpsize *= howmany_dims[i].n;

	tmp = (fftx_complex *)fftx_internal_malloc(tmpsize * sizeof(fftx_complex));

	if (fftx_global_flags & FFTX_MODE_OBSERVE) {
		fprintf(stderr, "\trec(op := \"fftx_create_data_complex\", rank := %d, dims := %s,\n",
			rank, iodim_print(rank, dims));
		fprintf(stderr, "howmany_rank := %d, howmany_dims := %s, ptr := IntHexString(\"%p\")),\n",
			howmany_rank, iodim_print(howmany_rank, howmany_dims), tmp);
	}

	return tmp;
}

// destroy data vectors

void fftx_destroy_data_real(fftx_real *tmp) {
	if (fftx_global_flags & FFTX_MODE_OBSERVE)
		fprintf(stderr, "\trec(op := \"fftx_destroy_data_real\", ptr := IntHexString(\"%p\")),\n", tmp);
	fftx_internal_free(tmp);
}

void fftx_destroy_data_complex(fftx_complex *tmp) {
	if (fftx_global_flags & FFTX_MODE_OBSERVE)
		fprintf(stderr, "\trec(op := \"fftx_destroy_data_complex\", ptr := IntHexString(\"%p\")),\n", tmp);
	fftx_internal_free(tmp);
}




// turn pointer to handle and back
// leverages the fact that in 64bit Intel the address space is only 47 bit
// setting bits 48--63 to 1 makes the pointer illegal but is easily reversible
// this is architecture dependent

fftx_real *fftx_pointer2handle_real(fftx_real *ptr) {
#ifdef _WIN64
	ptr = (fftx_real *)(((__int64)ptr) | 0xF000000000000000L);
#endif
	return ptr;
}

fftx_real *fftx_handle2pointer_real(fftx_real *ptr) {
#ifdef _WIN64
	ptr = (fftx_real *)(((__int64)ptr) & 0x0000FFFFFFFFFFFFL);
#endif
	return ptr;
}

fftx_complex *fftx_pointer2handle_complex(fftx_complex *ptr) {
#ifdef _WIN64
	ptr = (fftx_complex *)(((__int64)ptr) | 0xA000000000000000L);
#endif
	return ptr;
}

fftx_complex *fftx_handle2pointer_complex(fftx_complex *ptr) {
#ifdef _WIN64
	ptr = (fftx_complex *)(((__int64)ptr) & 0x0000FFFFFFFFFFFFL);
#endif
	return ptr;
}

fftx_plan fftx_pointer2handle_plan(fftx_plan p) {
#ifdef _WIN64
	p = (fftx_plan)(((__int64)p) | 0xB000000000000000L);
#endif
	return p;
}

fftx_plan fftx_handle2pointer_plan(fftx_plan p) {
#ifdef _WIN64
	p = (fftx_plan)(((__int64)p) & 0x0000FFFFFFFFFFFFL);
#endif
	return p;
}

fftx_plan fftx_pointer2persistent_handle_plan(fftx_plan p) {
#ifdef _WIN64
	p = (fftx_plan)(((__int64)p) | 0xC000000000000000L);
#endif
	return p;
}


// will return an illegal pointer (a used as handle)

fftx_real *fftx_create_zero_temp_real(int rank, fftx_iodim *zero_dims) {

	fftx_real *tmp;
	int tmpsize = 1;
	int i;

	for (i = 0; i < rank; i++)
		tmpsize *= zero_dims[i].n;

	tmp = (fftx_real *)fftx_internal_malloc(tmpsize * sizeof(fftx_real));
	for (i = 0; i < tmpsize; i++)
		tmp[i] = 0.0;
	tmp = fftx_pointer2handle_real(tmp);

	if (fftx_global_flags & FFTX_MODE_OBSERVE) {
		fprintf(stderr, "\trec(op := \"fftx_create_zero_temp_real\", rank := %d, dims := %s, ptr := IntHexString(\"%p\")),\n",
			rank, iodim_print(rank, zero_dims), tmp);
	}
	return tmp;
}

fftx_real *fftx_create_zero_temp_real_b(int rank, fftx_iodim *zero_dims, int howmany_rank, fftx_iodim *howmany_zero_dims) {

	fftx_real *tmp;
	int tmpsize = 1;
	int i;

	for (i = 0; i < rank; i++)
		tmpsize *= zero_dims[i].n;

	for (i = 0; i < howmany_rank; i++)
		tmpsize *= howmany_zero_dims[i].n;

	tmp = (fftx_real *)fftx_internal_malloc(tmpsize * sizeof(fftx_real));
	for (i = 0; i < tmpsize; i++)
		tmp[i] = 0.0;
	tmp = fftx_pointer2handle_real(tmp);

	if (fftx_global_flags & FFTX_MODE_OBSERVE) {
		fprintf(stderr, "\trec(op := \"fftx_create_zero_temp_real\", rank := %d, dims := %s,\n",
			rank, iodim_print(rank, zero_dims));
		fprintf(stderr, "howmany_rank := %d, howmany_dims := %s, ptr := IntHexString(\"%p\")),\n",
			howmany_rank, iodim_print(howmany_rank, howmany_zero_dims), tmp);
	}
	return tmp;
}


// will return an illegal pointer (a used as handle)

fftx_real *fftx_create_temp_real(int rank, fftx_iodim *dims) {

	fftx_real *tmp;
	int tmpsize = 1;
	int i;

	for (i = 0; i < rank; i++)
		tmpsize *= dims[i].n;

	tmp = fftx_pointer2handle_real((fftx_real *)fftx_internal_malloc(tmpsize * sizeof(fftx_real)));

	if (fftx_global_flags & FFTX_MODE_OBSERVE) {
		fprintf(stderr, "\trec(op := \"fftx_create_temp_real\", rank := %d, dims := %s, ptr := IntHexString(\"%p\")),\n",
			rank, iodim_print(rank, dims), tmp);
	}
	return tmp;
}

fftx_real *fftx_create_temp_real_b(int rank, fftx_iodim *dims, int howmany_rank, fftx_iodim *howmany_dims) {

	fftx_real *tmp;
	int tmpsize = 1;
	int i;

	for (i = 0; i < rank; i++)
		tmpsize *= dims[i].n;

	for (i = 0; i < howmany_rank; i++)
		tmpsize *= howmany_dims[i].n;

	tmp = fftx_pointer2handle_real((fftx_real *)fftx_internal_malloc(tmpsize * sizeof(fftx_real)));

	if (fftx_global_flags & FFTX_MODE_OBSERVE) {
		fprintf(stderr, "\trec(op := \"fftx_create_temp_real\", rank := %d, dims := %s,\n",
			rank, iodim_print(rank, dims));
		fprintf(stderr, "howmany_rank := %d, howmany_dims := %s, ptr := IntHexString(\"%p\")),\n",
			howmany_rank, iodim_print(howmany_rank, howmany_dims), tmp);
	}
	return tmp;
}


// will return an illegal pointer (a used as handle)

fftx_complex *fftx_create_zero_temp_complex(int rank, fftx_iodim *zero_dims) {

	fftx_complex *tmp;
	int tmpsize = 1;
	int i;

	for (i = 0; i < rank; i++)
		tmpsize *= zero_dims[i].n;

	tmp = (fftx_complex *)fftx_internal_malloc(tmpsize * sizeof(fftx_complex));
	for (i = 0; i < tmpsize; i++) {
		tmp[i][0] = 0.0;
		tmp[i][1] = 0.0;
	}
	tmp = fftx_pointer2handle_complex(tmp);

	if (fftx_global_flags & FFTX_MODE_OBSERVE) {
		fprintf(stderr, "\trec(op := \"fftx_create_zero_temp_complex\", rank := %d, dims := %s, ptr := IntHexString(\"%p\")),\n",
			rank, iodim_print(rank, zero_dims), tmp);
	}
	return tmp;
}

fftx_complex *fftx_create_zero_temp_complex_b(int rank, fftx_iodim *zero_dims, int howmany_rank, fftx_iodim *howmany_zero_dims) {

	fftx_complex *tmp;
	int tmpsize = 1;
	int i;

	for (i = 0; i < rank; i++)
		tmpsize *= zero_dims[i].n;

	for (i = 0; i < howmany_rank; i++)
		tmpsize *= howmany_zero_dims[i].n;

	tmp = (fftx_complex *)fftx_internal_malloc(tmpsize * sizeof(fftx_complex));
	for (i = 0; i < tmpsize; i++) {
		tmp[i][0] = 0.0;
		tmp[i][1] = 0.0;
	}

	tmp = fftx_pointer2handle_complex(tmp);

	if (fftx_global_flags & FFTX_MODE_OBSERVE) {
		fprintf(stderr, "\trec(op := \"fftx_create_zero_temp_complex\", rank := %d, dims := %s,\n",
			rank, iodim_print(rank, zero_dims));
		fprintf(stderr, "howmany_rank := %d, howmany_dims := %s, ptr := IntHexString(\"%p\")),\n",
			howmany_rank, iodim_print(howmany_rank, howmany_zero_dims), tmp);
	}
	return tmp;
}


// will return an illegal pointer (a used as handle)

fftx_complex *fftx_create_temp_complex(int rank, fftx_iodim *dims) {

	fftx_complex *tmp;
	int tmpsize = 1;
	int i;

	for (i = 0; i < rank; i++)
		tmpsize *= dims[i].n;

	tmp = fftx_pointer2handle_complex((fftx_complex *)fftx_internal_malloc(tmpsize * sizeof(fftx_complex)));

	if (fftx_global_flags & FFTX_MODE_OBSERVE) {
		fprintf(stderr, "\trec(op := \"fftx_create_temp_complex\", rank := %d, dims := %s, ptr := IntHexString(\"%p\")),\n",
			rank, iodim_print(rank, dims), tmp);
	}
	return tmp;
}

fftx_complex *fftx_create_temp_complex_b(int rank, fftx_iodim *dims, int howmany_rank, fftx_iodim *howmany_dims) {

	fftx_complex *tmp;
	int tmpsize = 1;
	int i;

	for (i = 0; i < rank; i++)
		tmpsize *= dims[i].n;

	for (i = 0; i < howmany_rank; i++)
		tmpsize *= howmany_dims[i].n;

	tmp = fftx_pointer2handle_complex((fftx_complex *)fftx_internal_malloc(tmpsize * sizeof(fftx_complex)));

	if (fftx_global_flags & FFTX_MODE_OBSERVE) {
		fprintf(stderr, "\trec(op := \"fftx_create_temp_complex\", rank := %d, dims := %s,\n",
			rank, iodim_print(rank, dims));
		fprintf(stderr, "howmany_rank := %d, howmany_dims := %s, ptr := IntHexString(\"%p\")),\n",
			howmany_rank, iodim_print(howmany_rank, howmany_dims), tmp);
	}
	return tmp;
}


// FFTX malloc and free
void *fftx_malloc(unsigned size) {
	void *p;
#ifdef USE_CUFFTW
	cudaMallocManaged(&p, size);
#else
	p = fftw_malloc(size);
#endif
	if (fftx_global_flags & FFTX_MODE_OBSERVE)
		fprintf(stderr, "rec(op := \"fftx_malloc\", ptr := IntHexString(\"%p\"), size := %u)\n", p, size);
	return p;
}

void fftx_free(void *p) {
#ifdef USE_CUFFTW
	cudaFree(p);
#else
	fftw_free(p);
#endif
}

// internal malloc and free
void *fftx_internal_malloc(unsigned size) {
	void *p;
#ifdef USE_CUFFTW
	cudaMallocManaged(&p, size);
#else
	p = fftw_malloc(size);
#endif

	return p;
}

void fftx_internal_free(void *p) {
#ifdef USE_CUFFTW
	cudaFree(p);
#else
	fftw_free(p);
#endif
}


// create a persistent handle for a given plan
fftx_plan_label fftx_make_persistent_plan(fftx_plan_label h, fftx_plan p) {
	if (fftx_global_flags & FFTX_MODE_OBSERVE)
		fprintf(stderr, "\trec(op := \"fftx_make_persistent_plan\", ptr := IntHexString(\"%p\"), handle := IntHexString(\"%X\")),\n", p, h);
	return h;
}


// turn a persistent handle into an executable plan
fftx_plan fftx_plan_from_persistent_label(fftx_plan_label h, fftx_real **ptrs, unsigned flags) {
	fftx_plan p = (fftx_plan)fftx_internal_malloc(sizeof(fftx_plan_s));
	p->type = FFFTX_PLAN_PERSISTENT_HANDLE;
	p->plan.persist.flags = flags;
	p->plan.persist.persistent_handle = h;
	p->plan.persist.ptrs = ptrs;
	return p;
}

