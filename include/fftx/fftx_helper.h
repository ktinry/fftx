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
 
#ifndef __FFTX_HELPER_H__
#define __FFTX_HELPER_H__

#include <stdlib.h>

/*
 * Fastest to slowest varying: X, Y, Z
 */

// N-dimensional static array
#define FFTX_VEC(_d0, ...) ( (int[]){ _d0, __VA_ARGS__ } )

// Dimension layout macros. X is assumed as the fastest varying dimension while Z as the slowest.
#define FFTX_ORDER_XYZ ( (int[]) {2, 1, 0} )
#define FFTX_ORDER_XZY ( (int[]) {1, 2, 0} )
#define FFTX_ORDER_YXZ ( (int[]) {2, 0, 1} )
#define FFTX_ORDER_YZX ( (int[]) {0, 2, 1} )
#define FFTX_ORDER_ZXY ( (int[]) {1, 0, 2} )
#define FFTX_ORDER_ZYX ( (int[]) {0, 1, 2} )

// Default orders (including 2D below) can be redefined in the app
#define DEFAULT_3D_ORDER FFTX_ORDER_XYZ

#define FFTX_ORDER_XY ( (int[]) {1, 0} )
#define FFTX_ORDER_YX ( (int[]) {0, 1} )

#define DEFAULT_2D_ORDER FFTX_ORDER_XY

#define FFTX_ORDER_X 0

#define DEFAULT_1D_ORDER FFTX_ORDER_X

/*
 * API for definition of geometrical objects.
 * Whenever a FFTX_VEC position refers to a dimension the order assumed is fastest to slowest (X to Z).
 */

/* 
 * 3D geometrical interface
 * _name: name of the 3D box
 * _dim, _d0, _d1, _d2:  scalar size
 * _d, _o, _s: dimensions, origin, and strides passed as a 3D FFTX_VEC
 */
#define FFTX_CUBE(_name, _dim)                              FFTX_3D_BOX_EXTDEF(_name, FFTX_VEC(_dim, _dim, _dim), _FFTX_0VEC_3D, _FFTX_1VEC_3D)
#define FFTX_CUBE_ORIGIN(_name, _dim, _o)                   FFTX_3D_BOX_EXTDEF(_name, FFTX_VEC(_dim, _dim, _dim), _o, _FFTX_1VEC_3D)
#define FFTX_CUBE_STRIDE(_name, _dim, _s)                   FFTX_3D_BOX_EXTDEF(_name, FFTX_VEC(_dim, _dim, _dim), _FFTX_0VEC_3D, _s)
#define FFTX_3D_BOX(_name, _d0, _d1, _d2) 			        FFTX_3D_BOX_EXTDEF(_name, FFTX_VEC(_d0, _d1, _d2), _FFTX_0VEC_3D, _FFTX_1VEC_3D)
#define FFTX_3D_BOX_ORIGIN(_name, _d0, _d1, _d2, _o)        FFTX_3D_BOX_EXTDEF(_name, FFTX_VEC(_d0, _d1, _d2), _o, _FFTX_1VEC_3D)
#define FFTX_3D_BOX_STRIDE(_name, _d0, _d1, _d2, _s)        FFTX_3D_BOX_EXTDEF(_name, FFTX_VEC(_d0, _d1, _d2), _FFTX_0VEC_3D, _s)
#define FFTX_3D_BOX_EXTDEF(_name, _d, _o, _s)  \
	FFTX_3D_BOX_FULLDEF(_name, _d, _o, _s, \
		FFTX_VEC(FFTX_1D_SIZE(_d[0], _o[0], _s[0]), FFTX_1D_SIZE(_d[1], _o[1], _s[1]), FFTX_1D_SIZE(_d[2], _o[2], _s[2])) )

/*
 * Creates a box with origin in _o, dimenions _d, and leading dimensions _l (All 3D FFTX_VECs).
 * The actual stride for a given dimension _i_ is the product of _s(_i_) and the leading dimensions for _j_ < _i_.
 * The 4th array is a support array that carries the full stride for the entire box.
 */
#define FFTX_3D_BOX_FULLDEF(_name, _d, _o, _s, _l) \
	FFTX_BOX_ACC(_name, { {_d[0], _o[0], _s[0], _s[0], _l[0]}, {_d[1], _o[1], _l[0]*_s[1], _s[1], _l[1]}, {_d[2], _o[2], _l[0]*_l[1]*_s[2], _s[2], _l[2]}, \
		{_l[0]*_l[1]*_l[2], 0, 0, 0, 0} } )

/* 
 * 2D geometrical interface
 * _name: name of the 2D box
 * _dim, _d0, _d1:  scalar size
 * _d, _o, _s, _l: dimensions, origin, strides, and leading dimensions passed as a 2D FFTX_VEC
 */
#define FFTX_SQUARE(_name, _dim)     FFTX_2D_BOX_EXTDEF(_name, FFTX_VEC(_dim, _dim), _FFTX_0VEC_2D, _FFTX_1VEC_2D)
#define FFTX_2D_BOX(_name, _d0, _d1) FFTX_2D_BOX_EXTDEF(_name, FFTX_VEC(_d0, _d1), _FFTX_0VEC_2D, _FFTX_1VEC_2D)

#define FFTX_2D_BOX_EXTDEF(_name, _d, _o, _s) \
	FFTX_2D_BOX_FULLDEF(_name, _d, _o, _s, \
		FFTX_VEC(FFTX_1D_SIZE(_d[0], _o[0], _s[0]), FFTX_1D_SIZE(_d[1], _o[1], _s[1])) )

// See comments for 3D version
#define FFTX_2D_BOX_FULLDEF(_name, _d, _o, _s, _l) \
	FFTX_BOX_ACC(_name, { {_d[0], _o[0], _s[0], _s[0], _l[0]}, {_d[1], _o[1], _l[0]*_s[1], _s[1], _l[1]}, \
		{_l[0]*_l[1], 0, 0, 0, 0} } )

/* 
 * 1D geometrical interface
 * _name: name of the line
 * _d, _o, _s, _l: scalar dimension, origin, stride, and leading dimension
 */
#define FFTX_LINE(_name, _dim)                   FFTX_LINE_EXTDEF(_name, _dim, _FFTX_0VEC_1D, _FFTX_1VEC_1D)
#define FFTX_LINE_EXTDEF(_name, _d, _o, _s)      FFTX_LINE_FULLDEF(_name, _d, _o, _s, FFTX_1D_SIZE(_d, _o, _s))
#define FFTX_LINE_FULLDEF(_name, _d, _o, _s, _l) FFTX_BOX_ACC(_name, { {_d, _o, _s, _s, _l}, {_l, 0, 0, 0, 0} })

/*
 * Embedded geometry. The macro below produce fftx_iodimx's. All FFTX_VEC should be given in X,Y,Z order.
 */

/* 
 * 1-to-1 mapping bitween I and O boxes of same structure.
 * _name: of the fftx_iodimx
 * _rank: rank of I and O boxes
 * _b:    Box description obtained with the geometrical API.
 */
#define FFTX_IO_BOX(_name, _rank, _b) \
			FFTX_EMBED_BOX_BOX(_name, _rank, _b, _b, _FFTX_0VEC_##_rank##D)

/* 
 * 1-to-1 mapping bitween I, O, and D boxes of same structure.
 * _name: of the fftx_iodimx
 * _rank: rank of I and O boxes
 * _b:    Box description obtained with the geometrical API.
 */
#define FFTX_IOD_BOX(_name, _rank, _b) \
			FFTX_IOD_DIMX_3D_FULLDEF(_name, _FFTX_BOX_N_TO_VEC(_rank, _b), \
				_b, _FFTX_1VEC_##_rank##D, DEFAULT_##_rank##D_ORDER, _b, _FFTX_1VEC_##_rank##D, DEFAULT_##_rank##D_ORDER, _b, _FFTX_1VEC_##_rank##D, DEFAULT_##_rank##D_ORDER )

/* 
 * Embed box _bi into _bo.
 * _name: of the fftx_iodimx
 * _rank: rank of I and O boxes
 * _bi, _bo: Box descriptions obtained with the geometrical API.
 * _oorig: FFTX_VEC position within _bo where to embed _bi.
 * _iorder, _oorder: Dimension layout specification when different than DEFAULT_<2,3>D_ORDER.
 */
#define FFTX_EMBED_BOX_BOX(_name, _rank, _bi, _bo, _oorig) \
			FFTX_EMBED_BOX_BOX_##_rank##D_DIMDIR(_name, _bi, _FFTX_1VEC_##_rank##D, DEFAULT_##_rank##D_ORDER, \
				_bo, _FFTX_1VEC_##_rank##D, DEFAULT_##_rank##D_ORDER, _oorig)
#define FFTX_EMBED_BOX_BOX_ORDER(_name, _rank, _bi, _iorder, _bo, _oorder, _oorig) \
			FFTX_EMBED_BOX_BOX_##_rank##D_DIMDIR(_name, _bi, _FFTX_1VEC_##_rank##D, _iorder, \
				_bo, _FFTX_1VEC_##_rank##D, _oorder, _oorig)

/* 
 * Embed box _bi into _bo.
 * _name: of the fftx_iodimx
 * _rank: rank of I and O boxes
 * _bi, _bo: Box descriptions obtained with the geometrical API.
 * _oorig: FFTX_VEC position within _bo where to embed _bi.
 * _idir, _odir: FFTX_VEC vector specifying in which direction the box's dimensions should be traversed.
 *               if ith position is 1 dimesion is traversed in increasing order, if -1 in reverse (decreasing) order.
 * _iorder, _oorder: Dimension layout specification when different than DEFAULT_<2,3>D_ORDER.
 */
#define FFTX_EMBED_BOX_BOX_EXTDEF(_name, _rank, _bi, _dim, _iorig, _idir, _bo, _oorig, _odir) \
			FFTX_EMBED_BOX_BOX_##_rank##D_FULLDEF(_name, _bi, _dim, _iorig, _idir, DEFAULT_##_rank##D_ORDER, _bo, _oorig, _odir, DEFAULT_##_rank##D_ORDER);

#define FFTX_EMBED_BOX_BOX_FULLDEF(_name, _rank, _bi, _dim, _iorig, _idir, _iorder, _bo, _oorig, _odir, _oorder) \
			FFTX_EMBED_BOX_BOX_##_rank##D_FULLDEF(_name, _bi, _dim, _iorig, _idir, _iorder, _bo, _oorig, _odir, _oorder);

/* 
 * Extract box _bi from box _bo.
 */

#define FFTX_EXTRACT_BOX_BOX(_name, _rank, _bi, _bo, _iorig) \
			FFTX_EXTRACT_BOX_BOX_##_rank##D_DIMDIR(_name, _bi, _FFTX_1VEC_##_rank##D, DEFAULT_##_rank##D_ORDER, \
				_bo, _FFTX_1VEC_##_rank##D, DEFAULT_##_rank##D_ORDER, _iorig)
#define FFTX_EXTRACT_BOX_BOX_ORDER(_name, _rank, _bi, _iorder, _bo, _oorder, _iorig) \
			FFTX_EXTRACT_BOX_BOX_##_rank##D_DIMDIR(_name, _bi, _FFTX_1VEC_##_rank##D, _iorder, \
				_bo, _FFTX_1VEC_##_rank##D, _oorder, _iorig)

#define FFTX_EXTRACT_BOX_BOX_EXTDEF(_name, _rank, _bi, _dim, _iorig, _idir, _bo, _oorig, _odir) \
			FFTX_EMBED_BOX_BOX_##_rank##D_FULLDEF(_name, _bi, _dim, _iorig, _idir, DEFAULT_##_rank##D_ORDER, _bo, _oorig, _odir, DEFAULT_##_rank##D_ORDER);

#define FFTX_EXTRACT_BOX_BOX_FULLDEF(_name, _rank, _bi, _dim, _iorig, _idir, _iorder, _bo, _oorig, _odir, _oorder) \
			FFTX_EMBED_BOX_BOX_##_rank##D_FULLDEF(_name, _bi, _dim, _iorig, _idir, _iorder, _bo, _oorig, _odir, _oorder)

/*
 * Create fftx_iodimx that to express multiple multiple (batched) passes
 * through IO(D) boxes.
 * 
 */

#define FFTX_IO_BOX_BATCH(_name, _brank, _bsize, _rank, _iob) \
			FFTX_IO_BATCH(_name, _brank, _bsize, _rank, _iob, _iob)

#define FFTX_IO_BOX_SHIFT_BATCH(_name, _bsize, _n_shift, _sh, _rank, _iob) \
			FFTX_IO_SHIFT_BATCH(_name, _bsize, _n_shift, _sh, _sh, _rank, _iob, _iob);

#define FFTX_IO_BATCH(_name, _brank, _bsize, _rank, _in, _out) \
			_FFTX_0DIM_##_rank##D_BOX(_d##_name); \
			FFTX_##_brank##D_IOD_BATCH(_name, _bsize, _rank, _in, _out, _d##_name)

#define FFTX_IOD_BATCH(_name, _brank, _bsize, _rank, _in, _out, _data) \
			FFTX_##_brank##D_IOD_BATCH(_name, _bsize, _rank, _in, _out, _data)

#define FFTX_IO_SHIFT_BATCH(_name, _bsize, _n_shift, _ish, _osh, _rank, _in, _out) \
			_FFTX_0DIM_##_rank##D_BOX(_d##_name); \
			FFTX_IOD_SHIFT_BATCH(_name, _bsize, _n_shift, _ish, _osh, 0, _rank, _in, _out, _d##_name);

#define FFTX_IOD_SHIFT_BATCH(_name, _bsize, _n_shift, _ish, _osh, _dsh, _rank, _in, _out, _data) \
			fftx_iodimx _name[] = \
			{ {_bsize, _FFTX_0VEC_1D, _FFTX_0VEC_1D, _FFTX_0VEC_1D, _in[_rank][0], _out[_rank][0], _data[_rank][0] }, \
			  {_n_shift, _FFTX_0VEC_1D, _FFTX_0VEC_1D, _FFTX_0VEC_1D, _ish, _osh, _dsh } };


//==============================================================================================================

#define FFTX_BOX_ACC(_name, ...) int _name[][5] = __VA_ARGS__ ;

#define _FFTX_1VEC_3D FFTX_VEC(1, 1, 1)
#define _FFTX_1VEC_2D FFTX_VEC(1, 1)
#define _FFTX_1VEC_1D 1

#define _FFTX_0VEC_3D FFTX_VEC(0, 0, 0)
#define _FFTX_0VEC_2D FFTX_VEC(0, 0)
#define _FFTX_0VEC_1D 0

#define _FFTX_0DIM_3D_BOX(_name) FFTX_BOX_ACC(_name, {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}});
#define _FFTX_0DIM_2D_BOX(_name) FFTX_BOX_ACC(_name, {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}});
#define _FFTX_0DIM_1D_BOX(_name) FFTX_BOX_ACC(_name, {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}});

#define _MAX(_a, _b) ( ((_a)>(_b)) ? (_a) : (_b) )
#define FFTX_1D_SIZE(_d, _o, _s) ( _o*abs(_s) + _MAX((_d) > 0 ? abs(_s) : 0, _d*_s) )

#define FFTX_3D_BOX_PLAINDEF(_name, _d, _o, _s) \
	FFTX_BOX_ACC(_name, { {_d[0], _o[0], _s[0], 0, 0}, {_d[1], _o[1], _s[1], 0, 0}, {_d[2], _o[2], _s[2], 0, 0} } )

#define FFTX_2D_BOX_PLAINDEF(_name, _d, _o, _s) \
	FFTX_BOX_ACC(_name, { {_d[0], _o[0], _s[0], 0, 0}, {_d[1], _o[1], _s[1], 0, 0} } )

/*
 * FFTX_IO_DIM
 */

#define FFTX_IO_DIM(_name, _rank, _in, _out) FFTX_IO_DIM_##_rank##D_NFROMI(_name, _in, DEFAULT_##_rank##D_ORDER, _out, DEFAULT_##_rank##D_ORDER)
#define FFTX_IO_DIM_ORDER(_name, _rank, _in, _iorder, _out, _oorder) FFTX_IO_DIM_##_rank##D_NFROMI(_name, _in, _iorder, _out, _oorder)

#define FFTX_IO_DIM_3D_NFROMI(_name, _in, _iorder, _out, _oorder) FFTX_IO_DIM_3D_FULLDEF(_name, FFTX_VEC(_in[0][0], _in[1][0], _in[2][0]), _in, _iorder, _out, _oorder)
#define FFTX_IO_DIM_3D_FULLDEF(_name, _n, _in, _iorder, _out, _oorder) fftx_iodim _name[] = \
			{ {_n[_iorder[0]], _in[_iorder[0]][2], _out[_oorder[0]][2] }, \
			  {_n[_iorder[1]], _in[_iorder[1]][2], _out[_oorder[1]][2] }, \
			  {_n[_iorder[2]], _in[_iorder[2]][2], _out[_oorder[2]][2] } };

#define FFTX_IO_DIM_2D_NFROMI(_name, _in, _iorder, _out, _oorder) FFTX_IO_DIM_2D_FULLDEF(_name, FFTX_VEC(_in[0][0], _in[1][0]), _in, _iorder, _out, _oorder)
#define FFTX_IO_DIM_2D_FULLDEF(_name, _n, _in, _iorder, _out, _oorder) fftx_iodim _name[] = \
			{ {_n[_iorder[0]], _in[_iorder[0]][2], _out[_oorder[0]][2] }, \
			  {_n[_iorder[1]], _in[_iorder[1]][2], _out[_oorder[1]][2] } }

#define FFTX_IO_DIM_1D_NFROMI(_name, _in, _iorder, _out, _oorder) FFTX_IO_DIM_1D_FULLDEF(_name, _in[0][0], _iorder, _out, _oorder)
#define FFTX_IO_DIM_1D_FULLDEF(_name, _n, _in, _iorder, _out, _oorder) fftx_iodim _name[] = \
			{ { _n, _in[0][2], _out[0][2] } }

/*
 * FFTX_IO(D)_DIMX
 */

#define FFTX_IO_DIMX(_name, _rank, _in, _out) \
			_FFTX_0DIM_##_rank##D_BOX(_d##name); \
			FFTX_IOD_DIMX(_name, _rank, _in, _out, _d##name);
#define FFTX_IO_DIMX_ORDER(_name, _rank, _in, _iorder, _out, _oorder) \
			_FFTX_0DIM_##_rank##D_BOX(_d##name); \
			FFTX_IOD_DIMX_ORDER(_name, _rank, _in, _iorder, _out, _oorder, _d##name, DEFAULT_##_rank##D_ORDER);

#define FFTX_IOD_DIMX(_name, _rank, _in, _out, _data) FFTX_IOD_DIMX_##_rank##D_NFROMI(_name, _in, DEFAULT_##_rank##D_ORDER, _out, DEFAULT_##_rank##D_ORDER, _data, DEFAULT_##_rank##D_ORDER);
#define FFTX_IOD_DIMX_ORDER(_name, _rank, _in, _iorder, _out, _oorder, _data, _dorder) FFTX_IOD_DIMX_##_rank##D_NFROMI(_name, _in, _iorder, _out, _oorder, _data, _dorder);

#define FFTX_IOD_DIMX_3D_NFROMI(_name, _in, _iorder, _out, _oorder, _data, _dorder) \
			FFTX_IOD_DIMX_3D_FULLDEF(_name, FFTX_VEC(_in[_iorder[0]][0],_in[_iorder[1]][0],_in[_iorder[2]][0]), \
				_in, _FFTX_1VEC_3D, _iorder, _out, _FFTX_1VEC_3D, _oorder, _data, _FFTX_1VEC_3D, _dorder);
#define FFTX_IOD_DIMX_2D_NFROMI(_name, _in, _iorder, _out, _oorder, _data, _dorder) \
			FFTX_IOD_DIMX_2D_FULLDEF(_name, FFTX_VEC(_in[_iorder[0]][0],_in[_iorder[1]][0]), \
				_in, _FFTX_1VEC_2D, _iorder, _out, _FFTX_1VEC_2D, _oorder, _data, _FFTX_1VEC_2D, _dorder);
#define FFTX_IOD_DIMX_1D_NFROMI(_name, _in, _iorder, _out, _oorder, _data, _dorder) \
			FFTX_IOD_DIMX_1D_FULLDEF(_name, _in[0][0], _in, _FFTX_1VEC_1D, _iorder, _out, _FFTX_1VEC_1D, _oorder, _data, _FFTX_1VEC_1D, _dorder);

// Directions: First accessed -> last accessed dimension
#define FFTX_IOD_DIMX_3D_FULLDEF(_name, _n, _in, _idir, _iorder, _out, _odir, _oorder, _data, _ddir, _dorder) \
			fftx_iodimx _name[] = \
			{ {_n[0], _in[_iorder[0]][1], _out[_oorder[0]][1], _data[_dorder[0]][1], _idir[_iorder[0]]*_in[_iorder[0]][2], _odir[_oorder[0]]*_out[_oorder[0]][2], _ddir[_dorder[0]]*_data[_dorder[0]][2] }, \
			  {_n[1], _in[_iorder[1]][1], _out[_oorder[1]][1], _data[_dorder[1]][1], _idir[_iorder[1]]*_in[_iorder[1]][2], _odir[_oorder[1]]*_out[_oorder[1]][2], _ddir[_dorder[1]]*_data[_dorder[1]][2] }, \
			  {_n[2], _in[_iorder[2]][1], _out[_oorder[2]][1], _data[_dorder[2]][1], _idir[_iorder[2]]*_in[_iorder[2]][2], _odir[_oorder[2]]*_out[_oorder[2]][2], _ddir[_dorder[2]]*_data[_dorder[2]][2] } };

#define FFTX_IOD_DIMX_2D_FULLDEF(_name, _n, _in, _idir, _iorder, _out, _odir, _oorder, _data, _ddir, _dorder) \
			fftx_iodimx _name[] = \
			{ {_n[0], _in[_iorder[0]][1], _out[_oorder[0]][1], _data[_dorder[0]][1], _idir[_iorder[0]]*_in[_iorder[0]][2], _odir[_oorder[0]]*_out[_oorder[0]][2], _ddir[_dorder[0]]*_data[_dorder[0]][2] }, \
			  {_n[1], _in[_iorder[1]][1], _out[_oorder[1]][1], _data[_dorder[1]][1], _idir[_iorder[1]]*_in[_iorder[1]][2], _odir[_oorder[1]]*_out[_oorder[1]][2], _ddir[_dorder[1]]*_data[_dorder[1]][2] } };

#define FFTX_IOD_DIMX_1D_FULLDEF(_name, _n, _in, _idir, _iorder, _out, _odir, _oorder, _data, _ddir, _dorder) \
			fftx_iodimx _name[] = \
			{ {_n, _in[0][1], _out[0][1], _data[0][1], _idir*_in[0][2], _odir*_out[0][2], _ddir*_data[0][2] } };




#define FFTX_1D_IOD_BATCH(_name, _bsize, _rank, _in, _out, _data) \
			fftx_iodimx _name[] = \
			{ {_bsize, _FFTX_0VEC_1D, _FFTX_0VEC_1D, _FFTX_0VEC_1D, _in[_rank][0], _out[_rank][0], _data[_rank][0] } };




#define _FFTX_BOX_N_TO_VEC(_rank, _b) _FFTX_##_rank##D_BOX_N_TO_VEC(_b)
#define _FFTX_3D_BOX_N_TO_VEC(_b) FFTX_VEC(_b[0][0], _b[1][0], _b[2][0])
#define _FFTX_2D_BOX_N_TO_VEC(_b) FFTX_VEC(_b[0][0], _b[1][0])
#define _FFTX_1D_BOX_N_TO_VEC(_b) _b[0][0]

#define FFTX_EMBED_BOX_BOX_3D_DIMDIR(_name, _bi, _idir, _iorder, _bo, _odir, _oorder, _oorig) \
			FFTX_3D_BOX_PLAINDEF(_o##_name, FFTX_VEC(_bo[0][0], _bo[1][0], _bo[2][0]), \
				FFTX_VEC(_bo[0][1]+_oorig[0], _bo[1][1]+_oorig[1], _bo[2][1]+_oorig[2]), \
				FFTX_VEC(_bo[0][2], _bo[1][2], _bo[2][2])); \
			_FFTX_0DIM_3D_BOX(_d##_name); \
			FFTX_IOD_DIMX_3D_FULLDEF(_name, FFTX_VEC(_bi[_iorder[0]][0], _bi[_iorder[1]][0], _bi[_iorder[2]][0]), _bi, _idir, _iorder, \
				_o##_name, _odir, _oorder, _d##_name, _FFTX_1VEC_3D, DEFAULT_3D_ORDER);
#define FFTX_EMBED_BOX_BOX_2D_DIMDIR(_name, _bi, _idir, _iorder, _bo, _odir, _oorder, _oorig) \
			FFTX_2D_BOX_PLAINDEF(_o##_name, FFTX_VEC(_bo[0][0], _bo[1][0]), \
				FFTX_VEC(_bo[0][1]+_oorig[0], _bo[1][1]+_oorig[1]), \
				FFTX_VEC(_bo[0][2], _bo[1][2])); \
			_FFTX_0DIM_2D_BOX(_d##_name); \
			FFTX_IOD_DIMX_2D_FULLDEF(_name, FFTX_VEC(_bi[_iorder[0]][0], _bi[_iorder[1]][0]), _bi, _idir, _iorder, \
				_d##_name, _odir, _oorder, _d##_name, _FFTX_1VEC_2D, DEFAULT_2D_ORDER);
#define FFTX_EMBED_BOX_BOX_1D_DIMDIR(_name, _bi, _idir, _iorder, _bo, _odir, _oorder, _oorig) \
			FFTX_LINE_FULLDEF(_o##_name, _bo[0][0], _bo[0][1]+_oorig, _bo[0][2]); \
			_FFTX_0DIM_1D_BOX(_d##_name); \
			FFTX_IOD_DIMX_1D_FULLDEF(_name, _bi[0][0], _bi, _idir, _iorder, \
				_o##_name, _odir, _oorder, _d##_name, _FFTX_1VEC_1D, DEFAULT_1D_ORDER);

#define FFTX_EMBED_BOX_BOX_3D_FULLDEF(_name, _bi, _dim, _iorig, _idir, _iorder, _bo, _oorig, _odir, _oorder) \
			FFTX_3D_BOX_FULLDEF(_i##_name, FFTX_VEC(_bi[0][0], _bi[1][0], _bi[2][0]), FFTX_VEC(_bi[0][1]+_iorig[0], _bi[1][1]+_iorig[1], _bi[2][1]+_iorig[2]), \
				FFTX_VEC(_bi[0][3], _bi[1][3], _bi[2][3]), \
				FFTX_VEC(_bi[0][4], _bi[1][4], _bi[2][4])); \
			FFTX_3D_BOX_FULLDEF(_o##_name, FFTX_VEC(_bo[0][0], _bo[1][0], _bo[2][0]), FFTX_VEC(_bo[0][1]+_oorig[0], _bo[1][1]+_oorig[1], _bo[2][1]+_oorig[2]), \
				FFTX_VEC(_bo[0][3], _bo[1][3], _bo[2][3]), \
				FFTX_VEC(_bo[0][4], _bo[1][4], _bo[2][4])); \
			_FFTX_0DIM_3D_BOX(_d##_name); \
			FFTX_IOD_DIMX_3D_FULLDEF(_name, FFTX_VEC(_dim[_iorder[0]], _dim[_iorder[1]], _dim[_iorder[2]]), _i##_name, _idir, _iorder, \
				_o##_name, _odir, _oorder, _d##_name, _FFTX_1VEC_3D, DEFAULT_3D_ORDER);




#define FFTX_EXTRACT_BOX_BOX_3D_DIMDIR(_name, _bi, _idir, _iorder, _bo, _odir, _oorder, _iorig) \
			FFTX_3D_BOX_PLAINDEF(_i##_name, FFTX_VEC(_bi[0][0], _bi[1][0], _bi[2][0]), \
				FFTX_VEC(_bi[0][1]+_iorig[0], _bi[1][1]+_iorig[1], _bi[2][1]+_iorig[2]), \
				FFTX_VEC(_bi[0][2], _bi[1][2], _bi[2][2])); \
			_FFTX_0DIM_3D_BOX(_d##_name); \
			FFTX_IOD_DIMX_3D_FULLDEF(_name, FFTX_VEC(_bo[_iorder[0]][0], _bo[_iorder[1]][0], _bo[_iorder[2]][0]), \
				_i##_name, _idir, _iorder, _bo, _odir, _oorder, _d##_name, _FFTX_1VEC_3D, DEFAULT_3D_ORDER);
#define FFTX_EXTRACT_BOX_BOX_2D_DIMDIR(_name, _bi, _idir, _iorder, _bo, _odir, _oorder, _iorig) \
			FFTX_2D_BOX_PLAINDEF(_i##_name, FFTX_VEC(_bi[0][0], _bi[1][0]), \
				FFTX_VEC(_bi[0][1]+_iorig[0], _bi[1][1]+_iorig[1]), \
				FFTX_VEC(_bi[0][2], _bi[1][2])); \
			_FFTX_0DIM_2D_BOX(_d##_name); \
			FFTX_IOD_DIMX_2D_FULLDEF(_name, FFTX_VEC(_bo[_iorder[0]][0], _bo[_iorder[1]][0]), \
				_i##_name, _idir, _iorder, _bo, _odir, _oorder, _d##_name, _FFTX_1VEC_2D, DEFAULT_2D_ORDER );
#define FFTX_EXTRACT_BOX_BOX_1D_DIMDIR(_name, _bi, _idir, _iorder, _bo, _odir, _oorder, _iorig) \
			FFTX_LINE_EXTDEF(_i##_name, _bi[0][0], _bi[0][1]+_iorig[0], _bi[0][2]); \
			_FFTX_0DIM_1D_BOX(_d##_name); \
			FFTX_IOD_DIMX_1D_FULLDEF(_name, _bo[0][0], \
				_i##_name, _idir, _iorder, _bo, _odir, _oorder, _d##_name, _FFTX_1VEC_1D, DEFAULT_1D_ORDER );


#endif