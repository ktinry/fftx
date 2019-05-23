# FFTX Copyright (c) 2019, The Regents of the University of California, through 
# Lawrence Berkeley National Laboratory (subject to receipt of any required 
# approvals from the U.S. Dept. of Energy), Carnegie Mellon University and 
# SpiralGen, Inc.  All rights reserved.
# 
# If you have questions about your rights to use or distribute this software, 
# please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.
# 
# NOTICE.  This Software was developed under funding from the U.S. Department of
# Energy and the U.S. Government  consequently  retains certain rights. As such, 
# the U.S. Government has been granted for itself and others acting on its 
# behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software 
# to reproduce, distribute copies to the public, prepare derivative works, and 
# perform publicly and display publicly, and to permit others to do so.

#
# Look for FFTW3 in known places
#

# Set default values for top variables
if (WIN32)
	set(LOCAL_FFTW_LIB_NAME "libfftw3-3.lib")
    set(LOCAL_FFTW_INCLUDE_DIR "C:/FFTW3")
    set(LOCAL_FFTW_LIBRARY "C:/FFTW3/${LOCAL_FFTW_LIB_NAME}")
else()
    set(LOCAL_FFTW_LIB_NAME "libfftw3.so")
    set(LOCAL_FFTW_INCLUDE_DIR "" )
    set(LOCAL_FFTW_LIBRARY "fftw3")
endif()

