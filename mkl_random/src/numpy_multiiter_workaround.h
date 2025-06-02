/*
 Copyright (c) 2024, Intel Corporation

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
     * Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
     * Neither the name of Intel Corporation nor the names of its contributors
       may be used to endorse or promote products derived from this software
       without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "Python.h"
#include "numpy/arrayobject.h"

/* This header file is a work-around for issue
 *   https://github.com/numpy/numpy/issues/26990 
 *
 * It is included once in mklrandom.pyx
 * 
 * The work-around is needed to support building with 
 * NumPy < 2.0.0
 * 
 * Once building transitions to using NumPy 2.0 only
 * this file can be removed and corresponding changes
 * in mklrand.pyx can be applied to always use
 * `PyArray_MultiIter_SIZE`, PyArray_MultiIter_NDIM`,
 * and `PyArray_MultiIter_DIMS`.
 */

#if (defined(NPY_2_0_API_VERSION) && (NPY_API_VERSION >= NPY_2_0_API_VERSION))
    #define WORKAROUND_NEEDED 
#endif

#if !defined(WORKAROUND_NEEDED)
typedef struct {
    PyObject_HEAD
    int numiter;
    npy_intp size;
    npy_intp index;
    int nd;
    npy_intp dimensions[32];
    void **iters;
} multi_iter_proxy_st;
#endif

npy_intp workaround_PyArray_MultiIter_SIZE(PyArrayMultiIterObject *multi) {
#if defined(WORKAROUND_NEEDED)
    return PyArray_MultiIter_SIZE(multi);
#else
    return ((multi_iter_proxy_st *)(multi))->size;
#endif
}

int workaround_PyArray_MultiIter_NDIM(PyArrayMultiIterObject *multi) {
#if defined(WORKAROUND_NEEDED)
    return PyArray_MultiIter_NDIM(multi);
#else
    return ((multi_iter_proxy_st *)(multi))->nd;
#endif
}

npy_intp* workaround_PyArray_MultiIter_DIMS(PyArrayMultiIterObject *multi) {
#if defined(WORKAROUND_NEEDED)
    return PyArray_MultiIter_DIMS(multi);
#else
    return (((multi_iter_proxy_st *)(multi))->dimensions);
#endif
}
