#!/usr/bin/env python
# Copyright (c) 2017-2024, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# cython: language_level=3

cdef extern from "Python.h":
    void* PyMem_Malloc(size_t n)
    void PyMem_Free(void* buf)

    double PyFloat_AsDouble(object ob)
    long PyInt_AsLong(object ob)

    int PyErr_Occurred()
    void PyErr_Clear()

cdef extern from "numpy/npy_no_deprecated_api.h":
    pass

cimport numpy as cnp
from libc.string cimport memset, memcpy
cimport cpython.tuple
cimport cython

cdef extern from "math.h":
    double floor(double x)

cdef extern from "numpy/npy_math.h":
    int npy_isfinite(double x)

cdef extern from "mklrand_py_helper.h":
    object empty_py_bytes(cnp.npy_intp length, void **bytesVec)
    char* py_bytes_DataPtr(object b)
    int is_bytes_object(object b)

cdef extern from "numpy_multiiter_workaround.h":
    cnp.npy_intp cnp_PyArray_MultiIter_SIZE "workaround_PyArray_MultiIter_SIZE"(cnp.broadcast multi) nogil
    int cnp_PyArray_MultiIter_NDIM "workaround_PyArray_MultiIter_NDIM"(cnp.broadcast multi) nogil
    cnp.npy_intp* cnp_PyArray_MultiIter_DIMS "workaround_PyArray_MultiIter_DIMS"(cnp.broadcast multi) nogil

cdef extern from "randomkit.h":

    ctypedef struct irk_state:
        pass

    ctypedef enum irk_error:
        RK_NOERR = 0
        RK_ENODEV = 1
        RK_ERR_MAX = 2

    ctypedef enum irk_brng_t:
        MT19937 = 0
        SFMT19937 = 1
        WH = 2
        MT2203 = 3
        MCG31 = 4
        R250 = 5
        MRG32K3A = 6
        MCG59 = 7
        PHILOX4X32X10 = 8
        NONDETERM = 9
        ARS5 = 10

    void irk_fill(void *buffer, size_t size, irk_state *state) noexcept nogil

    void irk_dealloc_stream(irk_state *state)
    void irk_seed_mkl(irk_state * state, unsigned int seed, irk_brng_t brng, unsigned int stream_id)
    void irk_seed_mkl_array(irk_state * state, unsigned int * seed_vec, int seed_len, irk_brng_t brng, unsigned int stream_id)
    irk_error irk_randomseed_mkl(irk_state * state, irk_brng_t brng, unsigned int stream_id)
    int irk_get_stream_size(irk_state * state) noexcept nogil
    void irk_get_state_mkl(irk_state * state, char * buf)
    int irk_set_state_mkl(irk_state * state, char * buf)
    int irk_get_brng_mkl(irk_state *state) noexcept nogil
    int irk_get_brng_and_stream_mkl(irk_state *state, unsigned int * stream_id) noexcept nogil
    int irk_leapfrog_stream_mkl(irk_state *state, int k, int nstreams) noexcept nogil
    int irk_skipahead_stream_mkl(irk_state *state, long long int nskips) noexcept nogil


cdef extern from "mkl_distributions.h":
    void irk_double_vec(irk_state *state, cnp.npy_intp len, double *res) noexcept nogil
    void irk_uniform_vec(irk_state *state, cnp.npy_intp len, double *res, double dlow, double dhigh) noexcept nogil

    void irk_normal_vec_BM1(irk_state *state, cnp.npy_intp len, double *res, double mean, double sigma) noexcept nogil
    void irk_normal_vec_BM2(irk_state *state, cnp.npy_intp len, double *res, double mean, double sigma) noexcept nogil
    void irk_normal_vec_ICDF(irk_state *state, cnp.npy_intp len, double *res, double mean, double sigma) noexcept nogil

    void irk_standard_normal_vec_BM1(irk_state *state, cnp.npy_intp len, double *res) noexcept nogil
    void irk_standard_normal_vec_BM2(irk_state *state, cnp.npy_intp len, double *res) noexcept nogil
    void irk_standard_normal_vec_ICDF(irk_state *state, cnp.npy_intp len, double *res) noexcept nogil

    void irk_standard_exponential_vec(irk_state *state, cnp.npy_intp len, double *res) noexcept nogil
    void irk_exponential_vec(irk_state *state, cnp.npy_intp len, double *res, double scale) noexcept nogil

    void irk_standard_cauchy_vec(irk_state *state, cnp.npy_intp len, double *res) noexcept nogil
    void irk_standard_gamma_vec(irk_state *state, cnp.npy_intp len, double *res, double shape) noexcept nogil
    void irk_gamma_vec(irk_state *state, cnp.npy_intp len, double *res, double shape, double scale) noexcept nogil

    void irk_beta_vec(irk_state *state, cnp.npy_intp len, double *res, double p, double q) noexcept nogil

    void irk_chisquare_vec(irk_state *state, cnp.npy_intp len, double *res, double df) noexcept nogil
    void irk_standard_t_vec(irk_state *state, cnp.npy_intp len, double *res, double df) noexcept nogil

    void irk_rayleigh_vec(irk_state *state, cnp.npy_intp len, double *res, double sigma) noexcept nogil
    void irk_pareto_vec(irk_state *state, cnp.npy_intp len, double *res, double alp) noexcept nogil
    void irk_power_vec(irk_state *state, cnp.npy_intp len, double *res, double alp) noexcept nogil
    void irk_weibull_vec(irk_state *state, cnp.npy_intp len, double *res, double alp) noexcept nogil
    void irk_f_vec(irk_state *state, cnp.npy_intp len, double *res, double df_num, double df_den) noexcept nogil
    void irk_noncentral_chisquare_vec(irk_state *state, cnp.npy_intp len, double *res, double df, double nonc) noexcept nogil
    void irk_laplace_vec(irk_state *state, cnp.npy_intp len, double *res, double loc, double scale) noexcept nogil
    void irk_gumbel_vec(irk_state *state, cnp.npy_intp len, double *res, double loc, double scale) noexcept nogil
    void irk_logistic_vec(irk_state *state, cnp.npy_intp len, double *res, double loc, double scale) noexcept nogil
    void irk_wald_vec(irk_state *state, cnp.npy_intp len, double *res, double mean, double scale) noexcept nogil
    void irk_lognormal_vec_ICDF(irk_state *state, cnp.npy_intp len, double *res, double mean, double scale) noexcept nogil
    void irk_lognormal_vec_BM(irk_state *state, cnp.npy_intp len, double *res, double mean, double scale) noexcept nogil
    void irk_vonmises_vec(irk_state *state, cnp.npy_intp len, double *res, double mu, double kappa) noexcept nogil

    void irk_noncentral_f_vec(irk_state *state, cnp.npy_intp len, double *res, double df_num, double df_den, double nonc) noexcept nogil
    void irk_triangular_vec(irk_state *state, cnp.npy_intp len, double *res, double left, double mode, double right) noexcept nogil

    void irk_geometric_vec(irk_state *state, cnp.npy_intp len, int *res, double p) noexcept nogil
    void irk_negbinomial_vec(irk_state *state, cnp.npy_intp len, int *res, double a, double p) noexcept nogil
    void irk_binomial_vec(irk_state *state, cnp.npy_intp len, int *res, int n, double p) noexcept nogil
    void irk_multinomial_vec(irk_state *state, cnp.npy_intp len, int *res, int n, int d, double *pvec) noexcept nogil
    void irk_hypergeometric_vec(irk_state *state, cnp.npy_intp len, int *res, int ls, int ss, int ms) noexcept nogil

    void irk_poisson_vec_PTPE(irk_state *state, cnp.npy_intp len, int *res, double lam) noexcept nogil
    void irk_poisson_vec_POISNORM(irk_state *state, cnp.npy_intp len, int *res, double lam) noexcept nogil
    void irk_poisson_vec_V(irk_state *state, cnp.npy_intp len, int *res, double *lam_vec) noexcept nogil

    void irk_zipf_long_vec(irk_state *state, cnp.npy_intp len, long *res, double alpha) noexcept nogil
    void irk_logseries_vec(irk_state *state, cnp.npy_intp len, int *res, double theta) noexcept nogil

    # random integers madness
    void irk_discrete_uniform_vec(irk_state *state, cnp.npy_intp len, int *res, int low, int high) noexcept nogil
    void irk_discrete_uniform_long_vec(irk_state *state, cnp.npy_intp len, long *res, long low, long high) noexcept nogil
    void irk_rand_bool_vec(irk_state *state, cnp.npy_intp len, cnp.npy_bool *res, cnp.npy_bool low, cnp.npy_bool high) noexcept nogil
    void irk_rand_uint8_vec(irk_state *state, cnp.npy_intp len, cnp.npy_uint8 *res, cnp.npy_uint8 low, cnp.npy_uint8 high) noexcept nogil
    void irk_rand_int8_vec(irk_state *state, cnp.npy_intp len, cnp.npy_int8 *res, cnp.npy_int8 low, cnp.npy_int8 high) noexcept nogil
    void irk_rand_uint16_vec(irk_state *state, cnp.npy_intp len, cnp.npy_uint16 *res, cnp.npy_uint16 low, cnp.npy_uint16 high) noexcept nogil
    void irk_rand_int16_vec(irk_state *state, cnp.npy_intp len, cnp.npy_int16 *res, cnp.npy_int16 low, cnp.npy_int16 high) noexcept nogil
    void irk_rand_uint32_vec(irk_state *state, cnp.npy_intp len, cnp.npy_uint32 *res, cnp.npy_uint32 low, cnp.npy_uint32 high) noexcept nogil
    void irk_rand_int32_vec(irk_state *state, cnp.npy_intp len, cnp.npy_int32 *res, cnp.npy_int32 low, cnp.npy_int32 high) noexcept nogil
    void irk_rand_uint64_vec(irk_state *state, cnp.npy_intp len, cnp.npy_uint64 *res, cnp.npy_uint64 low, cnp.npy_uint64 high) noexcept nogil
    void irk_rand_int64_vec(irk_state *state, cnp.npy_intp len, cnp.npy_int64 *res, cnp.npy_int64 low, cnp.npy_int64 high) noexcept nogil

    void irk_long_vec(irk_state *state, cnp.npy_intp len, long *res) noexcept nogil

    ctypedef enum ch_st_enum:
        MATRIX = 0
        PACKED = 1
        DIAGONAL = 2

    void irk_multinormal_vec_ICDF(irk_state *state, cnp.npy_intp len, double *res, int dim, double *mean_vec, double *ch, ch_st_enum storage_mode) noexcept nogil
    void irk_multinormal_vec_BM1(irk_state *state, cnp.npy_intp len, double *res, int dim, double *mean_vec, double *ch, ch_st_enum storage_mode) noexcept nogil
    void irk_multinormal_vec_BM2(irk_state *state, cnp.npy_intp len, double *res, int dim, double *mean_vec, double *ch, ch_st_enum storage_mode) noexcept nogil


ctypedef void (* irk_cont0_vec)(irk_state *state, cnp.npy_intp len, double *res) noexcept nogil
ctypedef void (* irk_cont1_vec)(irk_state *state, cnp.npy_intp len, double *res, double a) noexcept nogil
ctypedef void (* irk_cont2_vec)(irk_state *state, cnp.npy_intp len, double *res, double a, double b) noexcept nogil
ctypedef void (* irk_cont3_vec)(irk_state *state, cnp.npy_intp len, double *res, double a, double b, double c) noexcept nogil

ctypedef void (* irk_disc0_vec)(irk_state *state, cnp.npy_intp len, int *res) noexcept nogil
ctypedef void (* irk_disc0_vec_long)(irk_state *state, cnp.npy_intp len, long *res) noexcept nogil
ctypedef void (* irk_discnp_vec)(irk_state *state, cnp.npy_intp len, int *res, int n, double a) noexcept nogil
ctypedef void (* irk_discdd_vec)(irk_state *state, cnp.npy_intp len, int *res, double n, double p) noexcept nogil
ctypedef void (* irk_discnmN_vec)(irk_state *state, cnp.npy_intp len, int *res, int n, int m, int N) noexcept nogil
ctypedef void (* irk_discd_vec)(irk_state *state, cnp.npy_intp len, int *res, double a) noexcept nogil
ctypedef void (* irk_discd_long_vec)(irk_state *state, cnp.npy_intp len, long *res, double a) noexcept nogil
ctypedef void (* irk_discdptr_vec)(irk_state *state, cnp.npy_intp len, int *res, double *a) noexcept nogil


cdef int r = cnp._import_array()
if (r<0):
    raise ImportError("Failed to import NumPy")

import numpy as np
import operator
import warnings
try:
    from threading import Lock
except ImportError:
    from dummy_threading import Lock

cdef object vec_cont0_array(irk_state *state, irk_cont0_vec func, object size,
                        object lock):
    cdef double *array_data
    cdef double res
    cdef cnp.ndarray array "arrayObject"
    cdef cnp.npy_intp length

    if size is None:
        func(state, 1, &res)
        return res
    else:
        array = <cnp.ndarray>np.empty(size, np.float64)
        length = cnp.PyArray_SIZE(array)
        array_data = <double *>cnp.PyArray_DATA(array)
        with lock, nogil:
            func(state, length, array_data)

        return array

cdef object vec_cont1_array_sc(irk_state *state, irk_cont1_vec func, object size, double a,
                        object lock):
    cdef double *array_data
    cdef double res
    cdef cnp.ndarray array "arrayObject"
    cdef cnp.npy_intp length

    if size is None:
        func(state, 1, &res, a)
        return res
    else:
        array = <cnp.ndarray>np.empty(size, np.float64)
        length = cnp.PyArray_SIZE(array)
        array_data = <double *>cnp.PyArray_DATA(array)
        with lock, nogil:
            func(state, length, array_data, a)

        return array


cdef object vec_cont1_array(irk_state *state, irk_cont1_vec func, object size,
                        cnp.ndarray oa, object lock):
    cdef double *array_data
    cdef double *oa_data
    cdef cnp.ndarray array "arrayObject"
    cdef cnp.npy_intp i, n, imax, res_size
    cdef cnp.flatiter itera
    cdef cnp.broadcast multi
    cdef object arr_obj
    cdef Py_ssize_t multi_nd
    cdef tuple multi_shape
    cdef cnp.npy_intp *multi_dims

    if size is None:
        array = <cnp.ndarray>cnp.PyArray_SimpleNew(cnp.PyArray_NDIM(oa),
                cnp.PyArray_DIMS(oa) , cnp.NPY_DOUBLE)
        imax = cnp.PyArray_SIZE(array)
        array_data = <double *>cnp.PyArray_DATA(array)
        itera = <cnp.flatiter>cnp.PyArray_IterNew(<object>oa)
        with lock, nogil:
            for i from 0 <= i < imax:
                func(state, 1, array_data + i, (<double *>(cnp.PyArray_ITER_DATA(itera)))[0])
                cnp.PyArray_ITER_NEXT(itera)
        arr_obj = <object> array
    else:
        array = <cnp.ndarray>np.empty(size, np.float64)
        array_data = <double *>cnp.PyArray_DATA(array)
        multi = <cnp.broadcast>cnp.PyArray_MultiIterNew(2, <void *>array, <void *>oa)
        res_size = cnp.PyArray_SIZE(array)
        if (cnp_PyArray_MultiIter_SIZE(multi) != res_size):
            raise ValueError("size is not compatible with inputs")

        multi = <cnp.broadcast> cnp.PyArray_MultiIterNew(1, <void *>oa)
        imax = cnp_PyArray_MultiIter_SIZE(multi)
        n = res_size // imax
        with lock, nogil:
            for i from 0 <= i < imax:
                oa_data = <double *>cnp.PyArray_MultiIter_DATA(multi, 0)
                func(state, n, array_data + n*i, oa_data[0])
                cnp.PyArray_MultiIter_NEXT(multi)
        arr_obj = <object>array
        multi_nd = cnp_PyArray_MultiIter_NDIM(multi)
        multi_dims = cnp_PyArray_MultiIter_DIMS(multi)
        multi_shape = cpython.tuple.PyTuple_New(multi_nd)
        for i from 0 <= i < multi_nd:
            cpython.tuple.PyTuple_SetItem(multi_shape, i, multi_dims[i]) 
        arr_obj.shape = (multi_shape + arr_obj.shape)[:arr_obj.ndim]
        multi_ndim = len(multi_shape)
        arr_obj = arr_obj.transpose(tuple(range(multi_ndim, arr_obj.ndim)) + tuple(range(0, multi_ndim)))

    return arr_obj

cdef object vec_cont2_array_sc(irk_state *state, irk_cont2_vec func, object size, double a,
                        double b, object lock):
    cdef double *array_data
    cdef double res
    cdef cnp.ndarray array "arrayObject"
    cdef cnp.npy_intp length
    cdef cnp.npy_intp i

    if size is None:
        func(state, 1, &res, a, b)
        return res
    else:
        array = <cnp.ndarray>np.empty(size, np.float64)
        length = cnp.PyArray_SIZE(array)
        array_data = <double *>cnp.PyArray_DATA(array)
        with lock, nogil:
            func(state, length, array_data, a, b)

        return array

cdef object vec_cont2_array(irk_state *state, irk_cont2_vec func, object size,
                        cnp.ndarray oa, cnp.ndarray ob, object lock):
    cdef double *array_data
    cdef double *oa_data
    cdef double *ob_data
    cdef cnp.ndarray array "arrayObject"
    cdef cnp.npy_intp i, n, imax, res_size
    cdef cnp.broadcast multi
    cdef object arr_obj
    cdef Py_ssize_t multi_nd
    cdef tuple multi_shape
    cdef cnp.npy_intp *multi_dims

    if size is None:
        multi = <cnp.broadcast> cnp.PyArray_MultiIterNew(2, <void *>oa, <void *>ob)
        array = <cnp.ndarray> cnp.PyArray_SimpleNew(
            cnp_PyArray_MultiIter_NDIM(multi),
            cnp_PyArray_MultiIter_DIMS(multi),
            cnp.NPY_DOUBLE
        )
        array_data = <double *>cnp.PyArray_DATA(array)
        with lock, nogil:
            for i from 0 <= i < cnp_PyArray_MultiIter_SIZE(multi):
                oa_data = <double *>cnp.PyArray_MultiIter_DATA(multi, 0)
                ob_data = <double *>cnp.PyArray_MultiIter_DATA(multi, 1)
                func(state, 1, &array_data[i], oa_data[0], ob_data[0])
                cnp.PyArray_MultiIter_NEXT(multi)
        arr_obj = <object> array
    else:
        array = <cnp.ndarray>np.empty(size, np.float64)
        array_data = <double *>cnp.PyArray_DATA(array)
        multi = <cnp.broadcast >cnp.PyArray_MultiIterNew(3, <void*>array, <void *>oa, <void *>ob)
        res_size = cnp.PyArray_SIZE(array)
        if (cnp_PyArray_MultiIter_SIZE(multi) != res_size):
            raise ValueError("size is not compatible with inputs")

        multi = <cnp.broadcast> cnp.PyArray_MultiIterNew(2, <void *>oa, <void *>ob)
        imax = cnp_PyArray_MultiIter_SIZE(multi)
        n = res_size // imax
        with lock, nogil:
            for i from 0 <= i < imax:
                oa_data = <double *>cnp.PyArray_MultiIter_DATA(multi, 0)
                ob_data = <double *>cnp.PyArray_MultiIter_DATA(multi, 1)
                func(state, n, array_data + n*i, oa_data[0], ob_data[0])
                cnp.PyArray_MultiIter_NEXT(multi)
        arr_obj = <object> array
        multi_nd = cnp_PyArray_MultiIter_NDIM(multi)
        multi_dims = cnp_PyArray_MultiIter_DIMS(multi)
        multi_shape = cpython.tuple.PyTuple_New(multi_nd)
        for i from 0 <= i < multi_nd:
            cpython.tuple.PyTuple_SetItem(multi_shape, i, multi_dims[i]) 
        arr_obj.shape = (multi_shape + arr_obj.shape)[:arr_obj.ndim]
        multi_ndim = len(multi_shape)
        arr_obj = arr_obj.transpose(tuple(range(multi_ndim, arr_obj.ndim)) + tuple(range(0, multi_ndim)))

    return arr_obj


cdef object vec_cont3_array_sc(irk_state *state, irk_cont3_vec func, object size, double a,
                        double b, double c, object lock):
    cdef double *array_data
    cdef double res
    cdef cnp.ndarray array "arrayObject"
    cdef cnp.npy_intp length
    cdef cnp.npy_intp i

    if size is None:
        func(state, 1, &res, a, b, c)
        return res
    else:
        array = <cnp.ndarray>np.empty(size, np.float64)
        length = cnp.PyArray_SIZE(array)
        array_data = <double *>cnp.PyArray_DATA(array)
        with lock, nogil:
            func(state, length, array_data, a, b, c)

        return array

cdef object vec_cont3_array(irk_state *state, irk_cont3_vec func, object size,
                        cnp.ndarray oa, cnp.ndarray ob, cnp.ndarray oc, object lock):
    cdef double *array_data
    cdef double *oa_data
    cdef double *ob_data
    cdef double *oc_data
    cdef cnp.ndarray array "arrayObject"
    cdef cnp.npy_intp i, res_size, n, imax
    cdef cnp.broadcast multi
    cdef object arr_obj
    cdef Py_ssize_t multi_nd
    cdef tuple multi_shape
    cdef cnp.npy_intp *multi_dims

    if size is None:
        multi = <cnp.broadcast> cnp.PyArray_MultiIterNew(3, <void *>oa, <void *>ob, <void *>oc)
        array = <cnp.ndarray> cnp.PyArray_SimpleNew(cnp_PyArray_MultiIter_NDIM(multi), cnp_PyArray_MultiIter_DIMS(multi), cnp.NPY_DOUBLE)
        array_data = <double *>cnp.PyArray_DATA(array)
        with lock, nogil:
            for i from 0 <= i < cnp_PyArray_MultiIter_SIZE(multi):
                oa_data = <double *>cnp.PyArray_MultiIter_DATA(multi, 0)
                ob_data = <double *>cnp.PyArray_MultiIter_DATA(multi, 1)
                oc_data = <double *>cnp.PyArray_MultiIter_DATA(multi, 2)
                func(state, 1, &array_data[i], oa_data[0], ob_data[0], oc_data[0])
                cnp.PyArray_MultiIter_NEXT(multi)
        arr_obj = <object>array
    else:
        array = <cnp.ndarray>np.empty(size, np.float64)
        array_data = <double *>cnp.PyArray_DATA(array)
        multi = <cnp.broadcast>cnp.PyArray_MultiIterNew(4, <void*>array, <void *>oa,
                                                <void *>ob, <void *>oc)
        res_size = cnp.PyArray_SIZE(array)
        if (cnp_PyArray_MultiIter_SIZE(multi) != res_size):
            raise ValueError("size is not compatible with inputs")

        multi = <cnp.broadcast> cnp.PyArray_MultiIterNew(3, <void *>oa, <void *>ob, <void *>oc)
        imax = cnp_PyArray_MultiIter_SIZE(multi)
        n = res_size // imax
        with lock, nogil:
            for i from 0 <= i < imax:
                oa_data = <double *>cnp.PyArray_MultiIter_DATA(multi, 0)
                ob_data = <double *>cnp.PyArray_MultiIter_DATA(multi, 1)
                oc_data = <double *>cnp.PyArray_MultiIter_DATA(multi, 2)
                func(state, n, array_data + n*i, oa_data[0], ob_data[0], oc_data[0])
                cnp.PyArray_MultiIter_NEXT(multi)
        arr_obj = <object>array
        multi_nd = cnp_PyArray_MultiIter_NDIM(multi)
        multi_dims = cnp_PyArray_MultiIter_DIMS(multi)
        multi_shape = cpython.tuple.PyTuple_New(multi_nd)
        for i from 0 <= i < multi_nd:
            cpython.tuple.PyTuple_SetItem(multi_shape, i, multi_dims[i]) 
        arr_obj.shape = (multi_shape + arr_obj.shape)[:arr_obj.ndim]
        multi_ndim = len(multi_shape)
        arr_obj = arr_obj.transpose(tuple(range(multi_ndim, arr_obj.ndim)) + tuple(range(0, multi_ndim)))

    return arr_obj


cdef object vec_long_disc0_array(
    irk_state *state, irk_disc0_vec_long func,
    object size, object lock
):
    cdef long *array_data
    cdef long res
    cdef cnp.ndarray array "arrayObject"
    cdef cnp.npy_intp length
    cdef cnp.npy_intp i

    if size is None:
        func(state, 1, &res)
        return res
    array = <cnp.ndarray>np.empty(size, np.dtype("long"))
    length = cnp.PyArray_SIZE(array)
    array_data = <long *>cnp.PyArray_DATA(array)
    with lock, nogil:
        func(state, length, array_data)

    return array


cdef object vec_discnp_array_sc(
    irk_state *state, irk_discnp_vec func, object size,
    int n, double p, object lock
):
    cdef int *array_data
    cdef int res
    cdef cnp.ndarray array "arrayObject"
    cdef cnp.npy_intp length
    cdef cnp.npy_intp i

    if size is None:
        func(state, 1, &res, n, p)
        return res
    else:
        array = <cnp.ndarray>np.empty(size, np.intc)
        length = cnp.PyArray_SIZE(array)
        array_data = <int *>cnp.PyArray_DATA(array)
        with lock, nogil:
            func(state, length, array_data, n, p)
        return array


cdef object vec_discnp_array(irk_state *state, irk_discnp_vec func, object size,
                         cnp.ndarray on, cnp.ndarray op, object lock):
    cdef int *array_data
    cdef cnp.ndarray array "arrayObject"
    cdef cnp.npy_intp length
    cdef cnp.npy_intp i, n, imax, res_size
    cdef double *op_data
    cdef int *on_data
    cdef cnp.broadcast multi
    cdef object arr_obj
    cdef Py_ssize_t multi_nd
    cdef tuple multi_shape
    cdef cnp.npy_intp *multi_dims
    cdef int multi_nd_i

    if size is None:
        multi = <cnp.broadcast> cnp.PyArray_MultiIterNew(2, <void *>on, <void *>op)
        multi_nd_i = cnp_PyArray_MultiIter_NDIM(multi)
        multi_dims = cnp_PyArray_MultiIter_DIMS(multi)
        array = <cnp.ndarray> cnp.PyArray_SimpleNew(multi_nd_i, multi_dims, cnp.NPY_INT)
        array_data = <int *>cnp.PyArray_DATA(array)
        with lock, nogil:
            for i from 0 <= i < cnp_PyArray_MultiIter_SIZE(multi):
                on_data = <int *>cnp.PyArray_MultiIter_DATA(multi, 0)
                op_data = <double *>cnp.PyArray_MultiIter_DATA(multi, 1)
                func(state, 1, &array_data[i], on_data[0], op_data[0])
                cnp.PyArray_MultiIter_NEXT(multi)
        arr_obj = <object>array
    else:
        array = <cnp.ndarray>np.empty(size, np.intc)
        array_data = <int *>cnp.PyArray_DATA(array)
        multi = <cnp.broadcast>cnp.PyArray_MultiIterNew(3, <void*>array, <void *>on, <void *>op)
        res_size = cnp.PyArray_SIZE(array)
        if (cnp_PyArray_MultiIter_SIZE(multi) != res_size):
            raise ValueError("size is not compatible with inputs")

        multi = <cnp.broadcast> cnp.PyArray_MultiIterNew(2, <void *>on, <void *>op)
        imax = cnp_PyArray_MultiIter_SIZE(multi)
        n = res_size // imax
        with lock, nogil:
            for i from 0 <= i < imax:
                on_data = <int *>cnp.PyArray_MultiIter_DATA(multi, 0)
                op_data = <double *>cnp.PyArray_MultiIter_DATA(multi, 1)
                func(state, n, array_data + n * i, on_data[0], op_data[0])
                cnp.PyArray_MultiIter_NEXT(multi)
        arr_obj = <object>array
        multi_nd = cnp_PyArray_MultiIter_NDIM(multi)
        multi_dims = cnp_PyArray_MultiIter_DIMS(multi)
        multi_shape = cpython.tuple.PyTuple_New(multi_nd)
        for i from 0 <= i < multi_nd:
            cpython.tuple.PyTuple_SetItem(multi_shape, i, multi_dims[i]) 
        arr_obj.shape = (multi_shape + arr_obj.shape)[:arr_obj.ndim]
        multi_ndim = len(multi_shape)
        arr_obj = arr_obj.transpose(tuple(range(multi_ndim, arr_obj.ndim)) + tuple(range(0, multi_ndim)))

    return arr_obj


cdef object vec_discdd_array_sc(irk_state *state, irk_discdd_vec func, object size,
                            double n, double p, object lock):
    cdef int *array_data
    cdef int res
    cdef cnp.ndarray array "arrayObject"
    cdef cnp.npy_intp length
    cdef cnp.npy_intp i

    if size is None:
        func(state, 1, &res, n, p)
        return res
    else:
        array = <cnp.ndarray>np.empty(size, np.intc)
        length = cnp.PyArray_SIZE(array)
        array_data = <int *>cnp.PyArray_DATA(array)
        with lock, nogil:
            func(state, length, array_data, n, p)

        return array


cdef object vec_discdd_array(irk_state *state, irk_discdd_vec func, object size,
                         cnp.ndarray on, cnp.ndarray op, object lock):
    cdef int *array_data
    cdef cnp.ndarray array "arrayObject"
    cdef cnp.npy_intp i, imax, n, res_size
    cdef double *op_data
    cdef double *on_data
    cdef cnp.broadcast multi
    cdef object arr_obj
    cdef Py_ssize_t multi_nd
    cdef tuple multi_shape
    cdef cnp.npy_intp *multi_dims

    if size is None:
        multi = <cnp.broadcast> cnp.PyArray_MultiIterNew(2, <void *>on, <void *>op)
        array = <cnp.ndarray> cnp.PyArray_SimpleNew(cnp_PyArray_MultiIter_NDIM(multi), cnp_PyArray_MultiIter_DIMS(multi), cnp.NPY_INT)
        array_data = <int *>cnp.PyArray_DATA(array)
        with lock, nogil:
            for i from 0 <= i < cnp_PyArray_MultiIter_SIZE(multi):
                on_data = <double *>cnp.PyArray_MultiIter_DATA(multi, 0)
                op_data = <double *>cnp.PyArray_MultiIter_DATA(multi, 1)
                func(state, 1, &array_data[i], on_data[0], op_data[0])
                cnp.PyArray_MultiIter_NEXT(multi)
        arr_obj = <object>array
    else:
        array = <cnp.ndarray>np.empty(size, np.intc)
        array_data = <int *>cnp.PyArray_DATA(array)
        res_size = cnp.PyArray_SIZE(array)
        multi = <cnp.broadcast>cnp.PyArray_MultiIterNew(3, <void*>array, <void *>on, <void *>op)
        if (cnp_PyArray_MultiIter_SIZE(multi) != res_size):
            raise ValueError("size is not compatible with inputs")

        multi = <cnp.broadcast> cnp.PyArray_MultiIterNew(2, <void *>on, <void *>op)
        imax = cnp_PyArray_MultiIter_SIZE(multi)
        n = res_size // imax
        with lock, nogil:
            for i from 0 <= i < imax:
                on_data = <double *>cnp.PyArray_MultiIter_DATA(multi, 0)
                op_data = <double *>cnp.PyArray_MultiIter_DATA(multi, 1)
                func(state, n, array_data + n * i, on_data[0], op_data[0])
                cnp.PyArray_MultiIter_NEXT(multi)
        arr_obj = <object>array
        multi_nd = cnp_PyArray_MultiIter_NDIM(multi)
        multi_dims = cnp_PyArray_MultiIter_DIMS(multi)
        multi_shape = cpython.tuple.PyTuple_New(multi_nd)
        for i from 0 <= i < multi_nd:
            cpython.tuple.PyTuple_SetItem(multi_shape, i, multi_dims[i]) 
        arr_obj.shape = (multi_shape + arr_obj.shape)[:arr_obj.ndim]
        multi_ndim = len(multi_shape)
        arr_obj = arr_obj.transpose(tuple(range(multi_ndim, arr_obj.ndim)) + tuple(range(0, multi_ndim)))

    return arr_obj


cdef object vec_discnmN_array_sc(irk_state *state, irk_discnmN_vec func, object size,
                             int n, int m, int N, object lock):
    cdef int *array_data
    cdef int res
    cdef cnp.ndarray array "arrayObject"
    cdef cnp.npy_intp length
    cdef cnp.npy_intp i

    if size is None:
        func(state, 1, &res, n, m, N)
        return res
    else:
        array = <cnp.ndarray>np.empty(size, np.intc)
        length = cnp.PyArray_SIZE(array)
        array_data = <int *>cnp.PyArray_DATA(array)
        with lock, nogil:
            func(state, length, array_data, n, m, N)
        return array


cdef object vec_discnmN_array(irk_state *state, irk_discnmN_vec func, object size,
                          cnp.ndarray on, cnp.ndarray om, cnp.ndarray oN, object lock):
    cdef int *array_data
    cdef int *on_data
    cdef int *om_data
    cdef int *oN_data
    cdef cnp.ndarray array "arrayObject"
    cdef cnp.npy_intp i
    cdef cnp.broadcast multi, multi2
    cdef cnp.npy_intp imax, n, res_size
    cdef object arr_obj
    cdef Py_ssize_t multi_nd
    cdef tuple multi_shape
    cdef cnp.npy_intp *multi_dims

    if size is None:
        multi = <cnp.broadcast> cnp.PyArray_MultiIterNew(3, <void *>on, <void *>om, <void *>oN)
        array = <cnp.ndarray> cnp.PyArray_SimpleNew(cnp_PyArray_MultiIter_NDIM(multi), cnp_PyArray_MultiIter_DIMS(multi), cnp.NPY_INT)
        array_data = <int *>cnp.PyArray_DATA(array)
        with lock, nogil:
            for i from 0 <= i < cnp_PyArray_MultiIter_SIZE(multi):
                on_data = <int *>cnp.PyArray_MultiIter_DATA(multi, 0)
                om_data = <int *>cnp.PyArray_MultiIter_DATA(multi, 1)
                oN_data = <int *>cnp.PyArray_MultiIter_DATA(multi, 2)
                func(state, 1, array_data + i, on_data[0], om_data[0], oN_data[0])
                cnp.PyArray_MultiIter_NEXT(multi)
        arr_obj = <object>array
    else:
        array = <cnp.ndarray>np.empty(size, np.intc)
        array_data = <int *>cnp.PyArray_DATA(array)
        multi = <cnp.broadcast>cnp.PyArray_MultiIterNew(4, <void*>array, <void *>on, <void *>om,
                                                <void *>oN)
        res_size = cnp.PyArray_SIZE(array)
        if (cnp_PyArray_MultiIter_SIZE(multi) != res_size):
            raise ValueError("size is not compatible with inputs")

        multi = <cnp.broadcast> cnp.PyArray_MultiIterNew(3, <void *>on, <void *>om, <void *>oN)
        imax = cnp_PyArray_MultiIter_SIZE(multi)
        n = res_size // imax
        with lock, nogil:
            for i from 0 <= i < imax:
                on_data = <int *>cnp.PyArray_MultiIter_DATA(multi, 0)
                om_data = <int *>cnp.PyArray_MultiIter_DATA(multi, 1)
                oN_data = <int *>cnp.PyArray_MultiIter_DATA(multi, 2)
                func(state, n, array_data + n*i, on_data[0], om_data[0], oN_data[0])
                cnp.PyArray_MultiIter_NEXT(multi)
        arr_obj = <object>array
        multi_nd = cnp_PyArray_MultiIter_NDIM(multi)
        multi_dims = cnp_PyArray_MultiIter_DIMS(multi)
        multi_shape = cpython.tuple.PyTuple_New(multi_nd)
        for i from 0 <= i < multi_nd:
            cpython.tuple.PyTuple_SetItem(multi_shape, i, multi_dims[i]) 
        arr_obj.shape = (multi_shape + arr_obj.shape)[:arr_obj.ndim]
        multi_ndim = len(multi_shape)
        arr_obj = arr_obj.transpose(tuple(range(multi_ndim, arr_obj.ndim)) + tuple(range(0, multi_ndim)))

    return arr_obj

cdef object vec_discd_array_sc(irk_state *state, irk_discd_vec func, object size,
                           double a, object lock):
    cdef int *array_data
    cdef int res
    cdef cnp.ndarray array "arrayObject"
    cdef cnp.npy_intp length
    cdef cnp.npy_intp i

    if size is None:
        func(state, 1, &res, a)
        return res
    else:
        array = <cnp.ndarray>np.empty(size, np.intc)
        length = cnp.PyArray_SIZE(array)
        array_data = <int *>cnp.PyArray_DATA(array)
        with lock, nogil:
            func(state, length, array_data, a)

        return array

cdef object vec_long_discd_array_sc(irk_state *state, irk_discd_long_vec func, object size,
                           double a, object lock):
    cdef long *array_data
    cdef long res
    cdef cnp.ndarray array "arrayObject"
    cdef cnp.npy_intp length
    cdef cnp.npy_intp i

    if size is None:
        func(state, 1, &res, a)
        return res
    else:
        array = <cnp.ndarray>np.empty(size, np.dtype("long"))
        length = cnp.PyArray_SIZE(array)
        array_data = <long *>cnp.PyArray_DATA(array)
        with lock, nogil:
            func(state, length, array_data, a)

        return array

cdef object vec_discd_array(irk_state *state, irk_discd_vec func, object size, cnp.ndarray oa,
                        object lock):
    cdef int *array_data
    cdef double *oa_data
    cdef cnp.ndarray array "arrayObject"
    cdef cnp.npy_intp length, res_size
    cdef cnp.npy_intp i, imax, n
    cdef cnp.broadcast multi
    cdef cnp.flatiter itera
    cdef object arr_obj

    if size is None:
        array = <cnp.ndarray>cnp.PyArray_SimpleNew(cnp.PyArray_NDIM(oa),
                cnp.PyArray_DIMS(oa), cnp.NPY_INT32)
        length = cnp.PyArray_SIZE(array)
        array_data = <int *>cnp.PyArray_DATA(array)
        itera = <cnp.flatiter>cnp.PyArray_IterNew(<object>oa)
        with lock, nogil:
            for i from 0 <= i < length:
                func(state, 1, &array_data[i], (<double *>(cnp.PyArray_ITER_DATA(itera)))[0])
                cnp.PyArray_ITER_NEXT(itera)
        arr_obj = <object>array
    else:
        array = <cnp.ndarray>np.empty(size, np.intc)
        array_data = <int *>cnp.PyArray_DATA(array)
        multi = <cnp.broadcast>cnp.PyArray_MultiIterNew(2, <void *>array, <void *>oa)
        res_size = cnp.PyArray_SIZE(array)
        if (cnp_PyArray_MultiIter_SIZE(multi) != res_size):
            raise ValueError("size is not compatible with inputs")

        imax = oa.size
        n = res_size // imax
        with lock, nogil:
            for i from 0 <= i < imax:
                oa_data = <double *>cnp.PyArray_MultiIter_DATA(multi, 1)
                func(state, n, array_data + n*i, oa_data[0])
                cnp.PyArray_MultiIter_NEXTi(multi, 1)
        arr_obj = <object>array
        arr_obj.shape = ((<object >oa).shape + arr_obj.shape)[:arr_obj.ndim]
        arr_obj = arr_obj.transpose(tuple(range(oa.ndim, arr_obj.ndim)) + tuple(range(0, oa.ndim)))

    return arr_obj

cdef object vec_long_discd_array(irk_state *state, irk_discd_long_vec func, object size, cnp.ndarray oa,
                        object lock):
    cdef long *array_data
    cdef double *oa_data
    cdef cnp.ndarray array "arrayObject"
    cdef cnp.npy_intp length, res_size
    cdef cnp.npy_intp i, imax, n
    cdef cnp.broadcast multi
    cdef cnp.flatiter itera
    cdef object arr_obj

    if size is None:
        array = <cnp.ndarray>cnp.PyArray_SimpleNew(cnp.PyArray_NDIM(oa),
                cnp.PyArray_DIMS(oa), cnp.NPY_LONG)
        length = cnp.PyArray_SIZE(array)
        array_data = <long *>cnp.PyArray_DATA(array)
        itera = <cnp.flatiter>cnp.PyArray_IterNew(<object>oa)
        with lock, nogil:
            for i from 0 <= i < length:
                func(state, 1, array_data + i, (<double *>(cnp.PyArray_ITER_DATA(itera)))[0])
                cnp.PyArray_ITER_NEXT(itera)
        arr_obj = <object>array
    else:
        array = <cnp.ndarray>np.empty(size, np.dtype("long"))
        array_data = <long *>cnp.PyArray_DATA(array)
        multi = <cnp.broadcast>cnp.PyArray_MultiIterNew(2, <void *>array, <void *>oa)
        res_size = cnp.PyArray_SIZE(array)
        if (cnp_PyArray_MultiIter_SIZE(multi) != res_size):
            raise ValueError("size is not compatible with inputs")

        imax = oa.size
        n = res_size // imax
        with lock, nogil:
            for i from 0 <= i < imax:
                oa_data = <double *>cnp.PyArray_MultiIter_DATA(multi, 1)
                func(state, n, array_data + n*i, oa_data[0])
                cnp.PyArray_MultiIter_NEXTi(multi, 1)
        arr_obj = <object>array
        arr_obj.shape = ((<object> oa).shape + arr_obj.shape)[:arr_obj.ndim]
        arr_obj = arr_obj.transpose(tuple(range(oa.ndim, arr_obj.ndim)) + tuple(range(0, oa.ndim)))

    return arr_obj

cdef object vec_Poisson_array(irk_state *state, irk_discdptr_vec func1, irk_discd_vec func2, object size, cnp.ndarray olambda,
                        object lock):
    cdef int *array_data
    cdef double *oa_data
    cdef cnp.ndarray array "arrayObject"
    cdef cnp.npy_intp length, res_size
    cdef cnp.npy_intp i, imax, n
    cdef cnp.broadcast multi
    cdef cnp.flatiter itera
    cdef object arr_obj

    if size is None:
        array = <cnp.ndarray>cnp.PyArray_SimpleNew(cnp.PyArray_NDIM(olambda),
                cnp.PyArray_DIMS(olambda), cnp.NPY_INT)
        length = cnp.PyArray_SIZE(array)
        array_data = <int *>cnp.PyArray_DATA(array)
        oa_data = <double *>cnp.PyArray_DATA(olambda)
        with lock, nogil:
            func1(state, length, array_data, oa_data)
        arr_obj = <object>array
    else:
        array = <cnp.ndarray>np.empty(size, np.intc)
        array_data = <int *>cnp.PyArray_DATA(array)
        multi = <cnp.broadcast>cnp.PyArray_MultiIterNew(2, <void *>array, <void *>olambda)
        res_size = cnp.PyArray_SIZE(array)
        if (cnp_PyArray_MultiIter_SIZE(multi) != res_size):
            raise ValueError("size is not compatible with inputs")

        imax = olambda.size
        n = res_size // imax
        if imax < n:
            with lock, nogil:
                for i from 0 <= i < imax:
                    oa_data = <double *>cnp.PyArray_MultiIter_DATA(multi, 1)
                    func2(state, n, array_data + n*i, oa_data[0])
                    cnp.PyArray_MultiIter_NEXTi(multi, 1)
            arr_obj = <object>array
            arr_obj.shape = ((<object>olambda).shape + arr_obj.shape)[:arr_obj.ndim]
            arr_obj = arr_obj.transpose(tuple(range(olambda.ndim, arr_obj.ndim)) + tuple(range(0, olambda.ndim)))
        else:
            oa_data = <double *>cnp.PyArray_DATA(olambda)
            with lock, nogil:
                for i from 0 <= i < n:
                    func1(state, imax, array_data + imax*i, oa_data)
            arr_obj = <object>array

    return arr_obj


cdef double kahan_sum(double *darr, cnp.npy_intp n) nogil:
    cdef double c, y, t, sum
    cdef cnp.npy_intp i
    sum = darr[0]
    c = 0.0
    for i from 1 <= i < n:
        y = darr[i] - c
        t = sum + y
        c = (t-sum) - y
        sum = t
    return sum

# computes dim*(dim + 1)/2  -- number of elements in lower-triangular part of a square matrix of shape (dim, dim)
cdef inline int packed_cholesky_size(int dim):
    cdef int dh, lsb

    dh = (dim >> 1)
    lsb = (dim & 1)
    return (lsb + dh) * (dim + (1 - lsb))

def _shape_from_size(size, d):
    if size is None:
        shape = (d,)
    else:
        try:
           shape = (operator.index(size), d)
        except TypeError:
           shape = tuple(size) + (d,)
    return shape

# sampling methods enum
ICDF = 0
BOXMULLER = 1
BOXMULLER2 = 2
POISNORM = 3
PTPE = 4

_method_alias_dict_gaussian = {'ICDF': ICDF, 'Inversion': ICDF,
                      'BoxMuller': BOXMULLER, 'Box-Muller': BOXMULLER,
                      'BoxMuller2': BOXMULLER2, 'Box-Muller2': BOXMULLER2}

_method_alias_dict_gaussian_short = {'ICDF': ICDF, 'Inversion': ICDF,
                            'BoxMuller': BOXMULLER, 'Box-Muller': BOXMULLER}

_method_alias_dict_poisson = {'PTPE' : PTPE, 'Poisson-Normal': POISNORM, 'POISNORM' : POISNORM}

def choose_method(method, mlist, alias_dict = None):
    if (method not in mlist):
        if (alias_dict is None) or (not isinstance(alias_dict, dict)):
            # issue warning
            return mlist[0]
        else:
            if method not in alias_dict.keys():
                return mlist[0]
            else:
                return alias_dict[method]
    else:
        return method

_brng_dict = {
    'MT19937' : MT19937,
    'SFMT19937' : SFMT19937,
    'WH' : WH,
    'MT2203': MT2203,
    'MCG31' : MCG31,
    'R250' : R250,
    'MRG32K3A' : MRG32K3A,
    'MCG59' : MCG59,
    'PHILOX4X32X10' : PHILOX4X32X10,
    'NONDETERM' : NONDETERM,
    'NONDETERMINISTIC' : NONDETERM,
    'NON_DETERMINISTIC' : NONDETERM,
    'ARS5' : ARS5
}

_brng_dict_stream_max = {
    MT19937: 1,
    SFMT19937: 1,
    WH: 273,
    MT2203: 6024,
    MCG31: 1,
    R250: 1,
    MRG32K3A: 1,
    MCG59: 1,
    PHILOX4X32X10: 1,
    NONDETERM: 1,
    ARS5: 1,
}

cdef irk_brng_t _default_fallback_brng_token_(brng):
    cdef irk_brng_t brng_token
    warnings.warn(("The basic random generator specification {given} is not recognized. "
                   "\"MT19937\" will be used instead").format(given=brng),
                  UserWarning)
    brng_token = MT19937
    return brng_token

cdef irk_brng_t _parse_brng_token_(brng):
    cdef irk_brng_t brng_token

    if isinstance(brng, str):
        tmp = _brng_dict.get(brng.upper(), None)
        if tmp is None:
            brng_token = _default_fallback_brng_token_(brng)
        else:
            brng_token = tmp
    elif isinstance(brng, int):
        brng_token = operator.index(brng)
    else:
        brng_token = _default_fallback_brng_token_(brng)

    return brng_token

def _parse_brng_argument(brng):
    cdef irk_brng_t brng_token
    cdef unsigned int stream_id = 0

    if isinstance(brng, (list, tuple)) and len(brng) == 2:
        bt, s = brng;
        brng_token = _parse_brng_token_(bt)
        smax = _brng_dict_stream_max[brng_token]
        if isinstance(s, int):
            s = s % smax
            if (s != brng[1]):
                warnings.warn(("The generator index {actual} is not between 0 and {max}, "
                        "index {choice} will be used.").format(actual=brng[-1], max=smax-1, choice=s),
                        UserWarning)
            stream_id = s
    else:
        brng_token = _parse_brng_token_(brng)

    return (brng_token, stream_id)

def _brng_id_to_name(int brng_id):
    cdef object nm
    cdef object brng_name = None
    for nm in _brng_dict:
        if _brng_dict[nm] == brng_id:
            brng_name = nm

    return brng_name


cdef class RandomState:
    """
    RandomState(seed=None, brng='MT19937')

    Container for the Intel(R) MKL-powered (pseudo-)random number generators.

    `RandomState` exposes a number of methods for generating random numbers
    drawn from a variety of probability distributions. In addition to the
    distribution-specific arguments, each method takes a keyword argument
    `size` that defaults to ``None``. If `size` is ``None``, then a single
    value is generated and returned. If `size` is an integer, then a 1-D
    array filled with generated values is returned. If `size` is a tuple,
    then an array with that shape is filled and returned.

    *Compatibility Notice*
    This version of numpy.random has been rewritten to use MKL's vector
    statistics functionality, that provides efficient implementation of
    the MT19937 and many other basic psuedo-random number generation
    algorithms as well as efficient sampling from other common statistical
    distributions. As a consequence this version is NOT seed-compatible with
    the original numpy.random.

    Parameters
    ----------
    seed : {None, int, array_like}, optional
        Random seed initializing the pseudo-random number generator.
        Can be an integer, an array (or other sequence) of integers of
        any length, or ``None`` (the default).
        If `seed` is ``None``, then `RandomState` will try to read data from
        ``/dev/urandom`` (or the Windows analogue) if available or seed from
        the clock otherwise.
    brng : {'MT19937', 'SFMT19937', 'MT2203', 'R250', 'WH', 'MCG31', 'MCG59',
            'MRG32K3A', 'PHILOX4X32X10', 'NONDETERM', 'ARS5'}, optional
        basic pseudo-random number generation algorithms, or non-deterministic
        hardware-based generator, provided by Intel MKL. The default choice is 
        'MT19937' - the Mersenne Twister generator.

    Notes
    -----
    The Python stdlib module "random" also contains a Mersenne Twister
    pseudo-random number generator with a number of methods that are similar
    to the ones available in `RandomState`. `RandomState`, besides being
    NumPy-aware, has the advantage that it provides a much larger number
    of probability distributions to choose from.

    References
    -----
    MKL Documentation: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html

    """
    cdef irk_state *internal_state
    cdef object lock
    poisson_lam_max = np.iinfo('l').max - np.sqrt(np.iinfo('l').max)*10

    def __init__(self, seed=None, brng='MT19937'):
        self.internal_state = <irk_state*>PyMem_Malloc(sizeof(irk_state))
        memset(self.internal_state, 0, sizeof(irk_state))

        self.lock = Lock()
        self.seed(seed, brng)

    def __dealloc__(self):
        if self.internal_state != NULL:
            irk_dealloc_stream(self.internal_state)
            PyMem_Free(self.internal_state)
            self.internal_state = NULL

    def seed(self, seed=None, brng=None):
        """
        seed(seed=None, brng=None)

        Seed the generator.

        This method is called when `RandomState` is initialized. It can be
        called again to re-seed the generator. For details, see `RandomState`.

        Parameters
        ----------
        seed : int or array_like, optional
            Seed for `RandomState`.
            Must be convertible to 32 bit unsigned integers.
        brng : {'MT19937', 'SFMT19937', 'MT2203', 'R250', 'WH', 'MCG31',
                'MCG59', 'MRG32K3A', 'PHILOX4X32X10', 'NONDETERM',
                'ARS5', None}, optional
            basic pseudo-random number generation algorithms, or non-deterministic
            hardware-based generator, provided by Intel MKL. Use `brng==None` to keep
            the `brng` specified during construction of this class instance.

        See Also
        --------
        RandomState

        References
        --------
        MKL Documentation: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html

        """
        cdef irk_error errcode
        cdef irk_brng_t brng_token = MT19937
        cdef unsigned int stream_id
        cdef cnp.ndarray obj "arrayObject_obj"

        if (brng):
            brng_token, stream_id = _parse_brng_argument(brng);
        else:
            brng_token = <irk_brng_t> irk_get_brng_and_stream_mkl(self.internal_state, &stream_id)
        try:
            if seed is None:
                with self.lock:
                    errcode = irk_randomseed_mkl(self.internal_state, brng_token, stream_id)
            else:
                idx = operator.index(seed)
                if idx > int(2**32 - 1) or idx < 0:
                    raise ValueError("Seed must be between 0 and 4294967295")
                with self.lock:
                    irk_seed_mkl(self.internal_state, idx, brng_token, stream_id)
        except TypeError:
            obj = np.asarray(seed)
            if not obj.dtype is np.dtype('uint64'):
                obj = obj.astype(np.int64, casting='safe')
            if ((obj > int(2**32 - 1)) | (obj < 0)).any():
                raise ValueError("Seed must be between 0 and 4294967295")
            obj = obj.astype('uint32', casting='unsafe')
            with self.lock:
                irk_seed_mkl_array(self.internal_state, <unsigned int *>cnp.PyArray_DATA(obj),
                                        cnp.PyArray_DIM(obj, 0), brng_token, stream_id)

    def get_state(self):
        """
        get_state()

        Return a tuple representing the internal state of the generator.

        For more details, see `set_state`.

        Returns
        -------
        out : tuple(str, bytes)
            The returned tuple has the following items:

            1. the string, defaulting to 'MT19937', specifying the
               basic psedo-random number generation algorithm.
            2. a bytes object holding content of Intel MKL's stream for the
               given BRNG.

        See Also
        --------
        set_state

        Notes
        -----
        `set_state` and `get_state` are not needed to work with any of the
        random distributions in NumPy. If the internal state is manually altered,
        the user should know exactly what he/she is doing.

        References
        -----
        MKL Documentation: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html

        """
        cdef int state_buffer_size
        cdef int brng_id
        cdef void *bytesPtr

        with self.lock:
            state_buffer_size = irk_get_stream_size(self.internal_state)
        bytestring = empty_py_bytes(state_buffer_size, &bytesPtr)
        with self.lock:
            brng_id = irk_get_brng_mkl(self.internal_state)
            irk_get_state_mkl(self.internal_state, <char *>bytesPtr)

        brng_name = _brng_id_to_name(brng_id)
        return (brng_name, bytestring)

    def set_state(self, state):
        """
        set_state(state)

        Set the internal state of the generator from a tuple.

        For use if one has reason to manually (re-)set the internal state of the
        chosen pseudo-random number generating algorithm.

        Parameters
        ----------
        state : tuple(str, bytes)
            The `state` tuple has the following items:

            1. the string, defaulting to 'MT19937', specifying the
               basic psedo-random number generation algorithm.
            2. a bytes object holding content of Intel MKL's stream for the
               given BRNG.

        Returns
        -------
        out : None
            Returns 'None' on success.

        See Also
        --------
        get_state

        Notes
        -----
        `set_state` and `get_state` are not needed to work with any of the
        random distributions in NumPy. If the internal state is manually altered,
        the user should know exactly what he/she is doing.

        For backwards compatibility, the form (str, array of 624 uints, int) is
        also accepted although in such a case keys are used to seed the generator,
        and position index pos is ignored: ``state = ('MT19937', keys, pos)``.

        References
        ----------
        MKL Documentation: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html

        """
        cdef char *bytes_ptr
        cdef int brng_id
        cdef cnp.ndarray obj "arrayObject_obj"

        state_len = len(state)
        if(state_len != 2):
            if (state_len == 3 or state_len == 5):
                algo_name, key, pos = state[:3]
                if algo_name != 'MT19937':
                    raise ValueError("The legacy state input algorithm must be 'MT19937'")
                try:
                    obj = <cnp.ndarray> cnp.PyArray_ContiguousFromObject(key, cnp.NPY_ULONG, 1, 1)
                except TypeError:
                    # compatibility -- could be an older pickle
                    obj = <cnp.ndarray> cnp.PyArray_ContiguousFromObject(key, cnp.NPY_LONG, 1, 1)
                self.seed(obj, brng = algo_name)
                return
            raise ValueError("The argument to set_state must be a list of 2 elements")

        algorithm_name = state[0]
        if algorithm_name not in _brng_dict.keys():
            raise ValueError("basic number generator algorithm must be one of ['" + "',".join(_brng_dict.keys()) + "']")

        stream_buf = state[1]
        if not is_bytes_object(stream_buf):
            raise ValueError('state is expected to be bytes')

        bytes_ptr = py_bytes_DataPtr(stream_buf)

        with self.lock:
            err = irk_set_state_mkl(self.internal_state, bytes_ptr)
            if(err):
                raise ValueError('The stream state buffer is corrupted')
            brng_id = irk_get_brng_mkl(self.internal_state)
            if (_brng_dict[algorithm_name] != brng_id):
                raise ValueError('The algorithm name does not match content of the buffer')

    # Pickling support:
    def __getstate__(self):
        return self.get_state()

    def __setstate__(self, state):
        self.set_state(state)

    def __reduce__(self):
        global __RandomState_ctor
        return (__RandomState_ctor, (), self.get_state())

    def leapfrog(self, int k, int nstreams):
        """
        leapfrog(k, nstreams)

        Initializes the current state generator using leap-frog method,
        if supported for the basic random pseudo-random number generation
        algorithm.
        """
        cdef int err, brng_id

        err = irk_leapfrog_stream_mkl(self.internal_state, k, nstreams);

        if err == -1:
            raise ValueError('The stream state buffer is corrupted')
        elif err == 1:
            with self.lock:
                brng_id = irk_get_brng_mkl(self.internal_state)
            raise ValueError("Leap-frog method of stream initialization is not supported for " + str(_brng_id_to_name(brng_id)))

    def skipahead(self, long long int nskips):
        """
        skipahead(nskips)

        Initializes the current state generator using skip-ahead method,
        if supported for the basic random pseudo-random number generation
        algorithm.
        """
        cdef int err, brng_id

        err = irk_skipahead_stream_mkl(self.internal_state, nskips);

        if err == -1:
            raise ValueError('The stream state buffer is corrupted')
        elif err == 1:
            with self.lock:
                brng_id = irk_get_brng_mkl(self.internal_state)
            raise ValueError("Skip-ahead method of stream initialization is not supported for " + str(_brng_id_to_name(brng_id)))

    # Basic distributions:
    def random_sample(self, size=None):
        """
        random_sample(size=None)

        Return random floats in the half-open interval [0.0, 1.0).

        Results are from the "continuous uniform" distribution over the
        stated interval.  To sample :math:`Unif[a, b), b > a` multiply
        the output of `random_sample` by `(b-a)` and add `a`::

          (b - a) * random_sample() + a

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : float or ndarray of floats
            Array of random floats of shape `size` (unless ``size=None``, in which
            case a single float is returned).

        Examples
        --------
        >>> mkl_random.random_sample()
        0.47108547995356098
        >>> type(mkl_random.random_sample())
        <type 'float'>
        >>> mkl_random.random_sample((5,))
        array([ 0.30220482,  0.86820401,  0.1654503 ,  0.11659149,  0.54323428])

        Three-by-two array of random numbers from [-5, 0):

        >>> 5 * mkl_random.random_sample((3, 2)) - 5
        array([[-3.99149989, -0.52338984],
               [-2.99091858, -0.79479508],
               [-1.23204345, -1.75224494]])

        """
        return vec_cont0_array(self.internal_state, irk_double_vec, size, self.lock)

    def tomaxint(self, size=None):
        """
        tomaxint(size=None)

        Return a sample of uniformly distributed random integers in the interval
        [0, ``np.iinfo("long").max``].

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : ndarray
            Drawn samples, with shape `size`.

        See Also
        --------
        randint : Uniform sampling over a given half-open interval of integers.
        random_integers : Uniform sampling over a given closed interval of
            integers.

        Examples
        --------
        >>> RS = np.random_random_intel.RandomState() # need a RandomState object
        >>> RS.tomaxint((2,2,2))
        array([[[1170048599, 1600360186],
                [ 739731006, 1947757578]],
               [[1871712945,  752307660],
                [1601631370, 1479324245]]])
        >>> import sys
        >>> sys.maxint
        2147483647
        >>> RS.tomaxint((2,2,2)) < sys.maxint
        array([[[ True,  True],
                [ True,  True]],
               [[ True,  True],
                [ True,  True]]], dtype=bool)

        """
        return vec_long_disc0_array(self.internal_state, irk_long_vec, size, self.lock)

    # Set up dictionary of integer types and relevant functions.
    #
    # The dictionary is keyed by dtype(...).name and the values
    # are a tuple (low, high, function), where low and high are
    # the bounds of the largest half open interval `[low, high)`
    # and the function is the relevant function to call for
    # that precision.
    #
    # The functions are all the same except for changed types in
    # a few places. It would be easy to template them.

    def _choose_randint_type(self, dtype):
        _randint_type = {
            'bool': (0, 2, self._rand_bool),
            'int8': (-2**7, 2**7, self._rand_int8),
            'int16': (-2**15, 2**15, self._rand_int16),
            'int32': (-2**31, 2**31, self._rand_int32),
            'int64': (-2**63, 2**63, self._rand_int64),
            'uint8': (0, 2**8, self._rand_uint8),
            'uint16': (0, 2**16, self._rand_uint16),
            'uint32': (0, 2**32, self._rand_uint32),
            'uint64': (0, 2**64, self._rand_uint64)
        }

        key = np.dtype(dtype).name
        if not key in _randint_type:
            raise TypeError('Unsupported dtype "%s" for randint' % key)
        return _randint_type[key]

    # generates typed random integer in [low, high]
    def _rand_bool(self, cnp.npy_bool low, cnp.npy_bool high, size):
        """
        _rand_bool(low, high, size)

        See `_rand_int32` for documentation, only the return type changes.

        """
        cdef cnp.npy_bool buf
        cdef cnp.npy_bool *out
        cdef cnp.ndarray array "arrayObject"
        cdef cnp.npy_intp cnt

        if size is None:
            irk_rand_bool_vec(self.internal_state, 1, &buf, low, high)
            return np.bool_(buf)
        else:
            array = <cnp.ndarray>np.empty(size, np.bool_)
            cnt = cnp.PyArray_SIZE(array)
            out = <cnp.npy_bool *>cnp.PyArray_DATA(array)
            with nogil:
                irk_rand_bool_vec(self.internal_state, cnt, out, low, high)
            return array


    def _rand_int8(self, cnp.npy_int8 low, cnp.npy_int8 high, size):
        """
        _rand_int8(low, high, size)

        See `_rand_int32` for documentation, only the return type changes.

        """
        cdef cnp.npy_int8 buf
        cdef cnp.npy_int8 *out
        cdef cnp.ndarray array "arrayObject"
        cdef cnp.npy_intp cnt

        if size is None:
            irk_rand_int8_vec(self.internal_state, 1, &buf, low, high)
            return np.int8(<cnp.npy_int8>buf)
        else:
            array = <cnp.ndarray>np.empty(size, np.int8)
            cnt = cnp.PyArray_SIZE(array)
            out = <cnp.npy_int8 *>cnp.PyArray_DATA(array)
            with nogil:
                irk_rand_int8_vec(self.internal_state, cnt, out, low, high)
            return array


    def _rand_int16(self, cnp.npy_int16 low, cnp.npy_int16 high, size):
        """
        _rand_int16(low, high, size)

        See `_rand_int32` for documentation, only the return type changes.

        """
        cdef cnp.npy_int16 buf
        cdef cnp.npy_int16 *out
        cdef cnp.ndarray array "arrayObject"
        cdef cnp.npy_intp cnt

        if size is None:
            irk_rand_int16_vec(self.internal_state, 1, &buf, low, high)
            return np.int16(<cnp.npy_int16>buf)
        else:
            array = <cnp.ndarray>np.empty(size, np.int16)
            cnt = cnp.PyArray_SIZE(array)
            out = <cnp.npy_int16 *>cnp.PyArray_DATA(array)
            with nogil:
                irk_rand_int16_vec(self.internal_state, cnt, out, low, high)
            return array


    def _rand_int32(self, cnp.npy_int32 low, cnp.npy_int32 high, size):
        """
        _rand_int32(self, low, high, size)

        Return random np.int32 integers between `low` and `high`, inclusive.

        Return random integers from the "discrete uniform" distribution in the
        closed interval [`low`, `high`]. On entry the arguments are presumed
        to have been validated for size and order for the np.int32 type.

        Parameters
        ----------
        low : int
            Lowest (signed) integer to be drawn from the distribution.
        high : int
            Highest (signed) integer to be drawn from the distribution.
        size : int or tuple of ints
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : python scalar or ndarray of np.int32
              `size`-shaped array of random integers from the appropriate
              distribution, or a single such random int if `size` not provided.

        """
        cdef cnp.npy_int32 buf
        cdef cnp.npy_int32 *out
        cdef cnp.ndarray array "arrayObject"
        cdef cnp.npy_intp cnt

        if size is None:
            irk_rand_int32_vec(self.internal_state, 1, &buf, low, high)
            return np.int32(buf)
        else:
            array = <cnp.ndarray>np.empty(size, np.int32)
            cnt = cnp.PyArray_SIZE(array)
            out = <cnp.npy_int32 *>cnp.PyArray_DATA(array)
            with nogil:
                irk_rand_int32_vec(self.internal_state, cnt, out, low, high)
            return array


    def _rand_int64(self, cnp.npy_int64 low, cnp.npy_int64 high, size):
        """
        _rand_int64(low, high, size)

        See `_rand_int32` for documentation, only the return type changes.

        """
        cdef cnp.npy_int64 buf
        cdef cnp.npy_int64 *out
        cdef cnp.ndarray array "arrayObject"
        cdef cnp.npy_intp cnt

        if size is None:
            irk_rand_int64_vec(self.internal_state, 1, &buf, low, high)
            return np.int64(buf)
        else:
            array = <cnp.ndarray>np.empty(size, np.int64)
            cnt = cnp.PyArray_SIZE(array)
            out = <cnp.npy_int64 *>cnp.PyArray_DATA(array)
            with nogil:
                irk_rand_int64_vec(self.internal_state, cnt, out, low, high)
            return array

    def _rand_uint8(self, cnp.npy_uint8 low, cnp.npy_uint8 high, size):
        """
        _rand_uint8(low, high, size)

        See `_rand_int32` for documentation, only the return type changes.

        """
        cdef cnp.npy_uint8 buf
        cdef cnp.npy_uint8 *out
        cdef cnp.ndarray array "arrayObject"
        cdef cnp.npy_intp cnt

        if size is None:
            irk_rand_uint8_vec(self.internal_state, 1, &buf, low, high)
            return np.uint8(buf)
        else:
            array = <cnp.ndarray>np.empty(size, np.uint8)
            cnt = cnp.PyArray_SIZE(array)
            out = <cnp.npy_uint8 *>cnp.PyArray_DATA(array)
            with nogil:
                irk_rand_uint8_vec(self.internal_state, cnt, out, low, high)
            return array


    def _rand_uint16(self, cnp.npy_uint16 low, cnp.npy_uint16 high, size):
        """
        _rand_uint16(low, high, size)

        See `_rand_int32` for documentation, only the return type changes.

        """
        cdef cnp.npy_uint16 off, rng, buf
        cdef cnp.npy_uint16 *out
        cdef cnp.ndarray array "arrayObject"
        cdef cnp.npy_intp cnt

        if size is None:
            irk_rand_uint16_vec(self.internal_state, 1, &buf, low, high)
            return np.uint16(buf)
        else:
            array = <cnp.ndarray>np.empty(size, np.uint16)
            cnt = cnp.PyArray_SIZE(array)
            out = <cnp.npy_uint16 *>cnp.PyArray_DATA(array)
            with nogil:
                irk_rand_uint16_vec(self.internal_state, cnt, out, low, high)
            return array


    def _rand_uint32(self, cnp.npy_uint32 low, cnp.npy_uint32 high, size):
        """
        _rand_uint32(self, low, high, size)

        See `_rand_int32` for documentation, only the return type changes.

        """
        cdef cnp.npy_uint32 buf
        cdef cnp.npy_uint32 *out
        cdef cnp.ndarray array "arrayObject"
        cdef cnp.npy_intp cnt

        if size is None:
            irk_rand_uint32_vec(self.internal_state, 1, &buf, low, high)
            return np.uint32(buf)
        else:
            array = <cnp.ndarray>np.empty(size, np.uint32)
            cnt = cnp.PyArray_SIZE(array)
            out = <cnp.npy_uint32 *>cnp.PyArray_DATA(array)
            with nogil:
                irk_rand_uint32_vec(self.internal_state, cnt, out, low, high)
            return array


    def _rand_uint64(self, cnp.npy_uint64 low, cnp.npy_uint64 high, size):
        """
        _rand_uint64(low, high, size)

        See `_rand_int32` for documentation, only the return type changes.

        """
        cdef cnp.npy_uint64 buf
        cdef cnp.npy_uint64 *out
        cdef cnp.ndarray array "arrayObject"
        cdef cnp.npy_intp cnt

        if size is None:
            irk_rand_uint64_vec(self.internal_state, 1, &buf, low, high)
            return np.uint64(buf)
        else:
            array = <cnp.ndarray>np.empty(size, np.uint64)
            cnt = cnp.PyArray_SIZE(array)
            out = <cnp.npy_uint64 *>cnp.PyArray_DATA(array)
            with nogil:
                irk_rand_uint64_vec(self.internal_state, cnt, out, low, high)
            return array


    def randint(self, low, high=None, size=None, dtype=int):
        """
        randint(low, high=None, size=None, dtype=int)

        Return random integers from `low` (inclusive) to `high` (exclusive).

        Return random integers from the "discrete uniform" distribution of
        the specified dtype in the "half-open" interval [`low`, `high`). If
        `high` is None (the default), then results are from [0, `low`).

        Parameters
        ----------
        low : int
            Lowest (signed) integer to be drawn from the distribution (unless
            ``high=None``, in which case this parameter is the *highest* such
            integer).
        high : int, optional
            If provided, one above the largest (signed) integer to be drawn
            from the distribution (see above for behavior if ``high=None``).
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        dtype : dtype, optional
            Desired dtype of the result. All dtypes are determined by their
            name, i.e., 'int64', 'int', etc, so byteorder is not available
            and a specific precision may have different C types depending
            on the platform. The default value is 'np.int'.

            .. versionadded:: 1.11.0

        Returns
        -------
        out : int or ndarray of ints
            `size`-shaped array of random integers from the appropriate
            distribution, or a single such random int if `size` not provided.

        See Also
        --------
        random.random_integers : similar to `randint`, only for the closed
            interval [`low`, `high`], and 1 is the lowest value if `high` is
            omitted. In particular, this other one is the one to use to generate
            uniformly distributed discrete non-integers.

        Examples
        --------
        >>> mkl_random.randint(2, size=10)
        array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0])
        >>> mkl_random.randint(1, size=10)
        array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        Generate a 2 x 4 array of ints between 0 and 4, inclusive:

        >>> mkl_random.randint(5, size=(2, 4))
        array([[4, 0, 2, 1],
               [3, 2, 2, 0]])

        """
        if high is None:
            high = low
            low = 0

        lowbnd, highbnd, randfunc = self._choose_randint_type(dtype)

        if low < lowbnd:
            raise ValueError("low is out of bounds for %s" % (np.dtype(dtype).name,))
        if high > highbnd:
            raise ValueError("high is out of bounds for %s" % (np.dtype(dtype).name,))
        if low >= high:
            raise ValueError("low >= high")

        with self.lock:
            ret = randfunc(low, high - 1, size)

        if size is None:
            if dtype in (bool, int):
                return dtype(ret)

        return ret


    def randint_untyped(self, low, high=None, size=None):
        """
        randint_untyped(low, high=None, size=None, dtype=int)

        Return random integers from `low` (inclusive) to `high` (exclusive).

        Return random integers from the "discrete uniform" distribution of
        the specified dtype in the "half-open" interval [`low`, `high`). If
        `high` is None (the default), then results are from [0, `low`).

        Parameters
        ----------
        low : int
            Lowest (signed) integer to be drawn from the distribution (unless
            ``high=None``, in which case this parameter is the *highest* such
            integer).
        high : int, optional
            If provided, one above the largest (signed) integer to be drawn
            from the distribution (see above for behavior if ``high=None``).
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : int or ndarray of ints
            `size`-shaped array of random integers from the appropriate
            distribution, or a single such random int if `size` not provided.

        See Also
        --------
        random.random_integers : similar to `randint`, only for the closed
            interval [`low`, `high`], and 1 is the lowest value if `high` is
            omitted. In particular, this other one is the one to use to generate
            uniformly distributed discrete non-integers.

        Examples
        --------
        >>> mkl_random.randint(2, size=10)
        array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0])
        >>> mkl_random.randint(1, size=10)
        array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        Generate a 2 x 4 array of ints between 0 and 4, inclusive:

        >>> mkl_random.randint(5, size=(2, 4))
        array([[4, 0, 2, 1],
               [3, 2, 2, 0]])

        """
        cdef long lo, hi
        cdef long *array_long_data
        cdef int * array_int_data
        cdef cnp.ndarray array "arrayObject"
        cdef cnp.npy_intp length
        cdef int rv_int
        cdef long rv_long

        if high is None:
            lo = 0
            hi = low
        else:
            lo = low
            hi = high

        if lo >= hi :
            raise ValueError("low >= high")

        if ((<int> lo) == lo) and ((<int>hi) == hi):
            if size is None:
                irk_discrete_uniform_vec(self.internal_state, 1, &rv_int, <int>lo, <int>hi)
                return rv_int
            else:
                array = <cnp.ndarray>np.empty(size, np.int32)
                length = cnp.PyArray_SIZE(array)
                array_int_data = <int*>cnp.PyArray_DATA(array)
                with self.lock, nogil:
                    irk_discrete_uniform_vec(self.internal_state, length, array_int_data, <int>lo, <int>hi)
                return array
        else:
            if size is None:
                irk_discrete_uniform_long_vec(self.internal_state, 1, &rv_long, lo, hi)
                return rv_long
            else:
                array = <cnp.ndarray>np.empty(size, int)
                length = cnp.PyArray_SIZE(array)
                array_long_data = <long*>cnp.PyArray_DATA(array)
                with self.lock, nogil:
                    irk_discrete_uniform_long_vec(self.internal_state, length, array_long_data, lo, hi)
                return array

    def bytes(self, cnp.npy_intp length):
        """
        bytes(length)

        Return random bytes.

        Parameters
        ----------
        length : int
            Number of random bytes.

        Returns
        -------
        out : str
            String of length `length`.

        Examples
        --------
        >>> mkl_random.bytes(10)
        ' eh\\x85\\x022SZ\\xbf\\xa4' #random

        """
        cdef void *bytes
        bytestring = empty_py_bytes(length, &bytes)
        with self.lock, nogil:
            irk_fill(bytes, length, self.internal_state)
        return bytestring


    def choice(self, a, size=None, replace=True, p=None):
        """
        choice(a, size=None, replace=True, p=None)

        Generates a random sample from a given 1-D array

                .. versionadded:: 1.7.0

        Parameters
        -----------
        a : 1-D array-like or int
            If an ndarray, a random sample is generated from its elements.
            If an int, the random sample is generated as if a was np.arange(n)
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        replace : boolean, optional
            Whether the sample is with or without replacement
        p : 1-D array-like, optional
            The probabilities associated with each entry in a.
            If not given the sample assumes a uniform distribution over all
            entries in a.

        Returns
        --------
        samples : 1-D ndarray, shape (size,)
            The generated random samples

        Raises
        -------
        ValueError
            If a is an int and less than zero, if a or p are not 1-dimensional,
            if a is an array-like of size 0, if p is not a vector of
            probabilities, if a and p have different lengths, or if
            replace=False and the sample size is greater than the population
            size

        See Also
        ---------
        randint, shuffle, permutation

        Examples
        ---------
        Generate a uniform random sample from np.arange(5) of size 3:

        >>> mkl_random.choice(5, 3)
        array([0, 3, 4])
        >>> #This is equivalent to mkl_random.randint(0,5,3)

        Generate a non-uniform random sample from np.arange(5) of size 3:

        >>> mkl_random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
        array([3, 3, 0])

        Generate a uniform random sample from np.arange(5) of size 3 without
        replacement:

        >>> mkl_random.choice(5, 3, replace=False)
        array([3,1,0])
        >>> #This is equivalent to mkl_random.permutation(np.arange(5))[:3]

        Generate a non-uniform random sample from np.arange(5) of size
        3 without replacement:

        >>> mkl_random.choice(5, 3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])
        array([2, 3, 0])

        Any of the above can be repeated with an arbitrary array-like
        instead of just integers. For instance:

        >>> aa_milne_arr = ['pooh', 'rabbit', 'piglet', 'Christopher']
        >>> mkl_random.choice(aa_milne_arr, 5, p=[0.5, 0.1, 0.1, 0.3])
        array(['pooh', 'pooh', 'pooh', 'Christopher', 'piglet'],
              dtype='|S11')

        """
        cdef double *pix

        # Format and Verify input
        a = np.asarray(a)
        if a.ndim == 0:
            try:
                # __index__ must return an integer by python rules.
                pop_size = operator.index(a.item())
            except TypeError:
                raise ValueError("a must be 1-dimensional or an integer")
            if pop_size <= 0:
                raise ValueError("a must be greater than 0")
        elif a.ndim != 1:
            raise ValueError("a must be 1-dimensional")
        else:
            pop_size = a.shape[0]
            if pop_size is 0:
                raise ValueError("a must be non-empty")

        if p is not None:
            d = len(p)

            atol = np.sqrt(np.finfo(np.float64).eps)
            if isinstance(p, np.ndarray):
                if np.issubdtype(p.dtype, np.floating):
                    atol = max(atol, np.sqrt(np.finfo(p.dtype).eps))

            p = <cnp.ndarray>cnp.PyArray_ContiguousFromObject(p, cnp.NPY_DOUBLE, 1, 1)
            pix = <double*>cnp.PyArray_DATA(p)

            if p.ndim != 1:
                raise ValueError("p must be 1-dimensional")
            if p.size != pop_size:
                raise ValueError("a and p must have same size")
            if np.logical_or.reduce(p < 0):
                raise ValueError("probabilities are not non-negative")
            if abs(kahan_sum(pix, d) - 1.) > atol:
                raise ValueError("probabilities do not sum to 1")

        shape = size
        if shape is not None:
            size = np.prod(shape, dtype=np.intp)
        else:
            size = 1

        # Actual sampling
        if replace:
            if p is not None:
                cdf = p.cumsum()
                cdf /= cdf[-1]
                uniform_samples = self.random_sample(shape)
                idx = cdf.searchsorted(uniform_samples, side='right')
                idx = np.asarray(idx) # searchsorted returns a scalar
            else:
                idx = self.randint(0, pop_size, size=shape)
        else:
            if size > pop_size:
                raise ValueError("Cannot take a larger sample than "
                                 "population when 'replace=False'")

            if p is not None:
                if np.count_nonzero(p > 0) < size:
                    raise ValueError("Fewer non-zero entries in p than size")
                n_uniq = 0
                p = p.copy()
                found = np.zeros(tuple() if shape is None else shape, dtype=np.int64)
                flat_found = found.ravel()
                while n_uniq < size:
                    x = self.rand(size - n_uniq)
                    if n_uniq > 0:
                        p[flat_found[0:n_uniq]] = 0
                    cdf = np.cumsum(p)
                    cdf /= cdf[-1]
                    new = cdf.searchsorted(x, side='right')
                    _, unique_indices = np.unique(new, return_index=True)
                    unique_indices.sort()
                    new = new.take(unique_indices)
                    flat_found[n_uniq:n_uniq + new.size] = new
                    n_uniq += new.size
                idx = found
            else:
                idx = self.permutation(pop_size)[:size]
                if shape is not None:
                    idx.shape = shape

        if shape is None and isinstance(idx, np.ndarray):
            # In most cases a scalar will have been made an array
            idx = idx.item(0)

        #Use samples as indices for a if a is array-like
        if a.ndim == 0:
            return idx

        if shape is not None and idx.ndim == 0:
            # If size == () then the user requested a 0-d array as opposed to
            # a scalar object when size is None. However a[idx] is always a
            # scalar and not an array. So this makes sure the result is an
            # array, taking into account that np.array(item) may not work
            # for object arrays.
            res = np.empty((), dtype=a.dtype)
            res[()] = a[idx]
            return res

        return a[idx]


    def uniform(self, low=0.0, high=1.0, size=None):
        """
        uniform(low=0.0, high=1.0, size=None)

        Draw samples from a uniform distribution.

        Samples are uniformly distributed over the half-open interval
        ``[low, high)`` (includes low, but excludes high).  In other words,
        any value within the given interval is equally likely to be drawn
        by `uniform`.

        Parameters
        ----------
        low : float, optional
            Lower boundary of the output interval.  All values generated will be
            greater than or equal to low.  The default value is 0.
        high : float
            Upper boundary of the output interval.  All values generated will be
            less than high.  The default value is 1.0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : ndarray
            Drawn samples, with shape `size`.

        See Also
        --------
        randint : Discrete uniform distribution, yielding integers.
        random_integers : Discrete uniform distribution over the closed
                          interval ``[low, high]``.
        random_sample : Floats uniformly distributed over ``[0, 1)``.
        random : Alias for `random_sample`.
        rand : Convenience function that accepts dimensions as input, e.g.,
               ``rand(2,2)`` would generate a 2-by-2 array of floats,
               uniformly distributed over ``[0, 1)``.

        Notes
        -----
        The probability density function of the uniform distribution is

        .. math:: p(x) = \\frac{1}{b - a}

        anywhere within the interval ``[a, b)``, and zero elsewhere.

        Examples
        --------
        Draw samples from the distribution:

        >>> s = mkl_random.uniform(-1,0,1000)

        All values are within the given interval:

        >>> np.all(s >= -1)
        True
        >>> np.all(s < 0)
        True

        Display the histogram of the samples, along with the
        probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 15, normed=True)
        >>> plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
        >>> plt.show()

        """
        cdef cnp.ndarray olow, ohigh
        cdef double flow, fhigh
        cdef object temp

        flow = PyFloat_AsDouble(low)
        fhigh = PyFloat_AsDouble(high)
        if not npy_isfinite(flow) or not npy_isfinite(fhigh):
            raise OverflowError('Range exceeds valid bounds')
        if flow >= fhigh:
            raise ValueError("low >= high")

        if not PyErr_Occurred():
            return vec_cont2_array_sc(self.internal_state, irk_uniform_vec, size, flow,
                                  fhigh, self.lock)

        PyErr_Clear()
        olow = <cnp.ndarray>cnp.PyArray_FROM_OTF(low, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        ohigh = <cnp.ndarray>cnp.PyArray_FROM_OTF(high, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)

        if not np.all(np.isfinite(olow)) or not np.all(np.isfinite(ohigh)):
            raise OverflowError('Range exceeds valid bounds')

        if np.any(olow >= ohigh):
            raise ValueError("low >= high")

        return vec_cont2_array(self.internal_state, irk_uniform_vec, size, olow, ohigh,
                           self.lock)

    def rand(self, *args):
        """
        rand(d0, d1, ..., dn)

        Random values in a given shape.

        Create an array of the given shape and propagate it with
        random samples from a uniform distribution
        over ``[0, 1)``.

        Parameters
        ----------
        d0, d1, ..., dn : int, optional
            The dimensions of the returned array, should all be positive.
            If no argument is given a single Python float is returned.

        Returns
        -------
        out : ndarray, shape ``(d0, d1, ..., dn)``
            Random values.

        See Also
        --------
        random

        Notes
        -----
        This is a convenience function. If you want an interface that
        takes a shape-tuple as the first argument, refer to
        mkl_random.random_sample .

        Examples
        --------
        >>> mkl_random.rand(3,2)
        array([[ 0.14022471,  0.96360618],  #random
               [ 0.37601032,  0.25528411],  #random
               [ 0.49313049,  0.94909878]]) #random

        """
        if len(args) == 0:
            return self.random_sample()
        else:
            return self.random_sample(size=args)

    def randn(self, *args):
        """
        randn(d0, d1, ..., dn)

        Return a sample (or samples) from the "standard normal" distribution.

        If positive, int_like or int-convertible arguments are provided,
        `randn` generates an array of shape ``(d0, d1, ..., dn)``, filled
        with random floats sampled from a univariate "normal" (Gaussian)
        distribution of mean 0 and variance 1 (if any of the :math:`d_i` are
        floats, they are first converted to integers by truncation). A single
        float randomly sampled from the distribution is returned if no
        argument is provided.

        This is a convenience function.  If you want an interface that takes a
        tuple as the first argument, use `numpy.random.standard_normal` instead.

        Parameters
        ----------
        d0, d1, ..., dn : int, optional
            The dimensions of the returned array, should be all positive.
            If no argument is given a single Python float is returned.

        Returns
        -------
        Z : ndarray or float
            A ``(d0, d1, ..., dn)``-shaped array of floating-point samples from
            the standard normal distribution, or a single such float if
            no parameters were supplied.

        See Also
        --------
        random.standard_normal : Similar, but takes a tuple as its argument.

        Notes
        -----
        For random samples from :math:`N(\\mu, \\sigma^2)`, use:

        ``sigma * mkl_random.randn(...) + mu``

        Examples
        --------
        >>> mkl_random.randn()
        2.1923875335537315 #random

        Two-by-four array of samples from N(3, 6.25):

        >>> 2.5 * mkl_random.randn(2, 4) + 3
        array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],  #random
               [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]]) #random

        """
        if len(args) == 0:
            return self.standard_normal()
        else:
            return self.standard_normal(args)


    def random_integers(self, low, high=None, size=None):
        """
        random_integers(low, high=None, size=None)

        Random integers of type np.int between `low` and `high`, inclusive.

        Return random integers of type np.int from the "discrete uniform"
        distribution in the closed interval [`low`, `high`].  If `high` is
        None (the default), then results are from [1, `low`]. The np.int
        type translates to the C long type used by Python 2 for "short"
        integers and its precision is platform dependent.

        This function has been deprecated. Use randint instead.

        .. deprecated:: 1.11.0

        Parameters
        ----------
        low : int
            Lowest (signed) integer to be drawn from the distribution (unless
            ``high=None``, in which case this parameter is the *highest* such
            integer).
        high : int, optional
            If provided, the largest (signed) integer to be drawn from the
            distribution (see above for behavior if ``high=None``).
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : int or ndarray of ints
            `size`-shaped array of random integers from the appropriate
            distribution, or a single such random int if `size` not provided.

        See Also
        --------
        random.randint : Similar to `random_integers`, only for the half-open
            interval [`low`, `high`), and 0 is the lowest value if `high` is
            omitted.

        Notes
        -----
        To sample from N evenly spaced floating-point numbers between a and b,
        use::

          a + (b - a) * (mkl_random.random_integers(N) - 1) / (N - 1.)

        Examples
        --------
        >>> mkl_random.random_integers(5)
        4
        >>> type(mkl_random.random_integers(5))
        <type 'int'>
        >>> mkl_random.random_integers(5, size=(3.,2.))
        array([[5, 4],
               [3, 3],
               [4, 5]])

        Choose five random numbers from the set of five evenly-spaced
        numbers between 0 and 2.5, inclusive (*i.e.*, from the set
        :math:`{0, 5/8, 10/8, 15/8, 20/8}`):

        >>> 2.5 * (mkl_random.random_integers(5, size=(5,)) - 1) / 4.
        array([ 0.625,  1.25 ,  0.625,  0.625,  2.5  ])

        Roll two six sided dice 1000 times and sum the results:

        >>> d1 = mkl_random.random_integers(1, 6, 1000)
        >>> d2 = mkl_random.random_integers(1, 6, 1000)
        >>> dsums = d1 + d2

        Display results as a histogram:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(dsums, 11, normed=True)
        >>> plt.show()

        """
        if high is None:
            warnings.warn(("This function is deprecated. Please call "
                           "randint(1, {low} + 1) instead".format(low=low)),
                          DeprecationWarning)
            high = low
            low = 1

        else:
            warnings.warn(("This function is deprecated. Please call "
                           "randint({low}, {high} + 1) instead".format(
                low=low, high=high)), DeprecationWarning)

        return self.randint(low, high + 1, size=size, dtype='l')

    # Complicated, continuous distributions:
    def standard_normal(self, size=None, method=ICDF):
        """
        standard_normal(size=None, method='ICDF')

        Draw samples from a standard Normal distribution (mean=0, stdev=1).

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        method : 'ICDF, 'BoxMuller', 'BoxMuller2', optional
            Sampling method used by Intel MKL. Can also be specified using
            tokens mkl_random.ICDF, mkl_random.BOXMULLER, mkl_random.BOXMULLER2

        Returns
        -------
        out : float or ndarray
            Drawn samples.

        Examples
        --------
        >>> s = mkl_random.standard_normal(8000)
        >>> s
        array([ 0.6888893 ,  0.78096262, -0.89086505, ...,  0.49876311, #random
               -0.38672696, -0.4685006 ])                               #random
        >>> s.shape
        (8000,)
        >>> s = mkl_random.standard_normal(size=(3, 4, 2))
        >>> s.shape
        (3, 4, 2)

        """
        method = choose_method(method, [ICDF, BOXMULLER, BOXMULLER2], _method_alias_dict_gaussian)
        if method is ICDF:
            return vec_cont0_array(self.internal_state, irk_standard_normal_vec_ICDF, size, self.lock)
        elif method is BOXMULLER2:
            return vec_cont0_array(self.internal_state, irk_standard_normal_vec_BM2, size, self.lock)
        else:
            return vec_cont0_array(self.internal_state, irk_standard_normal_vec_BM1, size, self.lock);

    def normal(self, loc=0.0, scale=1.0, size=None, method=ICDF):
        """
        normal(loc=0.0, scale=1.0, size=None, method='ICDF')

        Draw random samples from a normal (Gaussian) distribution.

        The probability density function of the normal distribution, first
        derived by De Moivre and 200 years later by both Gauss and Laplace
        independently [2]_, is often called the bell curve because of
        its characteristic shape (see the example below).

        The normal distributions occurs often in nature.  For example, it
        describes the commonly occurring distribution of samples influenced
        by a large number of tiny, random disturbances, each with its own
        unique distribution [2]_.

        Parameters
        ----------
        loc : float
            Mean ("centre") of the distribution.
        scale : float
            Standard deviation (spread or "width") of the distribution.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        method : 'ICDF, 'BoxMuller', 'BoxMuller2', optional
            Sampling method used by Intel MKL. Can also be specified using
            tokens mkl_random.ICDF, mkl_random.BOXMULLER, mkl_random.BOXMULLER2

        See Also
        --------
        scipy.stats.distributions.norm : probability density function,
            distribution or cumulative density function, etc.

        Notes
        -----
        The probability density for the Gaussian distribution is

        .. math:: p(x) = \\frac{1}{\\sqrt{ 2 \\pi \\sigma^2 }}
                         e^{ - \\frac{ (x - \\mu)^2 } {2 \\sigma^2} },

        where :math:`\\mu` is the mean and :math:`\\sigma` the standard
        deviation. The square of the standard deviation, :math:`\\sigma^2`,
        is called the variance.

        The function has its peak at the mean, and its "spread" increases with
        the standard deviation (the function reaches 0.607 times its maximum at
        :math:`x + \\sigma` and :math:`x - \\sigma` [2]_).  This implies that
        `numpy.random.normal` is more likely to return samples lying close to
        the mean, rather than those far away.

        References
        ----------
        .. [1] Wikipedia, "Normal distribution",
               http://en.wikipedia.org/wiki/Normal_distribution
        .. [2] P. R. Peebles Jr., "Central Limit Theorem" in "Probability,
               Random Variables and Random Signal Principles", 4th ed., 2001,
               pp. 51, 51, 125.

        Examples
        --------
        Draw samples from the distribution:

        >>> mu, sigma = 0, 0.1 # mean and standard deviation
        >>> s = mkl_random.normal(mu, sigma, 1000)

        Verify the mean and the variance:

        >>> abs(mu - np.mean(s)) < 0.01
        True

        >>> abs(sigma - np.std(s, ddof=1)) < 0.01
        True

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 30, normed=True)
        >>> plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
        ...                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
        ...          linewidth=2, color='r')
        >>> plt.show()

        """
        cdef cnp.ndarray oloc, oscale
        cdef double floc, fscale

        floc = PyFloat_AsDouble(loc)
        fscale = PyFloat_AsDouble(scale)
        if not PyErr_Occurred():
            if fscale <= 0:
                raise ValueError("scale <= 0")
            method = choose_method(method, [ICDF, BOXMULLER, BOXMULLER2], _method_alias_dict_gaussian)
            if method is ICDF:
                return vec_cont2_array_sc(self.internal_state, irk_normal_vec_ICDF, size, floc, fscale, self.lock)
            elif method is BOXMULLER2:
                return vec_cont2_array_sc(self.internal_state, irk_normal_vec_BM2, size, floc, fscale, self.lock)
            else:
                return vec_cont2_array_sc(self.internal_state, irk_normal_vec_BM1, size, floc, fscale, self.lock)

        PyErr_Clear()

        oloc = <cnp.ndarray>cnp.PyArray_FROM_OTF(loc, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        oscale = <cnp.ndarray>cnp.PyArray_FROM_OTF(scale, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        if np.any(np.less_equal(oscale, 0)):
            raise ValueError("scale <= 0")
        method = choose_method(method, [ICDF, BOXMULLER, BOXMULLER2], _method_alias_dict_gaussian)
        if method is ICDF:
            return vec_cont2_array(self.internal_state, irk_normal_vec_ICDF, size, oloc, oscale, self.lock)
        elif method is BOXMULLER2:
            return vec_cont2_array(self.internal_state, irk_normal_vec_BM2, size, oloc, oscale, self.lock)
        else:
            return vec_cont2_array(self.internal_state, irk_normal_vec_BM1, size, oloc, oscale, self.lock)

    def beta(self, a, b, size=None):
        """
        beta(a, b, size=None)

        Draw samples from a Beta distribution.

        The Beta distribution is a special case of the Dirichlet distribution
        and is related to the Gamma distribution.  It has the probability
        distribution function

        .. math:: f(x; a,b) = \\frac{1}{B(\\alpha, \\beta)} x^{\\alpha - 1}
                                                         (1 - x)^{\\beta - 1},

        where the normalisation, B, is the beta function,

        .. math:: B(\\alpha, \\beta) = \\int_0^1 t^{\\alpha - 1}
                                     (1 - t)^{\\beta - 1} dt.

        It is often seen in Bayesian inference and order statistics.

        Parameters
        ----------
        a : float
            Alpha, non-negative.
        b : float
            Beta, non-negative.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : ndarray
            Array of the given shape, containing values drawn from a
            Beta distribution.

        """
        cdef cnp.ndarray oa, ob
        cdef double fa, fb

        fa = PyFloat_AsDouble(a)
        fb = PyFloat_AsDouble(b)
        if not PyErr_Occurred():
            if fa <= 0:
                raise ValueError("a <= 0")
            if fb <= 0:
                raise ValueError("b <= 0")
            return vec_cont2_array_sc(self.internal_state, irk_beta_vec, size, fa, fb,
                                  self.lock)

        PyErr_Clear()

        oa = <cnp.ndarray>cnp.PyArray_FROM_OTF(a, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        ob = <cnp.ndarray>cnp.PyArray_FROM_OTF(b, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        if np.any(np.less_equal(oa, 0)):
            raise ValueError("a <= 0")
        if np.any(np.less_equal(ob, 0)):
            raise ValueError("b <= 0")
        return vec_cont2_array(self.internal_state, irk_beta_vec, size, oa, ob,
                           self.lock)

    def exponential(self, scale=1.0, size=None):
        """
        exponential(scale=1.0, size=None)

        Draw samples from an exponential distribution.

        Its probability density function is

        .. math:: f(x; \\frac{1}{\\beta}) = \\frac{1}{\\beta} \\exp(-\\frac{x}{\\beta}),

        for ``x > 0`` and 0 elsewhere. :math:`\\beta` is the scale parameter,
        which is the inverse of the rate parameter :math:`\\lambda = 1/\\beta`.
        The rate parameter is an alternative, widely used parameterization
        of the exponential distribution [3]_.

        The exponential distribution is a continuous analogue of the
        geometric distribution.  It describes many common situations, such as
        the size of raindrops measured over many rainstorms [1]_, or the time
        between page requests to Wikipedia [2]_.

        Parameters
        ----------
        scale : float
            The scale parameter, :math:`\\beta = 1/\\lambda`.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        References
        ----------
        .. [1] Peyton Z. Peebles Jr., "Probability, Random Variables and
               Random Signal Principles", 4th ed, 2001, p. 57.
        .. [2] "Poisson Process", Wikipedia,
               http://en.wikipedia.org/wiki/Poisson_process
        .. [3] "Exponential Distribution, Wikipedia,
               http://en.wikipedia.org/wiki/Exponential_distribution

        """
        cdef cnp.ndarray oscale
        cdef double fscale

        fscale = PyFloat_AsDouble(scale)
        if not PyErr_Occurred():
            if fscale <= 0:
                raise ValueError("scale <= 0")
            return vec_cont1_array_sc(self.internal_state, irk_exponential_vec, size,
                                  fscale, self.lock)

        PyErr_Clear()

        oscale = <cnp.ndarray> cnp.PyArray_FROM_OTF(scale, cnp.NPY_DOUBLE,
                                            cnp.NPY_ARRAY_ALIGNED)
        if np.any(np.less_equal(oscale, 0.0)):
            raise ValueError("scale <= 0")
        return vec_cont1_array(self.internal_state, irk_exponential_vec, size, oscale,
                           self.lock)

    def standard_exponential(self, size=None):
        """
        standard_exponential(size=None)

        Draw samples from the standard exponential distribution.

        `standard_exponential` is identical to the exponential distribution
        with a scale parameter of 1.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : float or ndarray
            Drawn samples.

        Examples
        --------
        Output a 3x8000 array:

        >>> n = mkl_random.standard_exponential((3, 8000))

        """
        return vec_cont0_array(self.internal_state, irk_standard_exponential_vec, size,
                           self.lock)

    def standard_gamma(self, shape, size=None):
        """
        standard_gamma(shape, size=None)

        Draw samples from a standard Gamma distribution.

        Samples are drawn from a Gamma distribution with specified parameters,
        shape (sometimes designated "k") and scale=1.

        Parameters
        ----------
        shape : float
            Parameter, should be > 0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
            The drawn samples.

        See Also
        --------
        scipy.stats.distributions.gamma : probability density function,
            distribution or cumulative density function, etc.

        Notes
        -----
        The probability density for the Gamma distribution is

        .. math:: p(x) = x^{k-1}\\frac{e^{-x/\\theta}}{\\theta^k\\Gamma(k)},

        where :math:`k` is the shape and :math:`\\theta` the scale,
        and :math:`\\Gamma` is the Gamma function.

        The Gamma distribution is often used to model the times to failure of
        electronic components, and arises naturally in processes for which the
        waiting times between Poisson distributed events are relevant.

        References
        ----------
        .. [1] Weisstein, Eric W. "Gamma Distribution." From MathWorld--A
               Wolfram Web Resource.
               http://mathworld.wolfram.com/GammaDistribution.html
        .. [2] Wikipedia, "Gamma-distribution",
               http://en.wikipedia.org/wiki/Gamma-distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> shape, scale = 2., 1. # mean and width
        >>> s = mkl_random.standard_gamma(shape, 1000000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> import scipy.special as sps
        >>> count, bins, ignored = plt.hist(s, 50, normed=True)
        >>> y = bins**(shape-1) * ((np.exp(-bins/scale))/ \\
        ...                       (sps.gamma(shape) * scale**shape))
        >>> plt.plot(bins, y, linewidth=2, color='r')
        >>> plt.show()

        """
        cdef cnp.ndarray oshape
        cdef double fshape

        fshape = PyFloat_AsDouble(shape)
        if not PyErr_Occurred():
            if fshape <= 0:
                raise ValueError("shape <= 0")
            return vec_cont1_array_sc(self.internal_state, irk_standard_gamma_vec,
                                  size, fshape, self.lock)

        PyErr_Clear()
        oshape = <cnp.ndarray> cnp.PyArray_FROM_OTF(shape, cnp.NPY_DOUBLE,
                                            cnp.NPY_ARRAY_ALIGNED)
        if np.any(np.less_equal(oshape, 0.0)):
            raise ValueError("shape <= 0")
        return vec_cont1_array(self.internal_state, irk_standard_gamma_vec, size,
                           oshape, self.lock)

    def gamma(self, shape, scale=1.0, size=None):
        """
        gamma(shape, scale=1.0, size=None)

        Draw samples from a Gamma distribution.

        Samples are drawn from a Gamma distribution with specified parameters,
        `shape` (sometimes designated "k") and `scale` (sometimes designated
        "theta"), where both parameters are > 0.

        Parameters
        ----------
        shape : scalar > 0
            The shape of the gamma distribution.
        scale : scalar > 0, optional
            The scale of the gamma distribution.  Default is equal to 1.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : ndarray, float
            Returns one sample unless `size` parameter is specified.

        See Also
        --------
        scipy.stats.distributions.gamma : probability density function,
            distribution or cumulative density function, etc.

        Notes
        -----
        The probability density for the Gamma distribution is

        .. math:: p(x) = x^{k-1}\\frac{e^{-x/\\theta}}{\\theta^k\\Gamma(k)},

        where :math:`k` is the shape and :math:`\\theta` the scale,
        and :math:`\\Gamma` is the Gamma function.

        The Gamma distribution is often used to model the times to failure of
        electronic components, and arises naturally in processes for which the
        waiting times between Poisson distributed events are relevant.

        References
        ----------
        .. [1] Weisstein, Eric W. "Gamma Distribution." From MathWorld--A
               Wolfram Web Resource.
               http://mathworld.wolfram.com/GammaDistribution.html
        .. [2] Wikipedia, "Gamma-distribution",
               http://en.wikipedia.org/wiki/Gamma-distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> shape, scale = 2., 2. # mean and dispersion
        >>> s = mkl_random.gamma(shape, scale, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> import scipy.special as sps
        >>> count, bins, ignored = plt.hist(s, 50, normed=True)
        >>> y = bins**(shape-1)*(np.exp(-bins/scale) /
        ...                      (sps.gamma(shape)*scale**shape))
        >>> plt.plot(bins, y, linewidth=2, color='r')
        >>> plt.show()

        """
        cdef cnp.ndarray oshape, oscale
        cdef double fshape, fscale

        fshape = PyFloat_AsDouble(shape)
        fscale = PyFloat_AsDouble(scale)
        if not PyErr_Occurred():
            if fshape <= 0:
                raise ValueError("shape <= 0")
            if fscale <= 0:
                raise ValueError("scale <= 0")
            return vec_cont2_array_sc(self.internal_state, irk_gamma_vec, size, fshape,
                                  fscale, self.lock)

        PyErr_Clear()
        oshape = <cnp.ndarray>cnp.PyArray_FROM_OTF(shape, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        oscale = <cnp.ndarray>cnp.PyArray_FROM_OTF(scale, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        if np.any(np.less_equal(oshape, 0.0)):
            raise ValueError("shape <= 0")
        if np.any(np.less_equal(oscale, 0.0)):
            raise ValueError("scale <= 0")
        return vec_cont2_array(self.internal_state, irk_gamma_vec, size, oshape, oscale,
                           self.lock)

    def f(self, dfnum, dfden, size=None):
        """
        f(dfnum, dfden, size=None)

        Draw samples from an F distribution.

        Samples are drawn from an F distribution with specified parameters,
        `dfnum` (degrees of freedom in numerator) and `dfden` (degrees of
        freedom in denominator), where both parameters should be greater than
        zero.

        The random variate of the F distribution (also known as the
        Fisher distribution) is a continuous probability distribution
        that arises in ANOVA tests, and is the ratio of two chi-square
        variates.

        Parameters
        ----------
        dfnum : float
            Degrees of freedom in numerator. Should be greater than zero.
        dfden : float
            Degrees of freedom in denominator. Should be greater than zero.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
            Samples from the Fisher distribution.

        See Also
        --------
        scipy.stats.distributions.f : probability density function,
            distribution or cumulative density function, etc.

        Notes
        -----
        The F statistic is used to compare in-group variances to between-group
        variances. Calculating the distribution depends on the sampling, and
        so it is a function of the respective degrees of freedom in the
        problem.  The variable `dfnum` is the number of samples minus one, the
        between-groups degrees of freedom, while `dfden` is the within-groups
        degrees of freedom, the sum of the number of samples in each group
        minus the number of groups.

        References
        ----------
        .. [1] Glantz, Stanton A. "Primer of Biostatistics.", McGraw-Hill,
               Fifth Edition, 2002.
        .. [2] Wikipedia, "F-distribution",
               http://en.wikipedia.org/wiki/F-distribution

        Examples
        --------
        An example from Glantz[1], pp 47-40:

        Two groups, children of diabetics (25 people) and children from people
        without diabetes (25 controls). Fasting blood glucose was measured,
        case group had a mean value of 86.1, controls had a mean value of
        82.2. Standard deviations were 2.09 and 2.49 respectively. Are these
        data consistent with the null hypothesis that the parents diabetic
        status does not affect their children's blood glucose levels?
        Calculating the F statistic from the data gives a value of 36.01.

        Draw samples from the distribution:

        >>> dfnum = 1. # between group degrees of freedom
        >>> dfden = 48. # within groups degrees of freedom
        >>> s = mkl_random.f(dfnum, dfden, 1000)

        The lower bound for the top 1% of the samples is :

        >>> sort(s)[-10]
        7.61988120985

        So there is about a 1% chance that the F statistic will exceed 7.62,
        the measured value is 36, so the null hypothesis is rejected at the 1%
        level.

        """
        cdef cnp.ndarray odfnum, odfden
        cdef double fdfnum, fdfden

        fdfnum = PyFloat_AsDouble(dfnum)
        fdfden = PyFloat_AsDouble(dfden)
        if not PyErr_Occurred():
            if fdfnum <= 0:
                raise ValueError("shape <= 0")
            if fdfden <= 0:
                raise ValueError("scale <= 0")
            return vec_cont2_array_sc(self.internal_state, irk_f_vec, size, fdfnum,
                                  fdfden, self.lock)

        PyErr_Clear()

        odfnum = <cnp.ndarray>cnp.PyArray_FROM_OTF(dfnum, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        odfden = <cnp.ndarray>cnp.PyArray_FROM_OTF(dfden, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        if np.any(np.less_equal(odfnum, 0.0)):
            raise ValueError("dfnum <= 0")
        if np.any(np.less_equal(odfden, 0.0)):
            raise ValueError("dfden <= 0")
        return vec_cont2_array(self.internal_state, irk_f_vec, size, odfnum, odfden,
                           self.lock)

    def noncentral_f(self, dfnum, dfden, nonc, size=None):
        """
        noncentral_f(dfnum, dfden, nonc, size=None)

        Draw samples from the noncentral F distribution.

        Samples are drawn from an F distribution with specified parameters,
        `dfnum` (degrees of freedom in numerator) and `dfden` (degrees of
        freedom in denominator), where both parameters > 1.
        `nonc` is the non-centrality parameter.

        Parameters
        ----------
        dfnum : int
            Parameter, should be > 1.
        dfden : int
            Parameter, should be > 1.
        nonc : float
            Parameter, should be >= 0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : scalar or ndarray
            Drawn samples.

        Notes
        -----
        When calculating the power of an experiment (power = probability of
        rejecting the null hypothesis when a specific alternative is true) the
        non-central F statistic becomes important.  When the null hypothesis is
        true, the F statistic follows a central F distribution. When the null
        hypothesis is not true, then it follows a non-central F statistic.

        References
        ----------
        .. [1] Weisstein, Eric W. "Noncentral F-Distribution."
               From MathWorld--A Wolfram Web Resource.
               http://mathworld.wolfram.com/NoncentralF-Distribution.html
        .. [2] Wikipedia, "Noncentral F distribution",
               http://en.wikipedia.org/wiki/Noncentral_F-distribution

        Examples
        --------
        In a study, testing for a specific alternative to the null hypothesis
        requires use of the Noncentral F distribution. We need to calculate the
        area in the tail of the distribution that exceeds the value of the F
        distribution for the null hypothesis.  We'll plot the two probability
        distributions for comparison.

        >>> dfnum = 3 # between group deg of freedom
        >>> dfden = 20 # within groups degrees of freedom
        >>> nonc = 3.0
        >>> nc_vals = mkl_random.noncentral_f(dfnum, dfden, nonc, 1000000)
        >>> NF = np.histogram(nc_vals, bins=50, normed=True)
        >>> c_vals = mkl_random.f(dfnum, dfden, 1000000)
        >>> F = np.histogram(c_vals, bins=50, normed=True)
        >>> plt.plot(F[1][1:], F[0])
        >>> plt.plot(NF[1][1:], NF[0])
        >>> plt.show()

        """
        cdef cnp.ndarray odfnum, odfden, ononc
        cdef double fdfnum, fdfden, fnonc

        fdfnum = PyFloat_AsDouble(dfnum)
        fdfden = PyFloat_AsDouble(dfden)
        fnonc = PyFloat_AsDouble(nonc)
        if not PyErr_Occurred():
            if fdfnum <= 1:
                raise ValueError("dfnum <= 1")
            if fdfden <= 0:
                raise ValueError("dfden <= 0")
            if fnonc < 0:
                raise ValueError("nonc < 0")
            return vec_cont3_array_sc(self.internal_state, irk_noncentral_f_vec, size,
                                  fdfnum, fdfden, fnonc, self.lock)

        PyErr_Clear()

        odfnum = <cnp.ndarray>cnp.PyArray_FROM_OTF(dfnum, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        odfden = <cnp.ndarray>cnp.PyArray_FROM_OTF(dfden, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        ononc = <cnp.ndarray>cnp.PyArray_FROM_OTF(nonc, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)

        if np.any(np.less_equal(odfnum, 1.0)):
            raise ValueError("dfnum <= 1")
        if np.any(np.less_equal(odfden, 0.0)):
            raise ValueError("dfden <= 0")
        if np.any(np.less(ononc, 0.0)):
            raise ValueError("nonc < 0")
        return vec_cont3_array(self.internal_state, irk_noncentral_f_vec, size, odfnum,
                           odfden, ononc, self.lock)

    def chisquare(self, df, size=None):
        """
        chisquare(df, size=None)

        Draw samples from a chi-square distribution.

        When `df` independent random variables, each with standard normal
        distributions (mean 0, variance 1), are squared and summed, the
        resulting distribution is chi-square (see Notes).  This distribution
        is often used in hypothesis testing.

        Parameters
        ----------
        df : int
             Number of degrees of freedom.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        output : ndarray
            Samples drawn from the distribution, packed in a `size`-shaped
            array.

        Raises
        ------
        ValueError
            When `df` <= 0 or when an inappropriate `size` (e.g. ``size=-1``)
            is given.

        Notes
        -----
        The variable obtained by summing the squares of `df` independent,
        standard normally distributed random variables:

        .. math:: Q = \\sum_{i=0}^{\\mathtt{df}} X^2_i

        is chi-square distributed, denoted

        .. math:: Q \\sim \\chi^2_k.

        The probability density function of the chi-squared distribution is

        .. math:: p(x) = \\frac{(1/2)^{k/2}}{\\Gamma(k/2)}
                         x^{k/2 - 1} e^{-x/2},

        where :math:`\\Gamma` is the gamma function,

        .. math:: \\Gamma(x) = \\int_0^{-\\infty} t^{x - 1} e^{-t} dt.

        References
        ----------
        .. [1] NIST "Engineering Statistics Handbook"
               http://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm

        Examples
        --------
        >>> mkl_random.chisquare(2,4)
        array([ 1.89920014,  9.00867716,  3.13710533,  5.62318272])

        """
        cdef cnp.ndarray odf
        cdef double fdf

        fdf = PyFloat_AsDouble(df)
        if not PyErr_Occurred():
            if fdf <= 0:
                raise ValueError("df <= 0")
            return vec_cont1_array_sc(self.internal_state, irk_chisquare_vec, size, fdf,
                                  self.lock)

        PyErr_Clear()

        odf = <cnp.ndarray>cnp.PyArray_FROM_OTF(df, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        if np.any(np.less_equal(odf, 0.0)):
            raise ValueError("df <= 0")
        return vec_cont1_array(self.internal_state, irk_chisquare_vec, size, odf,
                           self.lock)

    def noncentral_chisquare(self, df, nonc, size=None):
        """
        noncentral_chisquare(df, nonc, size=None)

        Draw samples from a noncentral chi-square distribution.

        The noncentral :math:`\\chi^2` distribution is a generalisation of
        the :math:`\\chi^2` distribution.

        Parameters
        ----------
        df : int
            Degrees of freedom, should be > 0 as of Numpy 1.10,
            should be > 1 for earlier versions.
        nonc : float
            Non-centrality, should be non-negative.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Notes
        -----
        The probability density function for the noncentral Chi-square
        distribution is

        .. math:: P(x;df,nonc) = \\sum^{\\infty}_{i=0}
                               \\frac{e^{-nonc/2}(nonc/2)^{i}}{i!}
                               \\P_{Y_{df+2i}}(x),

        where :math:`Y_{q}` is the Chi-square with q degrees of freedom.

        In Delhi (2007), it is noted that the noncentral chi-square is
        useful in bombing and coverage problems, the probability of
        killing the point target given by the noncentral chi-squared
        distribution.

        References
        ----------
        .. [1] Delhi, M.S. Holla, "On a noncentral chi-square distribution in
               the analysis of weapon systems effectiveness", Metrika,
               Volume 15, Number 1 / December, 1970.
        .. [2] Wikipedia, "Noncentral chi-square distribution"
               http://en.wikipedia.org/wiki/Noncentral_chi-square_distribution

        Examples
        --------
        Draw values from the distribution and plot the histogram

        >>> import matplotlib.pyplot as plt
        >>> values = plt.hist(mkl_random.noncentral_chisquare(3, 20, 100000),
        ...                   bins=200, normed=True)
        >>> plt.show()

        Draw values from a noncentral chisquare with very small noncentrality,
        and compare to a chisquare.

        >>> plt.figure()
        >>> values = plt.hist(mkl_random.noncentral_chisquare(3, .0000001, 100000),
        ...                   bins=np.arange(0., 25, .1), normed=True)
        >>> values2 = plt.hist(mkl_random.chisquare(3, 100000),
        ...                    bins=np.arange(0., 25, .1), normed=True)
        >>> plt.plot(values[1][0:-1], values[0]-values2[0], 'ob')
        >>> plt.show()

        Demonstrate how large values of non-centrality lead to a more symmetric
        distribution.

        >>> plt.figure()
        >>> values = plt.hist(mkl_random.noncentral_chisquare(3, 20, 100000),
        ...                   bins=200, normed=True)
        >>> plt.show()

        """
        cdef cnp.ndarray odf, ononc
        cdef double fdf, fnonc

        fdf = PyFloat_AsDouble(df)
        fnonc = PyFloat_AsDouble(nonc)
        if not PyErr_Occurred():
            if fdf <= 0:
                raise ValueError("df <= 0")
            if fnonc < 0:
                raise ValueError("nonc < 0")
            return vec_cont2_array_sc(self.internal_state, irk_noncentral_chisquare_vec,
                                  size, fdf, fnonc, self.lock)

        PyErr_Clear()

        odf = <cnp.ndarray>cnp.PyArray_FROM_OTF(df, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        ononc = <cnp.ndarray>cnp.PyArray_FROM_OTF(nonc, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        if np.any(np.less_equal(odf, 0.0)):
            raise ValueError("df <= 0")
        if np.any(np.less(ononc, 0.0)):
            raise ValueError("nonc < 0")
        return vec_cont2_array(self.internal_state, irk_noncentral_chisquare_vec, size,
                           odf, ononc, self.lock)

    def standard_cauchy(self, size=None):
        """
        standard_cauchy(size=None)

        Draw samples from a standard Cauchy distribution with mode = 0.

        Also known as the Lorentz distribution.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
            The drawn samples.

        Notes
        -----
        The probability density function for the full Cauchy distribution is

        .. math:: P(x; x_0, \\gamma) = \\frac{1}{\\pi \\gamma \\bigl[ 1+
                  (\\frac{x-x_0}{\\gamma})^2 \\bigr] }

        and the Standard Cauchy distribution just sets :math:`x_0=0` and
        :math:`\\gamma=1`

        The Cauchy distribution arises in the solution to the driven harmonic
        oscillator problem, and also describes spectral line broadening. It
        also describes the distribution of values at which a line tilted at
        a random angle will cut the x axis.

        When studying hypothesis tests that assume normality, seeing how the
        tests perform on data from a Cauchy distribution is a good indicator of
        their sensitivity to a heavy-tailed distribution, since the Cauchy looks
        very much like a Gaussian distribution, but with heavier tails.

        References
        ----------
        .. [1] NIST/SEMATECH e-Handbook of Statistical Methods, "Cauchy
              Distribution",
              http://www.itl.nist.gov/div898/handbook/eda/section3/eda3663.htm
        .. [2] Weisstein, Eric W. "Cauchy Distribution." From MathWorld--A
              Wolfram Web Resource.
              http://mathworld.wolfram.com/CauchyDistribution.html
        .. [3] Wikipedia, "Cauchy distribution"
              http://en.wikipedia.org/wiki/Cauchy_distribution

        Examples
        --------
        Draw samples and plot the distribution:

        >>> s = mkl_random.standard_cauchy(1000000)
        >>> s = s[(s>-25) & (s<25)]  # truncate distribution so it plots well
        >>> plt.hist(s, bins=100)
        >>> plt.show()

        """
        return vec_cont0_array(self.internal_state, irk_standard_cauchy_vec, size,
                           self.lock)

    def standard_t(self, df, size=None):
        """
        standard_t(df, size=None)

        Draw samples from a standard Student's t distribution with `df` degrees
        of freedom.

        A special case of the hyperbolic distribution.  As `df` gets
        large, the result resembles that of the standard normal
        distribution (`standard_normal`).

        Parameters
        ----------
        df : int
            Degrees of freedom, should be > 0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
            Drawn samples.

        Notes
        -----
        The probability density function for the t distribution is

        .. math:: P(x, df) = \\frac{\\Gamma(\\frac{df+1}{2})}{\\sqrt{\\pi df}
                  \\Gamma(\\frac{df}{2})}\\Bigl( 1+\\frac{x^2}{df} \\Bigr)^{-(df+1)/2}

        The t test is based on an assumption that the data come from a
        Normal distribution. The t test provides a way to test whether
        the sample mean (that is the mean calculated from the data) is
        a good estimate of the true mean.

        The derivation of the t-distribution was first published in
        1908 by William Gisset while working for the Guinness Brewery
        in Dublin. Due to proprietary issues, he had to publish under
        a pseudonym, and so he used the name Student.

        References
        ----------
        .. [1] Dalgaard, Peter, "Introductory Statistics With R",
               Springer, 2002.
        .. [2] Wikipedia, "Student's t-distribution"
               http://en.wikipedia.org/wiki/Student's_t-distribution

        Examples
        --------
        From Dalgaard page 83 [1]_, suppose the daily energy intake for 11
        women in Kj is:

        >>> intake = np.array([5260., 5470, 5640, 6180, 6390, 6515, 6805, 7515, \\
        ...                    7515, 8230, 8770])

        Does their energy intake deviate systematically from the recommended
        value of 7725 kJ?

        We have 10 degrees of freedom, so is the sample mean within 95% of the
        recommended value?

        >>> s = mkl_random.standard_t(10, size=100000)
        >>> np.mean(intake)
        6753.636363636364
        >>> intake.std(ddof=1)
        1142.1232221373727

        Calculate the t statistic, setting the ddof parameter to the unbiased
        value so the divisor in the standard deviation will be degrees of
        freedom, N-1.

        >>> t = (np.mean(intake)-7725)/(intake.std(ddof=1)/np.sqrt(len(intake)))
        >>> import matplotlib.pyplot as plt
        >>> h = plt.hist(s, bins=100, normed=True)

        For a one-sided t-test, how far out in the distribution does the t
        statistic appear?

        >>> np.sum(s<t) / float(len(s))
        0.0090699999999999999  #random

        So the p-value is about 0.009, which says the null hypothesis has a
        probability of about 99% of being true.

        """
        cdef cnp.ndarray odf
        cdef double fdf

        fdf = PyFloat_AsDouble(df)
        if not PyErr_Occurred():
            if fdf <= 0:
                raise ValueError("df <= 0")
            return vec_cont1_array_sc(self.internal_state, irk_standard_t_vec, size,
                                  fdf, self.lock)

        PyErr_Clear()

        odf = <cnp.ndarray> cnp.PyArray_FROM_OTF(df, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        if np.any(np.less_equal(odf, 0.0)):
            raise ValueError("df <= 0")
        return vec_cont1_array(self.internal_state, irk_standard_t_vec, size, odf,
                           self.lock)

    def vonmises(self, mu, kappa, size=None):
        """
        vonmises(mu, kappa, size=None)

        Draw samples from a von Mises distribution.

        Samples are drawn from a von Mises distribution with specified mode
        (mu) and dispersion (kappa), on the interval [-pi, pi].

        The von Mises distribution (also known as the circular normal
        distribution) is a continuous probability distribution on the unit
        circle.  It may be thought of as the circular analogue of the normal
        distribution.

        Parameters
        ----------
        mu : float
            Mode ("center") of the distribution.
        kappa : float
            Dispersion of the distribution, has to be >=0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : scalar or ndarray
            The returned samples, which are in the interval [-pi, pi].

        See Also
        --------
        scipy.stats.distributions.vonmises : probability density function,
            distribution, or cumulative density function, etc.

        Notes
        -----
        The probability density for the von Mises distribution is

        .. math:: p(x) = \\frac{e^{\\kappa cos(x-\\mu)}}{2\\pi I_0(\\kappa)},

        where :math:`\\mu` is the mode and :math:`\\kappa` the dispersion,
        and :math:`I_0(\\kappa)` is the modified Bessel function of order 0.

        The von Mises is named for Richard Edler von Mises, who was born in
        Austria-Hungary, in what is now the Ukraine.  He fled to the United
        States in 1939 and became a professor at Harvard.  He worked in
        probability theory, aerodynamics, fluid mechanics, and philosophy of
        science.

        References
        ----------
        .. [1] Abramowitz, M. and Stegun, I. A. (Eds.). "Handbook of
               Mathematical Functions with Formulas, Graphs, and Mathematical
               Tables, 9th printing," New York: Dover, 1972.
        .. [2] von Mises, R., "Mathematical Theory of Probability
               and Statistics", New York: Academic Press, 1964.

        Examples
        --------
        Draw samples from the distribution:

        >>> mu, kappa = 0.0, 4.0 # mean and dispersion
        >>> s = mkl_random.vonmises(mu, kappa, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> from scipy.special import i0
        >>> plt.hist(s, 50, normed=True)
        >>> x = np.linspace(-np.pi, np.pi, num=51)
        >>> y = np.exp(kappa*np.cos(x-mu))/(2*np.pi*i0(kappa))
        >>> plt.plot(x, y, linewidth=2, color='r')
        >>> plt.show()

        """
        cdef cnp.ndarray omu, okappa
        cdef double fmu, fkappa

        fmu = PyFloat_AsDouble(mu)
        fkappa = PyFloat_AsDouble(kappa)
        if not PyErr_Occurred():
            if fkappa < 0:
                raise ValueError("kappa < 0")
            return vec_cont2_array_sc(self.internal_state, irk_vonmises_vec, size, fmu,
                                  fkappa, self.lock)

        PyErr_Clear()

        omu = <cnp.ndarray> cnp.PyArray_FROM_OTF(mu, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        okappa = <cnp.ndarray> cnp.PyArray_FROM_OTF(kappa, cnp.NPY_DOUBLE,
                                            cnp.NPY_ARRAY_ALIGNED)
        if np.any(np.less(okappa, 0.0)):
            raise ValueError("kappa < 0")
        return vec_cont2_array(self.internal_state, irk_vonmises_vec, size, omu, okappa,
                           self.lock)

    def pareto(self, a, size=None):
        """
        pareto(a, size=None)

        Draw samples from a Pareto II or Lomax distribution with
        specified shape.

        The Lomax or Pareto II distribution is a shifted Pareto
        distribution. The classical Pareto distribution can be
        obtained from the Lomax distribution by adding 1 and
        multiplying by the scale parameter ``m`` (see Notes).  The
        smallest value of the Lomax distribution is zero while for the
        classical Pareto distribution it is ``mu``, where the standard
        Pareto distribution has location ``mu = 1``.  Lomax can also
        be considered as a simplified version of the Generalized
        Pareto distribution (available in SciPy), with the scale set
        to one and the location set to zero.

        The Pareto distribution must be greater than zero, and is
        unbounded above.  It is also known as the "80-20 rule".  In
        this distribution, 80 percent of the weights are in the lowest
        20 percent of the range, while the other 20 percent fill the
        remaining 80 percent of the range.

        Parameters
        ----------
        shape : float, > 0.
            Shape of the distribution.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        See Also
        --------
        scipy.stats.distributions.lomax.pdf : probability density function,
            distribution or cumulative density function, etc.
        scipy.stats.distributions.genpareto.pdf : probability density function,
            distribution or cumulative density function, etc.

        Notes
        -----
        The probability density for the Pareto distribution is

        .. math:: p(x) = \\frac{a m^a}{(1+x)^{a+1}}

        where :math:`a` is the shape and :math:`m` the scale.

        The Pareto distribution, named after the Italian economist
        Vilfredo Pareto, is a power law probability distribution
        useful in many real world problems.  Outside the field of
        economics it is generally referred to as the Bradford
        distribution. Pareto developed the distribution to describe
        the distribution of wealth in an economy.  It has also found
        use in insurance, web page access statistics, oil field sizes,
        and many other problems, including the download frequency for
        projects in Sourceforge [1]_.  It is one of the so-called
        "fat-tailed" distributions.


        References
        ----------
        .. [1] Francis Hunt and Paul Johnson, On the Pareto Distribution of
               Sourceforge projects.
        .. [2] Pareto, V. (1896). Course of Political Economy. Lausanne.
        .. [3] Reiss, R.D., Thomas, M.(2001), Statistical Analysis of Extreme
               Values, Birkhauser Verlag, Basel, pp 23-30.
        .. [4] Wikipedia, "Pareto distribution",
               http://en.wikipedia.org/wiki/Pareto_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> a, m = 3., 2.  # shape and mode
        >>> s = (mkl_random.pareto(a, 1000) + 1) * m

        Display the histogram of the samples, along with the probability
        density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, _ = plt.hist(s, 100, normed=True)
        >>> fit = a*m**a / bins**(a+1)
        >>> plt.plot(bins, max(count)*fit/max(fit), linewidth=2, color='r')
        >>> plt.show()

        """
        cdef cnp.ndarray oa
        cdef double fa

        fa = PyFloat_AsDouble(a)
        if not PyErr_Occurred():
            if fa <= 0:
                raise ValueError("a <= 0")
            return vec_cont1_array_sc(self.internal_state, irk_pareto_vec, size, fa,
                                  self.lock)

        PyErr_Clear()

        oa = <cnp.ndarray>cnp.PyArray_FROM_OTF(a, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        if np.any(np.less_equal(oa, 0.0)):
            raise ValueError("a <= 0")
        return vec_cont1_array(self.internal_state, irk_pareto_vec, size, oa, self.lock)

    def weibull(self, a, size=None):
        """
        weibull(a, size=None)

        Draw samples from a Weibull distribution.

        Draw samples from a 1-parameter Weibull distribution with the given
        shape parameter `a`.

        .. math:: X = (-ln(U))^{1/a}

        Here, U is drawn from the uniform distribution over (0,1].

        The more common 2-parameter Weibull, including a scale parameter
        :math:`\\lambda` is just :math:`X = \\lambda(-ln(U))^{1/a}`.

        Parameters
        ----------
        a : float
            Shape of the distribution.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray

        See Also
        --------
        scipy.stats.distributions.weibull_max
        scipy.stats.distributions.weibull_min
        scipy.stats.distributions.genextreme
        gumbel

        Notes
        -----
        The Weibull (or Type III asymptotic extreme value distribution
        for smallest values, SEV Type III, or Rosin-Rammler
        distribution) is one of a class of Generalized Extreme Value
        (GEV) distributions used in modeling extreme value problems.
        This class includes the Gumbel and Frechet distributions.

        The probability density for the Weibull distribution is

        .. math:: p(x) = \\frac{a}
                         {\\lambda}(\\frac{x}{\\lambda})^{a-1}e^{-(x/\\lambda)^a},

        where :math:`a` is the shape and :math:`\\lambda` the scale.

        The function has its peak (the mode) at
        :math:`\\lambda(\\frac{a-1}{a})^{1/a}`.

        When ``a = 1``, the Weibull distribution reduces to the exponential
        distribution.

        References
        ----------
        .. [1] Waloddi Weibull, Royal Technical University, Stockholm,
               1939 "A Statistical Theory Of The Strength Of Materials",
               Ingeniorsvetenskapsakademiens Handlingar Nr 151, 1939,
               Generalstabens Litografiska Anstalts Forlag, Stockholm.
        .. [2] Waloddi Weibull, "A Statistical Distribution Function of
               Wide Applicability", Journal Of Applied Mechanics ASME Paper
               1951.
        .. [3] Wikipedia, "Weibull distribution",
               http://en.wikipedia.org/wiki/Weibull_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> a = 5. # shape
        >>> s = mkl_random.weibull(a, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> x = np.arange(1,100.)/50.
        >>> def weib(x,n,a):
        ...     return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)

        >>> count, bins, ignored = plt.hist(mkl_random.weibull(5.,1000))
        >>> x = np.arange(1,100.)/50.
        >>> scale = count.max()/weib(x, 1., 5.).max()
        >>> plt.plot(x, weib(x, 1., 5.)*scale)
        >>> plt.show()

        """
        cdef cnp.ndarray oa
        cdef double fa

        fa = PyFloat_AsDouble(a)
        if not PyErr_Occurred():
            if fa <= 0:
                raise ValueError("a <= 0")
            return vec_cont1_array_sc(self.internal_state, irk_weibull_vec, size, fa,
                                  self.lock)

        PyErr_Clear()

        oa = <cnp.ndarray>cnp.PyArray_FROM_OTF(a, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        if np.any(np.less_equal(oa, 0.0)):
            raise ValueError("a <= 0")
        return vec_cont1_array(self.internal_state, irk_weibull_vec, size, oa,
                           self.lock)

    def power(self, a, size=None):
        """
        power(a, size=None)

        Draws samples in [0, 1] from a power distribution with positive
        exponent a - 1.

        Also known as the power function distribution.

        Parameters
        ----------
        a : float
            parameter, > 0
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
            The returned samples lie in [0, 1].

        Raises
        ------
        ValueError
            If a < 1.

        Notes
        -----
        The probability density function is

        .. math:: P(x; a) = ax^{a-1}, 0 \\le x \\le 1, a>0.

        The power function distribution is just the inverse of the Pareto
        distribution. It may also be seen as a special case of the Beta
        distribution.

        It is used, for example, in modeling the over-reporting of insurance
        claims.

        References
        ----------
        .. [1] Christian Kleiber, Samuel Kotz, "Statistical size distributions
               in economics and actuarial sciences", Wiley, 2003.
        .. [2] Heckert, N. A. and Filliben, James J. "NIST Handbook 148:
               Dataplot Reference Manual, Volume 2: Let Subcommands and Library
               Functions", National Institute of Standards and Technology
               Handbook Series, June 2003.
               http://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/powpdf.pdf

        Examples
        --------
        Draw samples from the distribution:

        >>> a = 5. # shape
        >>> samples = 1000
        >>> s = mkl_random.power(a, samples)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, bins=30)
        >>> x = np.linspace(0, 1, 100)
        >>> y = a*x**(a-1.)
        >>> normed_y = samples*np.diff(bins)[0]*y
        >>> plt.plot(x, normed_y)
        >>> plt.show()

        Compare the power function distribution to the inverse of the Pareto.

        >>> from scipy import stats
        >>> rvs = mkl_random.power(5, 1000000)
        >>> rvsp = mkl_random.pareto(5, 1000000)
        >>> xx = np.linspace(0,1,100)
        >>> powpdf = stats.powerlaw.pdf(xx,5)

        >>> plt.figure()
        >>> plt.hist(rvs, bins=50, normed=True)
        >>> plt.plot(xx,powpdf,'r-')
        >>> plt.title('mkl_random.power(5)')

        >>> plt.figure()
        >>> plt.hist(1./(1.+rvsp), bins=50, normed=True)
        >>> plt.plot(xx,powpdf,'r-')
        >>> plt.title('inverse of 1 + mkl_random.pareto(5)')

        >>> plt.figure()
        >>> plt.hist(1./(1.+rvsp), bins=50, normed=True)
        >>> plt.plot(xx,powpdf,'r-')
        >>> plt.title('inverse of stats.pareto(5)')

        """
        cdef cnp.ndarray oa
        cdef double fa

        fa = PyFloat_AsDouble(a)
        if not PyErr_Occurred():
            if fa <= 0:
                raise ValueError("a <= 0")
            return vec_cont1_array_sc(self.internal_state, irk_power_vec, size, fa,
                                  self.lock)

        PyErr_Clear()

        oa = <cnp.ndarray>cnp.PyArray_FROM_OTF(a, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        if np.any(np.less_equal(oa, 0.0)):
            raise ValueError("a <= 0")
        return vec_cont1_array(self.internal_state, irk_power_vec, size, oa, self.lock)

    def laplace(self, loc=0.0, scale=1.0, size=None):
        """
        laplace(loc=0.0, scale=1.0, size=None)

        Draw samples from the Laplace or double exponential distribution with
        specified location (or mean) and scale (decay).

        The Laplace distribution is similar to the Gaussian/normal distribution,
        but is sharper at the peak and has fatter tails. It represents the
        difference between two independent, identically distributed exponential
        random variables.

        Parameters
        ----------
        loc : float, optional
            The position, :math:`\\mu`, of the distribution peak.
        scale : float, optional
            :math:`\\lambda`, the exponential decay.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or float

        Notes
        -----
        It has the probability density function

        .. math:: f(x; \\mu, \\lambda) = \\frac{1}{2\\lambda}
                                       \\exp\\left(-\\frac{|x - \\mu|}{\\lambda}\\right).

        The first law of Laplace, from 1774, states that the frequency
        of an error can be expressed as an exponential function of the
        absolute magnitude of the error, which leads to the Laplace
        distribution. For many problems in economics and health
        sciences, this distribution seems to model the data better
        than the standard Gaussian distribution.

        References
        ----------
        .. [1] Abramowitz, M. and Stegun, I. A. (Eds.). "Handbook of
               Mathematical Functions with Formulas, Graphs, and Mathematical
               Tables, 9th printing," New York: Dover, 1972.
        .. [2] Kotz, Samuel, et. al. "The Laplace Distribution and
               Generalizations, " Birkhauser, 2001.
        .. [3] Weisstein, Eric W. "Laplace Distribution."
               From MathWorld--A Wolfram Web Resource.
               http://mathworld.wolfram.com/LaplaceDistribution.html
        .. [4] Wikipedia, "Laplace Distribution",
               http://en.wikipedia.org/wiki/Laplace_distribution

        Examples
        --------
        Draw samples from the distribution

        >>> loc, scale = 0., 1.
        >>> s = mkl_random.laplace(loc, scale, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 30, normed=True)
        >>> x = np.arange(-8., 8., .01)
        >>> pdf = np.exp(-abs(x-loc)/scale)/(2.*scale)
        >>> plt.plot(x, pdf)

        Plot Gaussian for comparison:

        >>> g = (1/(scale * np.sqrt(2 * np.pi)) *
        ...      np.exp(-(x - loc)**2 / (2 * scale**2)))
        >>> plt.plot(x,g)

        """
        cdef cnp.ndarray oloc, oscale
        cdef double floc, fscale

        floc = PyFloat_AsDouble(loc)
        fscale = PyFloat_AsDouble(scale)
        if not PyErr_Occurred():
            if fscale <= 0:
                raise ValueError("scale <= 0")
            return vec_cont2_array_sc(self.internal_state, irk_laplace_vec, size, floc,
                                  fscale, self.lock)

        PyErr_Clear()
        oloc = cnp.PyArray_FROM_OTF(loc, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        oscale = cnp.PyArray_FROM_OTF(scale, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        if np.any(np.less_equal(oscale, 0.0)):
            raise ValueError("scale <= 0")
        return vec_cont2_array(self.internal_state, irk_laplace_vec, size, oloc, oscale,
                           self.lock)

    def gumbel(self, loc=0.0, scale=1.0, size=None):
        """
        gumbel(loc=0.0, scale=1.0, size=None)

        Draw samples from a Gumbel distribution.

        Draw samples from a Gumbel distribution with specified location and
        scale.  For more information on the Gumbel distribution, see
        Notes and References below.

        Parameters
        ----------
        loc : float
            The location of the mode of the distribution.
        scale : float
            The scale parameter of the distribution.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar

        See Also
        --------
        scipy.stats.gumbel_l
        scipy.stats.gumbel_r
        scipy.stats.genextreme
        weibull

        Notes
        -----
        The Gumbel (or Smallest Extreme Value (SEV) or the Smallest Extreme
        Value Type I) distribution is one of a class of Generalized Extreme
        Value (GEV) distributions used in modeling extreme value problems.
        The Gumbel is a special case of the Extreme Value Type I distribution
        for maximums from distributions with "exponential-like" tails.

        The probability density for the Gumbel distribution is

        .. math:: p(x) = \\frac{e^{-(x - \\mu)/ \\beta}}{\\beta} e^{ -e^{-(x - \\mu)/
                  \\beta}},

        where :math:`\\mu` is the mode, a location parameter, and
        :math:`\\beta` is the scale parameter.

        The Gumbel (named for German mathematician Emil Julius Gumbel) was used
        very early in the hydrology literature, for modeling the occurrence of
        flood events. It is also used for modeling maximum wind speed and
        rainfall rates.  It is a "fat-tailed" distribution - the probability of
        an event in the tail of the distribution is larger than if one used a
        Gaussian, hence the surprisingly frequent occurrence of 100-year
        floods. Floods were initially modeled as a Gaussian process, which
        underestimated the frequency of extreme events.

        It is one of a class of extreme value distributions, the Generalized
        Extreme Value (GEV) distributions, which also includes the Weibull and
        Frechet.

        The function has a mean of :math:`\\mu + 0.57721\\beta` and a variance
        of :math:`\\frac{\\pi^2}{6}\\beta^2`.

        References
        ----------
        .. [1] Gumbel, E. J., "Statistics of Extremes,"
               New York: Columbia University Press, 1958.
        .. [2] Reiss, R.-D. and Thomas, M., "Statistical Analysis of Extreme
               Values from Insurance, Finance, Hydrology and Other Fields,"
               Basel: Birkhauser Verlag, 2001.

        Examples
        --------
        Draw samples from the distribution:

        >>> mu, beta = 0, 0.1 # location and scale
        >>> s = mkl_random.gumbel(mu, beta, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 30, normed=True)
        >>> plt.plot(bins, (1/beta)*np.exp(-(bins - mu)/beta)
        ...          * np.exp( -np.exp( -(bins - mu) /beta) ),
        ...          linewidth=2, color='r')
        >>> plt.show()

        Show how an extreme value distribution can arise from a Gaussian process
        and compare to a Gaussian:

        >>> means = []
        >>> maxima = []
        >>> for i in range(0,1000) :
        ...    a = mkl_random.normal(mu, beta, 1000)
        ...    means.append(a.mean())
        ...    maxima.append(a.max())
        >>> count, bins, ignored = plt.hist(maxima, 30, normed=True)
        >>> beta = np.std(maxima) * np.sqrt(6) / np.pi
        >>> mu = np.mean(maxima) - 0.57721*beta
        >>> plt.plot(bins, (1/beta)*np.exp(-(bins - mu)/beta)
        ...          * np.exp(-np.exp(-(bins - mu)/beta)),
        ...          linewidth=2, color='r')
        >>> plt.plot(bins, 1/(beta * np.sqrt(2 * np.pi))
        ...          * np.exp(-(bins - mu)**2 / (2 * beta**2)),
        ...          linewidth=2, color='g')
        >>> plt.show()

        """
        cdef cnp.ndarray oloc, oscale
        cdef double floc, fscale

        floc = PyFloat_AsDouble(loc)
        fscale = PyFloat_AsDouble(scale)
        if not PyErr_Occurred():
            if fscale <= 0:
                raise ValueError("scale <= 0")
            return vec_cont2_array_sc(self.internal_state, irk_gumbel_vec, size, floc,
                                  fscale, self.lock)

        PyErr_Clear()
        oloc = cnp.PyArray_FROM_OTF(loc, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        oscale = cnp.PyArray_FROM_OTF(scale, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        if np.any(np.less_equal(oscale, 0.0)):
            raise ValueError("scale <= 0")
        return vec_cont2_array(self.internal_state, irk_gumbel_vec, size, oloc, oscale,
                           self.lock)

    def logistic(self, loc=0.0, scale=1.0, size=None):
        """
        logistic(loc=0.0, scale=1.0, size=None)

        Draw samples from a logistic distribution.

        Samples are drawn from a logistic distribution with specified
        parameters, loc (location or mean, also median), and scale (>0).

        Parameters
        ----------
        loc : float

        scale : float > 0.

        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
                  where the values are all integers in  [0, n].

        See Also
        --------
        scipy.stats.distributions.logistic : probability density function,
            distribution or cumulative density function, etc.

        Notes
        -----
        The probability density for the Logistic distribution is

        .. math:: P(x) = P(x) = \\frac{e^{-(x-\\mu)/s}}{s(1+e^{-(x-\\mu)/s})^2},

        where :math:`\\mu` = location and :math:`s` = scale.

        The Logistic distribution is used in Extreme Value problems where it
        can act as a mixture of Gumbel distributions, in Epidemiology, and by
        the World Chess Federation (FIDE) where it is used in the Elo ranking
        system, assuming the performance of each player is a logistically
        distributed random variable.

        References
        ----------
        .. [1] Reiss, R.-D. and Thomas M. (2001), "Statistical Analysis of
               Extreme Values, from Insurance, Finance, Hydrology and Other
               Fields," Birkhauser Verlag, Basel, pp 132-133.
        .. [2] Weisstein, Eric W. "Logistic Distribution." From
               MathWorld--A Wolfram Web Resource.
               http://mathworld.wolfram.com/LogisticDistribution.html
        .. [3] Wikipedia, "Logistic-distribution",
               http://en.wikipedia.org/wiki/Logistic_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> loc, scale = 10, 1
        >>> s = mkl_random.logistic(loc, scale, 10000)
        >>> count, bins, ignored = plt.hist(s, bins=50)

        #   plot against distribution

        >>> def logist(x, loc, scale):
        ...     return exp((loc-x)/scale)/(scale*(1+exp((loc-x)/scale))**2)
        >>> plt.plot(bins, logist(bins, loc, scale)*count.max()/\\
        ... logist(bins, loc, scale).max())
        >>> plt.show()

        """
        cdef cnp.ndarray oloc, oscale
        cdef double floc, fscale

        floc = PyFloat_AsDouble(loc)
        fscale = PyFloat_AsDouble(scale)
        if not PyErr_Occurred():
            if fscale <= 0:
                raise ValueError("scale <= 0")
            return vec_cont2_array_sc(self.internal_state, irk_logistic_vec, size, floc,
                                  fscale, self.lock)

        PyErr_Clear()
        oloc = cnp.PyArray_FROM_OTF(loc, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        oscale = cnp.PyArray_FROM_OTF(scale, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        if np.any(np.less_equal(oscale, 0.0)):
            raise ValueError("scale <= 0")
        return vec_cont2_array(self.internal_state, irk_logistic_vec, size, oloc,
                           oscale, self.lock)

    def lognormal(self, mean=0.0, sigma=1.0, size=None, method=ICDF):
        """
        lognormal(mean=0.0, sigma=1.0, size=None, method='ICDF')

        Draw samples from a log-normal distribution.

        Draw samples from a log-normal distribution with specified mean,
        standard deviation, and array shape.  Note that the mean and standard
        deviation are not the values for the distribution itself, but of the
        underlying normal distribution it is derived from.

        Parameters
        ----------
        mean : float
            Mean value of the underlying normal distribution
        sigma : float, > 0.
            Standard deviation of the underlying normal distribution
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        method : 'ICDF, 'BoxMuller', optional
            Sampling method used by Intel MKL. Can also be specified using
            tokens mkl_random.ICDF, mkl_random.BOXMULLER

        Returns
        -------
        samples : ndarray or float
            The desired samples. An array of the same shape as `size` if given,
            if `size` is None a float is returned.

        See Also
        --------
        scipy.stats.lognorm : probability density function, distribution,
            cumulative density function, etc.

        Notes
        -----
        A variable `x` has a log-normal distribution if `log(x)` is normally
        distributed.  The probability density function for the log-normal
        distribution is:

        .. math:: p(x) = \\frac{1}{\\sigma x \\sqrt{2\\pi}}
                         e^{(-\\frac{(ln(x)-\\mu)^2}{2\\sigma^2})}

        where :math:`\\mu` is the mean and :math:`\\sigma` is the standard
        deviation of the normally distributed logarithm of the variable.
        A log-normal distribution results if a random variable is the *product*
        of a large number of independent, identically-distributed variables in
        the same way that a normal distribution results if the variable is the
        *sum* of a large number of independent, identically-distributed
        variables.

        References
        ----------
        .. [1] Limpert, E., Stahel, W. A., and Abbt, M., "Log-normal
               Distributions across the Sciences: Keys and Clues,"
               BioScience, Vol. 51, No. 5, May, 2001.
               http://stat.ethz.ch/~stahel/lognormal/bioscience.pdf
        .. [2] Reiss, R.D. and Thomas, M., "Statistical Analysis of Extreme
               Values," Basel: Birkhauser Verlag, 2001, pp. 31-32.

        Examples
        --------
        Draw samples from the distribution:

        >>> mu, sigma = 3., 1. # mean and standard deviation
        >>> s = mkl_random.lognormal(mu, sigma, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 100, normed=True, align='mid')

        >>> x = np.linspace(min(bins), max(bins), 10000)
        >>> pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
        ...        / (x * sigma * np.sqrt(2 * np.pi)))

        >>> plt.plot(x, pdf, linewidth=2, color='r')
        >>> plt.axis('tight')
        >>> plt.show()

        Demonstrate that taking the products of random samples from a uniform
        distribution can be fit well by a log-normal probability density
        function.

        >>> # Generate a thousand samples: each is the product of 100 random
        >>> # values, drawn from a normal distribution.
        >>> b = []
        >>> for i in range(1000):
        ...    a = 10. + mkl_random.random(100)
        ...    b.append(np.product(a))

        >>> b = np.array(b) / np.min(b) # scale values to be positive
        >>> count, bins, ignored = plt.hist(b, 100, normed=True, align='mid')
        >>> sigma = np.std(np.log(b))
        >>> mu = np.mean(np.log(b))

        >>> x = np.linspace(min(bins), max(bins), 10000)
        >>> pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
        ...        / (x * sigma * np.sqrt(2 * np.pi)))

        >>> plt.plot(x, pdf, color='r', linewidth=2)
        >>> plt.show()

        """
        cdef cnp.ndarray omean, osigma
        cdef double fmean, fsigma

        fmean = PyFloat_AsDouble(mean)
        fsigma = PyFloat_AsDouble(sigma)

        if not PyErr_Occurred():
            if fsigma <= 0:
                raise ValueError("sigma <= 0")
            method = choose_method(method, [ICDF, BOXMULLER], _method_alias_dict_gaussian_short)
            if method is ICDF:
                return vec_cont2_array_sc(self.internal_state, irk_lognormal_vec_ICDF, size,
                                  fmean, fsigma, self.lock)
            else:
                return vec_cont2_array_sc(self.internal_state, irk_lognormal_vec_BM, size,
                                  fmean, fsigma, self.lock)

        PyErr_Clear()

        omean = cnp.PyArray_FROM_OTF(mean, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        osigma = cnp.PyArray_FROM_OTF(sigma, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        if np.any(np.less_equal(osigma, 0.0)):
            raise ValueError("sigma <= 0.0")

        method = choose_method(method, [ICDF, BOXMULLER], _method_alias_dict_gaussian_short)
        if method is ICDF:
            return vec_cont2_array(self.internal_state, irk_lognormal_vec_ICDF, size,
                                  omean, osigma, self.lock)
        else:
            return vec_cont2_array(self.internal_state, irk_lognormal_vec_BM, size,
                                  omean, osigma, self.lock)


    def rayleigh(self, scale=1.0, size=None):
        """
        rayleigh(scale=1.0, size=None)

        Draw samples from a Rayleigh distribution.

        The :math:`\\chi` and Weibull distributions are generalizations of the
        Rayleigh.

        Parameters
        ----------
        scale : scalar
            Scale, also equals the mode. Should be >= 0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Notes
        -----
        The probability density function for the Rayleigh distribution is

        .. math:: P(x;scale) = \\frac{x}{scale^2}e^{\\frac{-x^2}{2 \\cdotp scale^2}}

        The Rayleigh distribution would arise, for example, if the East
        and North components of the wind velocity had identical zero-mean
        Gaussian distributions.  Then the wind speed would have a Rayleigh
        distribution.

        References
        ----------
        .. [1] Brighton Webs Ltd., "Rayleigh Distribution,"
               http://www.brighton-webs.co.uk/distributions/rayleigh.asp
        .. [2] Wikipedia, "Rayleigh distribution"
               http://en.wikipedia.org/wiki/Rayleigh_distribution

        Examples
        --------
        Draw values from the distribution and plot the histogram

        >>> values = hist(mkl_random.rayleigh(3, 100000), bins=200, normed=True)

        Wave heights tend to follow a Rayleigh distribution. If the mean wave
        height is 1 meter, what fraction of waves are likely to be larger than 3
        meters?

        >>> meanvalue = 1
        >>> modevalue = np.sqrt(2 / np.pi) * meanvalue
        >>> s = mkl_random.rayleigh(modevalue, 1000000)

        The percentage of waves larger than 3 meters is:

        >>> 100.*sum(s>3)/1000000.
        0.087300000000000003

        """
        cdef cnp.ndarray oscale
        cdef double fscale

        fscale = PyFloat_AsDouble(scale)

        if not PyErr_Occurred():
            if fscale <= 0:
                raise ValueError("scale <= 0")
            return vec_cont1_array_sc(self.internal_state, irk_rayleigh_vec, size,
                                  fscale, self.lock)

        PyErr_Clear()

        oscale = <cnp.ndarray>cnp.PyArray_FROM_OTF(scale, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        if np.any(np.less_equal(oscale, 0.0)):
            raise ValueError("scale <= 0.0")
        return vec_cont1_array(self.internal_state, irk_rayleigh_vec, size, oscale,
                           self.lock)

    def wald(self, mean, scale, size=None):
        """
        wald(mean, scale, size=None)

        Draw samples from a Wald, or inverse Gaussian, distribution.

        As the scale approaches infinity, the distribution becomes more like a
        Gaussian. Some references claim that the Wald is an inverse Gaussian
        with mean equal to 1, but this is by no means universal.

        The inverse Gaussian distribution was first studied in relationship to
        Brownian motion. In 1956 M.C.K. Tweedie used the name inverse Gaussian
        because there is an inverse relationship between the time to cover a
        unit distance and distance covered in unit time.

        Parameters
        ----------
        mean : scalar
            Distribution mean, should be > 0.
        scale : scalar
            Scale parameter, should be >= 0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
            Drawn sample, all greater than zero.

        Notes
        -----
        The probability density function for the Wald distribution is

        .. math:: P(x;mean,scale) = \\sqrt{\\frac{scale}{2\\pi x^3}}e^
                                    \\frac{-scale(x-mean)^2}{2\\cdotp mean^2x}

        As noted above the inverse Gaussian distribution first arise
        from attempts to model Brownian motion. It is also a
        competitor to the Weibull for use in reliability modeling and
        modeling stock returns and interest rate processes.

        References
        ----------
        .. [1] Brighton Webs Ltd., Wald Distribution,
               http://www.brighton-webs.co.uk/distributions/wald.asp
        .. [2] Chhikara, Raj S., and Folks, J. Leroy, "The Inverse Gaussian
               Distribution: Theory : Methodology, and Applications", CRC Press,
               1988.
        .. [3] Wikipedia, "Wald distribution"
               http://en.wikipedia.org/wiki/Wald_distribution

        Examples
        --------
        Draw values from the distribution and plot the histogram:

        >>> import matplotlib.pyplot as plt
        >>> h = plt.hist(mkl_random.wald(3, 2, 100000), bins=200, normed=True)
        >>> plt.show()

        """
        cdef cnp.ndarray omean, oscale
        cdef double fmean, fscale

        fmean = PyFloat_AsDouble(mean)
        fscale = PyFloat_AsDouble(scale)
        if not PyErr_Occurred():
            if fmean <= 0:
                raise ValueError("mean <= 0")
            if fscale <= 0:
                raise ValueError("scale <= 0")
            return vec_cont2_array_sc(self.internal_state, irk_wald_vec, size, fmean,
                                  fscale, self.lock)

        PyErr_Clear()
        omean = cnp.PyArray_FROM_OTF(mean, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        oscale = cnp.PyArray_FROM_OTF(scale, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        if np.any(np.less_equal(omean,0.0)):
            raise ValueError("mean <= 0.0")
        elif np.any(np.less_equal(oscale,0.0)):
            raise ValueError("scale <= 0.0")
        return vec_cont2_array(self.internal_state, irk_wald_vec, size, omean, oscale,
                           self.lock)

    def triangular(self, left, mode, right, size=None):
        """
        triangular(left, mode, right, size=None)

        Draw samples from the triangular distribution.

        The triangular distribution is a continuous probability
        distribution with lower limit left, peak at mode, and upper
        limit right. Unlike the other distributions, these parameters
        directly define the shape of the pdf.

        Parameters
        ----------
        left : scalar
            Lower limit.
        mode : scalar
            The value where the peak of the distribution occurs.
            The value should fulfill the condition ``left <= mode <= right``.
        right : scalar
            Upper limit, should be larger than `left`.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
            The returned samples all lie in the interval [left, right].

        Notes
        -----
        The probability density function for the triangular distribution is

        .. math:: P(x;l, m, r) = \\begin{cases}
                  \\frac{2(x-l)}{(r-l)(m-l)}& \\text{for $l \\leq x \\leq m$},\\\\
                  \\frac{2(r-x)}{(r-l)(r-m)}& \\text{for $m \\leq x \\leq r$},\\\\
                  0& \\text{otherwise}.
                  \\end{cases}

        The triangular distribution is often used in ill-defined
        problems where the underlying distribution is not known, but
        some knowledge of the limits and mode exists. Often it is used
        in simulations.

        References
        ----------
        .. [1] Wikipedia, "Triangular distribution"
               http://en.wikipedia.org/wiki/Triangular_distribution

        Examples
        --------
        Draw values from the distribution and plot the histogram:

        >>> import matplotlib.pyplot as plt
        >>> h = plt.hist(mkl_random.triangular(-3, 0, 8, 100000), bins=200,
        ...              normed=True)
        >>> plt.show()

        """
        cdef cnp.ndarray oleft, omode, oright
        cdef double fleft, fmode, fright

        fleft = PyFloat_AsDouble(left)
        fright = PyFloat_AsDouble(right)
        fmode = PyFloat_AsDouble(mode)
        if not PyErr_Occurred():
            if fleft > fmode:
                raise ValueError("left > mode")
            if fmode > fright:
                raise ValueError("mode > right")
            if fleft == fright:
                raise ValueError("left == right")
            return vec_cont3_array_sc(self.internal_state, irk_triangular_vec, size,
                                  fleft, fmode, fright, self.lock)

        PyErr_Clear()
        oleft = <cnp.ndarray>cnp.PyArray_FROM_OTF(left, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        omode = <cnp.ndarray>cnp.PyArray_FROM_OTF(mode, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        oright = <cnp.ndarray>cnp.PyArray_FROM_OTF(right, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)

        if np.any(np.greater(oleft, omode)):
            raise ValueError("left > mode")
        if np.any(np.greater(omode, oright)):
            raise ValueError("mode > right")
        if np.any(np.equal(oleft, oright)):
            raise ValueError("left == right")
        return vec_cont3_array(self.internal_state, irk_triangular_vec, size, oleft,
                           omode, oright, self.lock)

    # Complicated, discrete distributions:
    def binomial(self, n, p, size=None):
        """
        binomial(n, p, size=None)

        Draw samples from a binomial distribution.

        Samples are drawn from a binomial distribution with specified
        parameters, n trials and p probability of success where
        n an integer >= 0 and p is in the interval [0,1]. (n may be
        input as a float, but it is truncated to an integer in use)

        Parameters
        ----------
        n : float (but truncated to an integer)
                parameter, >= 0.
        p : float
                parameter, >= 0 and <=1.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
                  where the values are all integers in  [0, n].

        See Also
        --------
        scipy.stats.distributions.binom : probability density function,
            distribution or cumulative density function, etc.

        Notes
        -----
        The probability density for the binomial distribution is

        .. math:: P(N) = \\binom{n}{N}p^N(1-p)^{n-N},

        where :math:`n` is the number of trials, :math:`p` is the probability
        of success, and :math:`N` is the number of successes.

        When estimating the standard error of a proportion in a population by
        using a random sample, the normal distribution works well unless the
        product p*n <=5, where p = population proportion estimate, and n =
        number of samples, in which case the binomial distribution is used
        instead. For example, a sample of 15 people shows 4 who are left
        handed, and 11 who are right handed. Then p = 4/15 = 27%. 0.27*15 = 4,
        so the binomial distribution should be used in this case.

        References
        ----------
        .. [1] Dalgaard, Peter, "Introductory Statistics with R",
               Springer-Verlag, 2002.
        .. [2] Glantz, Stanton A. "Primer of Biostatistics.", McGraw-Hill,
               Fifth Edition, 2002.
        .. [3] Lentner, Marvin, "Elementary Applied Statistics", Bogden
               and Quigley, 1972.
        .. [4] Weisstein, Eric W. "Binomial Distribution." From MathWorld--A
               Wolfram Web Resource.
               http://mathworld.wolfram.com/BinomialDistribution.html
        .. [5] Wikipedia, "Binomial-distribution",
               http://en.wikipedia.org/wiki/Binomial_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> n, p = 10, .5  # number of trials, probability of each trial
        >>> s = mkl_random.binomial(n, p, 1000)
        # result of flipping a coin 10 times, tested 1000 times.

        A real world example. A company drills 9 wild-cat oil exploration
        wells, each with an estimated probability of success of 0.1. All nine
        wells fail. What is the probability of that happening?

        Let's do 20,000 trials of the model, and count the number that
        generate zero positive results.

        >>> sum(mkl_random.binomial(9, 0.1, 20000) == 0)/20000.
        # answer = 0.38885, or 38%.

        """
        cdef cnp.ndarray on, op
        cdef long ln
        cdef double fp

        fp = PyFloat_AsDouble(p)
        ln = PyInt_AsLong(n)
        if not PyErr_Occurred():
            if ln < 0:
                raise ValueError("n < 0")
            if fp < 0:
                raise ValueError("p < 0")
            elif fp > 1:
                raise ValueError("p > 1")
            elif np.isnan(fp):
                raise ValueError("p is nan")
            if n > int(2**31-1):
                raise ValueError("n > 2147483647")
            else:
                return vec_discnp_array_sc(self.internal_state, irk_binomial_vec, size, <int> ln,
                            fp, self.lock)


        PyErr_Clear()

        on = <cnp.ndarray>cnp.PyArray_FROM_OTF(n, cnp.NPY_LONG, cnp.NPY_ARRAY_IN_ARRAY)
        op = <cnp.ndarray>cnp.PyArray_FROM_OTF(p, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_IN_ARRAY)
        if np.any(np.less(n, 0)):
            raise ValueError("n < 0")
        if np.any(np.less(op, 0)):
            raise ValueError("p < 0")
        if np.any(np.greater(op, 1)):
            raise ValueError("p > 1")
        if np.any(np.greater(n, int(2**31-1))):
            raise ValueError("n > 2147483647")

        on = on.astype(np.int32, casting='unsafe')
        return vec_discnp_array(self.internal_state, irk_binomial_vec, size, on, op,
                            self.lock)

    def negative_binomial(self, n, p, size=None):
        """
        negative_binomial(n, p, size=None)

        Draw samples from a negative binomial distribution.

        Samples are drawn from a negative binomial distribution with specified
        parameters, `n` trials and `p` probability of success where `n` is an
        integer > 0 and `p` is in the interval [0, 1].

        Parameters
        ----------
        n : int
            Parameter, > 0.
        p : float
            Parameter, >= 0 and <=1.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : int or ndarray of ints
            Drawn samples.

        Notes
        -----
        The probability density for the negative binomial distribution is

        .. math:: P(N;n,p) = \\binom{N+n-1}{n-1}p^{n}(1-p)^{N},

        where :math:`n-1` is the number of successes, :math:`p` is the
        probability of success, and :math:`N+n-1` is the number of trials.
        The negative binomial distribution gives the probability of n-1
        successes and N failures in N+n-1 trials, and success on the (N+n)th
        trial.

        If one throws a die repeatedly until the third time a "1" appears,
        then the probability distribution of the number of non-"1"s that
        appear before the third "1" is a negative binomial distribution.

        References
        ----------
        .. [1] Weisstein, Eric W. "Negative Binomial Distribution." From
               MathWorld--A Wolfram Web Resource.
               http://mathworld.wolfram.com/NegativeBinomialDistribution.html
        .. [2] Wikipedia, "Negative binomial distribution",
               http://en.wikipedia.org/wiki/Negative_binomial_distribution

        Examples
        --------
        Draw samples from the distribution:

        A real world example. A company drills wild-cat oil
        exploration wells, each with an estimated probability of
        success of 0.1.  What is the probability of having one success
        for each successive well, that is what is the probability of a
        single success after drilling 5 wells, after 6 wells, etc.?

        >>> s = mkl_random.negative_binomial(1, 0.1, 100000)
        >>> for i in range(1, 11):
        ...    probability = sum(s<i) / 100000.
        ...    print i, "wells drilled, probability of one success =", probability

        """
        cdef cnp.ndarray on
        cdef cnp.ndarray op
        cdef double fn
        cdef double fp

        fp = PyFloat_AsDouble(p)
        fn = PyFloat_AsDouble(n)
        if not PyErr_Occurred():
            if fn <= 0:
                raise ValueError("n <= 0")
            if fp < 0:
                raise ValueError("p < 0")
            elif fp > 1:
                raise ValueError("p > 1")
            return vec_discdd_array_sc(self.internal_state, irk_negbinomial_vec,
                                   size, fn, fp, self.lock)

        PyErr_Clear()

        on = <cnp.ndarray>cnp.PyArray_FROM_OTF(n, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_IN_ARRAY)
        op = <cnp.ndarray>cnp.PyArray_FROM_OTF(p, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_IN_ARRAY)
        if np.any(np.less_equal(n, 0)):
            raise ValueError("n <= 0")
        if np.any(np.less(p, 0)):
            raise ValueError("p < 0")
        if np.any(np.greater(p, 1)):
            raise ValueError("p > 1")
        return vec_discdd_array(self.internal_state, irk_negbinomial_vec, size,
                            on, op, self.lock)

    def poisson(self, lam=1.0, size=None, method=POISNORM):
        """
        poisson(lam=1.0, size=None, method='POISNORM')

        Draw samples from a Poisson distribution.

        The Poisson distribution is the limit of the binomial distribution
        for large N.

        Parameters
        ----------
        lam : float or sequence of float
            Expectation of interval, should be >= 0. A sequence of expectation
            intervals must be broadcastable over the requested size.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        method : 'POISNORM, 'PTPE', optional
            Sampling method used by Intel MKL. Can also be specified using
            tokens mkl_random.POISNORM, mkl_random.PTPE

        Returns
        -------
        samples : ndarray or scalar
            The drawn samples, of shape *size*, if it was provided.

        Notes
        -----
        The Poisson distribution

        .. math:: f(k; \\lambda)=\\frac{\\lambda^k e^{-\\lambda}}{k!}

        For events with an expected separation :math:`\\lambda` the Poisson
        distribution :math:`f(k; \\lambda)` describes the probability of
        :math:`k` events occurring within the observed
        interval :math:`\\lambda`.

        Because the output is limited to the range of the C long type, a
        ValueError is raised when `lam` is within 10 sigma of the maximum
        representable value.

        References
        ----------
        .. [1] Weisstein, Eric W. "Poisson Distribution."
               From MathWorld--A Wolfram Web Resource.
               http://mathworld.wolfram.com/PoissonDistribution.html
        .. [2] Wikipedia, "Poisson distribution",
               http://en.wikipedia.org/wiki/Poisson_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> import numpy as np
        >>> s = mkl_random.poisson(5, 10000)

        Display histogram of the sample:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 14, normed=True)
        >>> plt.show()

        Draw each 100 values for lambda 100 and 500:

        >>> s = mkl_random.poisson(lam=(100., 500.), size=(100, 2))

        """
        cdef cnp.ndarray olam
        cdef double flam
        flam = PyFloat_AsDouble(lam)
        if not PyErr_Occurred():
            if lam < 0:
                raise ValueError("lam < 0")
            if lam > self.poisson_lam_max:
                raise ValueError("lam value too large")
            method = choose_method(method, [POISNORM, PTPE], _method_alias_dict_poisson);
            if method is POISNORM:
                return vec_discd_array_sc(self.internal_state, irk_poisson_vec_POISNORM, size, flam, self.lock)
            else:
                return vec_discd_array_sc(self.internal_state, irk_poisson_vec_PTPE, size, flam, self.lock)

        PyErr_Clear()

        olam = <cnp.ndarray>cnp.PyArray_FROM_OTF(lam, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_IN_ARRAY)
        if np.any(np.less(olam, 0)):
            raise ValueError("lam < 0")
        if np.any(np.greater(olam, self.poisson_lam_max)):
            raise ValueError("lam value too large.")
        method = choose_method(method, [POISNORM, PTPE], _method_alias_dict_poisson);
        if method is POISNORM:
            return vec_Poisson_array(self.internal_state, irk_poisson_vec_V, irk_poisson_vec_POISNORM, size, olam, self.lock)
        else:
            return vec_discd_array(self.internal_state, irk_poisson_vec_PTPE, size, olam, self.lock)


    def zipf(self, a, size=None):
        """
        zipf(a, size=None)

        Draw samples from a Zipf distribution.

        Samples are drawn from a Zipf distribution with specified parameter
        `a` > 1.

        The Zipf distribution (also known as the zeta distribution) is a
        continuous probability distribution that satisfies Zipf's law: the
        frequency of an item is inversely proportional to its rank in a
        frequency table.

        Parameters
        ----------
        a : float > 1
            Distribution parameter.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : scalar or ndarray
            The returned samples are greater than or equal to one.

        See Also
        --------
        scipy.stats.distributions.zipf : probability density function,
            distribution, or cumulative density function, etc.

        Notes
        -----
        The probability density for the Zipf distribution is

        .. math:: p(x) = \\frac{x^{-a}}{\\zeta(a)},

        where :math:`\\zeta` is the Riemann Zeta function.

        It is named for the American linguist George Kingsley Zipf, who noted
        that the frequency of any word in a sample of a language is inversely
        proportional to its rank in the frequency table.

        References
        ----------
        .. [1] Zipf, G. K., "Selected Studies of the Principle of Relative
               Frequency in Language," Cambridge, MA: Harvard Univ. Press,
               1932.

        Examples
        --------
        Draw samples from the distribution:

        >>> a = 2. # parameter
        >>> s = mkl_random.zipf(a, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> import scipy.special as sps
        Truncate s values at 50 so plot is interesting
        >>> count, bins, ignored = plt.hist(s[s<50], 50, normed=True)
        >>> x = np.arange(1., 50.)
        >>> y = x**(-a)/sps.zetac(a)
        >>> plt.plot(x, y/max(y), linewidth=2, color='r')
        >>> plt.show()

        """
        cdef cnp.ndarray oa
        cdef double fa

        fa = PyFloat_AsDouble(a)
        if not PyErr_Occurred():
            if fa <= 1.0:
                raise ValueError("a <= 1.0")
            return vec_long_discd_array_sc(self.internal_state, irk_zipf_long_vec, size, fa,
                                  self.lock)

        PyErr_Clear()

        oa = <cnp.ndarray>cnp.PyArray_FROM_OTF(a, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_IN_ARRAY)
        if np.any(np.less_equal(oa, 1.0)):
            raise ValueError("a <= 1.0")
        return vec_long_discd_array(self.internal_state, irk_zipf_long_vec, size, oa, self.lock)

    def geometric(self, p, size=None):
        """
        geometric(p, size=None)

        Draw samples from the geometric distribution.

        Bernoulli trials are experiments with one of two outcomes:
        success or failure (an example of such an experiment is flipping
        a coin).  The geometric distribution models the number of trials
        that must be run in order to achieve success.  It is therefore
        supported on the positive integers, ``k = 1, 2, ...``.

        The probability mass function of the geometric distribution is

        .. math:: f(k) = (1 - p)^{k - 1} p

        where `p` is the probability of success of an individual trial.

        Parameters
        ----------
        p : float
            The probability of success of an individual trial.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : ndarray
            Samples from the geometric distribution, shaped according to
            `size`.

        Examples
        --------
        Draw ten thousand values from the geometric distribution,
        with the probability of an individual success equal to 0.35:

        >>> z = mkl_random.geometric(p=0.35, size=10000)

        How many trials succeeded after a single run?

        >>> (z == 1).sum() / 10000.
        0.34889999999999999 #random

        """
        cdef cnp.ndarray op
        cdef double fp

        fp = PyFloat_AsDouble(p)
        if not PyErr_Occurred():
            if fp <= 0.0:
                raise ValueError("p <= 0.0")
            if fp > 1.0:
                raise ValueError("p > 1.0")
            return vec_discd_array_sc(self.internal_state, irk_geometric_vec, size, fp,
                                  self.lock)

        PyErr_Clear()


        op = <cnp.ndarray>cnp.PyArray_FROM_OTF(p, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_IN_ARRAY)
        if np.any(np.less_equal(op, 0.0)):
            raise ValueError("p < 0.0")
        if np.any(np.greater(op, 1.0)):
            raise ValueError("p > 1.0")
        return vec_discd_array(self.internal_state, irk_geometric_vec, size, op,
                           self.lock)

    def hypergeometric(self, ngood, nbad, nsample, size=None):
        """
        hypergeometric(ngood, nbad, nsample, size=None)

        Draw samples from a Hypergeometric distribution.

        Samples are drawn from a hypergeometric distribution with specified
        parameters, ngood (ways to make a good selection), nbad (ways to make
        a bad selection), and nsample = number of items sampled, which is less
        than or equal to the sum ngood + nbad.

        Parameters
        ----------
        ngood : int or array_like
            Number of ways to make a good selection.  Must be nonnegative.
        nbad : int or array_like
            Number of ways to make a bad selection.  Must be nonnegative.
        nsample : int or array_like
            Number of items sampled.  Must be at least 1 and at most
            ``ngood + nbad``.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(d1, d2, d3)``, then
            ``d1 * d2 * d3`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
            The values are all integers in  [0, n].

        See Also
        --------
        scipy.stats.distributions.hypergeom : probability density function,
            distribution or cumulative density function, etc.

        Notes
        -----
        The probability density for the Hypergeometric distribution is

        .. math:: P(x) = \\frac{\\binom{m}{x}\\binom{N-m}{n-x}}{\\binom{N}{n}},

        where :math:`0 \\le x \\le m` and :math:`n+m-N \\le x \\le n`

        for P(x) the probability of x successes, m = ngood, N = ngood + nbad, and
        n = number of samples.

        Consider an urn with black and white marbles in it, ngood of them
        black and nbad are white. If you draw nsample balls without
        replacement, then the hypergeometric distribution describes the
        distribution of black balls in the drawn sample.

        Note that this distribution is very similar to the binomial
        distribution, except that in this case, samples are drawn without
        replacement, whereas in the Binomial case samples are drawn with
        replacement (or the sample space is infinite). As the sample space
        becomes large, this distribution approaches the binomial.

        References
        ----------
        .. [1] Lentner, Marvin, "Elementary Applied Statistics", Bogden
               and Quigley, 1972.
        .. [2] Weisstein, Eric W. "Hypergeometric Distribution." From
               MathWorld--A Wolfram Web Resource.
               http://mathworld.wolfram.com/HypergeometricDistribution.html
        .. [3] Wikipedia, "Hypergeometric-distribution",
               http://en.wikipedia.org/wiki/Hypergeometric_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> ngood, nbad, nsamp = 100, 2, 10
        # number of good, number of bad, and number of samples
        >>> s = mkl_random.hypergeometric(ngood, nbad, nsamp, 1000)
        >>> hist(s)
        #   note that it is very unlikely to grab both bad items

        Suppose you have an urn with 15 white and 15 black marbles.
        If you pull 15 marbles at random, how likely is it that
        12 or more of them are one color?

        >>> s = mkl_random.hypergeometric(15, 15, 15, 100000)
        >>> sum(s>=12)/100000. + sum(s<=3)/100000.
        #   answer = 0.003 ... pretty unlikely!

        """
        cdef cnp.ndarray ongood, onbad, onsample, otot
        cdef long lngood, lnbad, lnsample, lntot

        lngood = PyInt_AsLong(ngood)
        lnbad = PyInt_AsLong(nbad)
        lnsample = PyInt_AsLong(nsample)
        if not PyErr_Occurred():
            if lngood < 0:
                raise ValueError("ngood < 0")
            if lnbad < 0:
                raise ValueError("nbad < 0")
            if lnsample < 1:
                raise ValueError("nsample < 1")
            if ((<int> lngood) != lngood) or ((<int> lnbad) != lnbad) or ((<int> lnsample) != lnsample):
                raise ValueError("All parameters should not exceed 2147483647")
            lntot = lngood + lnbad
            if lntot < lnsample:
                raise ValueError("ngood + nbad < nsample")
            return vec_discnmN_array_sc(self.internal_state, irk_hypergeometric_vec,
                                    size, lntot, lnsample, lngood, self.lock)

        PyErr_Clear()

        ongood = <cnp.ndarray>cnp.PyArray_FROM_OTF(ngood, cnp.NPY_LONG, cnp.NPY_ARRAY_IN_ARRAY)
        onbad = <cnp.ndarray>cnp.PyArray_FROM_OTF(nbad, cnp.NPY_LONG, cnp.NPY_ARRAY_IN_ARRAY)
        onsample = <cnp.ndarray>cnp.PyArray_FROM_OTF(nsample, cnp.NPY_LONG, cnp.NPY_ARRAY_IN_ARRAY)
        if np.any(np.less(ongood, 0)):
            raise ValueError("ngood < 0")
        if np.any(np.less(onbad, 0)):
            raise ValueError("nbad < 0")
        if np.any(np.less(onsample, 1)):
            raise ValueError("nsample < 1")
        otot = np.asarray(np.add(ongood, onbad));
        if np.any(np.less_equal(otot, 0)):
            raise ValueError("Number of balls in each urn should not exceed 2147483647")
        if np.any(np.less(otot,onsample)):
            raise ValueError("ngood + nbad < nsample")

        otot = otot.astype(np.int32, casting='unsafe')
        onsample = onsample.astype(np.int32, casting='unsafe')
        ongood = ongood.astype(np.int32, casting='unsafe')
        return vec_discnmN_array(self.internal_state, irk_hypergeometric_vec, size,
                             otot, onsample, ongood, self.lock)

    def logseries(self, p, size=None):
        """
        logseries(p, size=None)

        Draw samples from a logarithmic series distribution.

        Samples are drawn from a log series distribution with specified
        shape parameter, 0 < ``p`` < 1.

        Parameters
        ----------
        loc : float

        scale : float > 0.

        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
                  where the values are all integers in  [0, n].

        See Also
        --------
        scipy.stats.distributions.logser : probability density function,
            distribution or cumulative density function, etc.

        Notes
        -----
        The probability density for the Log Series distribution is

        .. math:: P(k) = \\frac{-p^k}{k \\ln(1-p)},

        where p = probability.

        The log series distribution is frequently used to represent species
        richness and occurrence, first proposed by Fisher, Corbet, and
        Williams in 1943 [2].  It may also be used to model the numbers of
        occupants seen in cars [3].

        References
        ----------
        .. [1] Buzas, Martin A.; Culver, Stephen J.,  Understanding regional
               species diversity through the log series distribution of
               occurrences: BIODIVERSITY RESEARCH Diversity & Distributions,
               Volume 5, Number 5, September 1999 , pp. 187-195(9).
        .. [2] Fisher, R.A,, A.S. Corbet, and C.B. Williams. 1943. The
               relation between the number of species and the number of
               individuals in a random sample of an animal population.
               Journal of Animal Ecology, 12:42-58.
        .. [3] D. J. Hand, F. Daly, D. Lunn, E. Ostrowski, A Handbook of Small
               Data Sets, CRC Press, 1994.
        .. [4] Wikipedia, "Logarithmic-distribution",
               http://en.wikipedia.org/wiki/Logarithmic-distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> a = .6
        >>> s = mkl_random.logseries(a, 10000)
        >>> count, bins, ignored = plt.hist(s)

        #   plot against distribution

        >>> def logseries(k, p):
        ...     return -p**k/(k*log(1-p))
        >>> plt.plot(bins, logseries(bins, a)*count.max()/
                     logseries(bins, a).max(), 'r')
        >>> plt.show()

        """
        cdef cnp.ndarray op
        cdef double fp

        fp = PyFloat_AsDouble(p)
        if not PyErr_Occurred():
            if fp <= 0.0:
                raise ValueError("p <= 0.0")
            if fp >= 1.0:
                raise ValueError("p >= 1.0")
            return vec_discd_array_sc(self.internal_state, irk_logseries_vec, size, fp,
                                  self.lock)

        PyErr_Clear()

        op = <cnp.ndarray>cnp.PyArray_FROM_OTF(p, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        if np.any(np.less_equal(op, 0.0)):
            raise ValueError("p <= 0.0")
        if np.any(np.greater_equal(op, 1.0)):
            raise ValueError("p >= 1.0")
        return vec_discd_array(self.internal_state, irk_logseries_vec, size, op,
                           self.lock)

    # Multivariate distributions:
    def multivariate_normal(self, mean, cov, size=None):
        """
        multivariate_normal(mean, cov[, size])

        Draw random samples from a multivariate normal distribution.

        The multivariate normal, multinormal or Gaussian distribution is a
        generalization of the one-dimensional normal distribution to higher
        dimensions.  Such a distribution is specified by its mean and
        covariance matrix.  These parameters are analogous to the mean
        (average or "center") and variance (standard deviation, or "width,"
        squared) of the one-dimensional normal distribution.

        Parameters
        ----------
        mean : 1-D array_like, of length N
            Mean of the N-dimensional distribution.
        cov : 2-D array_like, of shape (N, N)
            Covariance matrix of the distribution. It must be symmetric and
            positive-semidefinite for proper sampling.
        size : int or tuple of ints, optional
            Given a shape of, for example, ``(m,n,k)``, ``m*n*k`` samples are
            generated, and packed in an `m`-by-`n`-by-`k` arrangement.  Because
            each sample is `N`-dimensional, the output shape is ``(m,n,k,N)``.
            If no shape is specified, a single (`N`-D) sample is returned.

        Returns
        -------
        out : ndarray
            The drawn samples, of shape *size*, if that was provided.  If not,
            the shape is ``(N,)``.

            In other words, each entry ``out[i,j,...,:]`` is an N-dimensional
            value drawn from the distribution.

        Notes
        -----
        The mean is a coordinate in N-dimensional space, which represents the
        location where samples are most likely to be generated.  This is
        analogous to the peak of the bell curve for the one-dimensional or
        univariate normal distribution.

        Covariance indicates the level to which two variables vary together.
        From the multivariate normal distribution, we draw N-dimensional
        samples, :math:`X = [x_1, x_2, ... x_N]`.  The covariance matrix
        element :math:`C_{ij}` is the covariance of :math:`x_i` and :math:`x_j`.
        The element :math:`C_{ii}` is the variance of :math:`x_i` (i.e. its
        "spread").

        Instead of specifying the full covariance matrix, popular
        approximations include:

          - Spherical covariance (*cov* is a multiple of the identity matrix)
          - Diagonal covariance (*cov* has non-negative elements, and only on
            the diagonal)

        This geometrical property can be seen in two dimensions by plotting
        generated data-points:

        >>> mean = [0, 0]
        >>> cov = [[1, 0], [0, 100]]  # diagonal covariance

        Diagonal covariance means that points are oriented along x or y-axis:

        >>> import matplotlib.pyplot as plt
        >>> x, y = mkl_random.multivariate_normal(mean, cov, 5000).T
        >>> plt.plot(x, y, 'x')
        >>> plt.axis('equal')
        >>> plt.show()

        Note that the covariance matrix must be positive semidefinite (a.k.a.
        nonnegative-definite). Otherwise, the behavior of this method is
        undefined and backwards compatibility is not guaranteed.

        References
        ----------
        .. [1] Papoulis, A., "Probability, Random Variables, and Stochastic
               Processes," 3rd ed., New York: McGraw-Hill, 1991.
        .. [2] Duda, R. O., Hart, P. E., and Stork, D. G., "Pattern
               Classification," 2nd ed., New York: Wiley, 2001.

        Examples
        --------
        >>> mean = (1, 2)
        >>> cov = [[1, 0], [0, 1]]
        >>> x = mkl_random.multivariate_normal(mean, cov, (3, 3))
        >>> x.shape
        (3, 3, 2)

        The following is probably true, given that 0.6 is roughly twice the
        standard deviation:

        >>> list((x[0,0,:] - mean) < 0.6)
        [True, True]

        """
        from numpy.linalg import svd

        # Check preconditions on arguments
        mean = np.array(mean)
        cov = np.array(cov)
        if size is None:
            shape = []
        elif isinstance(size, (int, long, np.integer)):
            shape = [size]
        else:
            shape = size

        if len(mean.shape) != 1:
               raise ValueError("mean must be 1 dimensional")
        if (len(cov.shape) != 2) or (cov.shape[0] != cov.shape[1]):
               raise ValueError("cov must be 2 dimensional and square")
        if mean.shape[0] != cov.shape[0]:
               raise ValueError("mean and cov must have same length")

        # Compute shape of output and create a matrix of independent
        # standard normally distributed random numbers. The matrix has rows
        # with the same length as mean and as many rows are necessary to
        # form a matrix of shape final_shape.
        final_shape = list(shape[:])
        final_shape.append(mean.shape[0])
        x = self.standard_normal(final_shape).reshape(-1, mean.shape[0])

        # Transform matrix of standard normals into matrix where each row
        # contains multivariate normals with the desired covariance.
        # Compute A such that dot(transpose(A),A) == cov.
        # Then the matrix products of the rows of x and A has the desired
        # covariance. Note that sqrt(s)*v where (u,s,v) is the singular value
        # decomposition of cov is such an A.
        #
        # Also check that cov is positive-semidefinite. If so, the u.T and v
        # matrices should be equal up to roundoff error if cov is
        # symmetrical and the singular value of the corresponding row is
        # not zero. We continue to use the SVD rather than Cholesky in
        # order to preserve current outputs. Note that symmetry has not
        # been checked.
        (u, s, v) = svd(cov)
        neg = (np.sum(u.T * v, axis=1) < 0) & (s > 0)
        if np.any(neg):
            s[neg] = 0.
            warnings.warn("covariance is not positive-semidefinite.",
                          RuntimeWarning)

        x = np.dot(x, np.sqrt(s)[:, None] * v)
        x += mean
        x.shape = tuple(final_shape)
        return x

    def multinormal_cholesky(self, mean, ch, size=None, method=ICDF):
        """
        multivariate_normal(mean, ch, size=None, method='ICDF')

        Draw random samples from a multivariate normal distribution.

        The multivariate normal, multinormal or Gaussian distribution is a
        generalization of the one-dimensional normal distribution to higher
        dimensions.  Such a distribution is specified by its mean and
        covariance matrix, specified by its lower triangular Cholesky factor.
        These parameters are analogous to the mean
        (average or "center") and standard deviation, or "width,"
        of the one-dimensional normal distribution.

        Parameters
        ----------
        mean : 1-D array_like, of length N
            Mean of the N-dimensional distribution.
        ch : 2-D array_like, of shape (N, N)
            Cholesky factor of the covariance matrix of the distribution. Only lower-triangular
            part of the matrix is actually used.
        size : int or tuple of ints, optional
            Given a shape of, for example, ``(m,n,k)``, ``m*n*k`` samples are
            generated, and packed in an `m`-by-`n`-by-`k` arrangement.  Because
            each sample is `N`-dimensional, the output shape is ``(m,n,k,N)``.
            If no shape is specified, a single (`N`-D) sample is returned.
        method : 'ICDF, 'BoxMuller', 'BoxMuller2', optional
            Sampling method used by Intel MKL. Can also be specified using
            tokens mkl_random.ICDF, mkl_random.BOXMULLER, mkl_random.BOXMULLER2

        Returns
        -------
        out : ndarray
            The drawn samples, of shape *size*, if that was provided.  If not,
            the shape is ``(N,)``.

            In other words, each entry ``out[i,j,...,:]`` is an N-dimensional
            value drawn from the distribution.

        Notes
        -----
        The mean is a coordinate in N-dimensional space, which represents the
        location where samples are most likely to be generated.  This is
        analogous to the peak of the bell curve for the one-dimensional or
        univariate normal distribution.

        Covariance indicates the level to which two variables vary together.
        From the multivariate normal distribution, we draw N-dimensional
        samples, :math:`X = [x_1, x_2, ... x_N]`.  The covariance matrix
        element :math:`C_{ij}` is the covariance of :math:`x_i` and :math:`x_j`.
        The element :math:`C_{ii}` is the variance of :math:`x_i` (i.e. its
        "spread").

        Instead of specifying the full covariance matrix, popular
        approximations include:

          - Spherical covariance (*cov* is a multiple of the identity matrix)
          - Diagonal covariance (*cov* has non-negative elements, and only on
            the diagonal)

        This geometrical property can be seen in two dimensions by plotting
        generated data-points:

        >>> mean = [0, 0]
        >>> cov = [[1, 0], [0, 100]]  # diagonal covariance

        Diagonal covariance means that points are oriented along x or y-axis:

        >>> import matplotlib.pyplot as plt
        >>> x, y = mkl_random.multivariate_normal(mean, cov, 5000).T
        >>> plt.plot(x, y, 'x')
        >>> plt.axis('equal')
        >>> plt.show()

        Note that the covariance matrix must be positive semidefinite (a.k.a.
        nonnegative-definite). Otherwise, the behavior of this method is
        undefined and backwards compatibility is not guaranteed.

        References
        ----------
        .. [1] Papoulis, A., "Probability, Random Variables, and Stochastic
               Processes," 3rd ed., New York: McGraw-Hill, 1991.
        .. [2] Duda, R. O., Hart, P. E., and Stork, D. G., "Pattern
               Classification," 2nd ed., New York: Wiley, 2001.

        Examples
        --------
        >>> mean = (1, 2)
        >>> cov = [[1, 0], [0, 1]]
        >>> x = mkl_random.multivariate_normal(mean, cov, (3, 3))
        >>> x.shape
        (3, 3, 2)

        The following is probably true, given that 0.6 is roughly twice the
        standard deviation:

        >>> list((x[0,0,:] - mean) < 0.6)
        [True, True]

        """
        cdef cnp.ndarray resarr "arrayObject_resarr"
        cdef cnp.ndarray marr "arrayObject_marr"
        cdef cnp.ndarray tarr "arrayObject_tarr"
        cdef double *res_data
        cdef double *mean_data
        cdef double *t_data
        cdef cnp.npy_intp dim, n
        cdef ch_st_enum storage_mode

        # Check preconditions on arguments
        marr = <cnp.ndarray>cnp.PyArray_FROM_OTF(mean, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_IN_ARRAY)
        tarr = <cnp.ndarray>cnp.PyArray_FROM_OTF(ch, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_IN_ARRAY)

        if size is None:
            shape = []
        elif isinstance(size, (int, long, np.integer)):
            shape = [size]
        else:
            shape = size

        if marr.ndim != 1:
               raise ValueError("mean must be 1 dimensional")
        dim = marr.shape[0];
        if (tarr.ndim == 2):
            storage_mode = MATRIX
            if (tarr.shape[0] != tarr.shape[1]):
                   raise ValueError("ch must be a square lower triangular 2-dimensional array or a row-packed one-dimensional representation of such")
            if dim != tarr.shape[0]:
                   raise ValueError("mean and ch must have consistent shapes")
        elif (tarr.ndim == 1):
            if (tarr.shape[0] == dim):
                storage_mode = DIAGONAL
            elif (tarr.shape[0] == packed_cholesky_size(dim)):
                storage_mode = PACKED
            else:
                raise ValueError("ch must be a square lower triangular 2-dimensional array or a row-packed one-dimensional representation of such")
        else:
            raise ValueError("ch must be a square lower triangular 2-dimensional array or a row-packed one-dimensional representation of such")

        # Compute shape of output and create a matrix of independent
        # standard normally distributed random numbers. The matrix has rows
        # with the same length as mean and as many rows are necessary to
        # form a matrix of shape final_shape.
        final_shape = list(shape[:])
        final_shape.append(int(dim))

        resarr = <cnp.ndarray>np.empty(final_shape, np.float64)
        res_data = <double*>cnp.PyArray_DATA(resarr)
        mean_data = <double*>cnp.PyArray_DATA(marr)
        t_data = <double*>cnp.PyArray_DATA(tarr)

        n = cnp.PyArray_SIZE(resarr) // dim

        method = choose_method(method, [ICDF, BOXMULLER2, BOXMULLER], _method_alias_dict_gaussian)
        if (method is ICDF):
            irk_multinormal_vec_ICDF(self.internal_state, n, res_data, dim, mean_data, t_data, storage_mode)
        elif (method is BOXMULLER2):
            irk_multinormal_vec_BM2(self.internal_state, n, res_data, dim, mean_data, t_data, storage_mode)
        else:
            irk_multinormal_vec_BM1(self.internal_state, n, res_data, dim, mean_data, t_data, storage_mode)

        return resarr

    def multinomial(self, int n, object pvals, size=None):
        """
        multinomial(n, pvals, size=None, method='ICDF')

        Draw samples from a multinomial distribution.

        The multinomial distribution is a multivariate generalisation of the
        binomial distribution.  Take an experiment with one of ``p``
        possible outcomes.  An example of such an experiment is throwing a dice,
        where the outcome can be 1 through 6.  Each sample drawn from the
        distribution represents `n` such experiments.  Its values,
        ``X_i = [X_0, X_1, ..., X_p]``, represent the number of times the
        outcome was ``i``.

        Parameters
        ----------
        n : int
            Number of experiments.
        pvals : sequence of floats, length p
            Probabilities of each of the ``p`` different outcomes.  These
            should sum to 1 (however, the last element is always assumed to
            account for the remaining probability, as long as
            ``sum(pvals[:-1]) <= 1)``.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        method : 'ICDF, 'BoxMuller', 'BoxMuller2', optional
            Sampling method used by Intel MKL. Can also be specified using
            tokens mkl_random.ICDF, mkl_random.BOXMULLER, mkl_random.BOXMULLER2

        Returns
        -------
        out : ndarray
            The drawn samples, of shape *size*, if that was provided.  If not,
            the shape is ``(N,)``.

            In other words, each entry ``out[i,j,...,:]`` is an N-dimensional
            value drawn from the distribution.

        Examples
        --------
        Throw a dice 20 times:

        >>> mkl_random.multinomial(20, [1/6.]*6, size=1)
        array([[4, 1, 7, 5, 2, 1]])

        It landed 4 times on 1, once on 2, etc.

        Now, throw the dice 20 times, and 20 times again:

        >>> mkl_random.multinomial(20, [1/6.]*6, size=2)
        array([[3, 4, 3, 3, 4, 3],
               [2, 4, 3, 4, 0, 7]])

        For the first run, we threw 3 times 1, 4 times 2, etc.  For the second,
        we threw 2 times 1, 4 times 2, etc.

        A loaded die is more likely to land on number 6:

        >>> mkl_random.multinomial(100, [1/7.]*5 + [2./7.])
        array([11, 16, 14, 17, 16, 26])

        The probability inputs should be normalized. As an implementation
        detail, the value of the last entry is ignored and assumed to take
        up any leftover probability mass, but this should not be relied on.
        A biased coin which has twice as much weight on one side as on the
        other should be sampled like so:

        >>> mkl_random.multinomial(100, [1.0 / 3, 2.0 / 3])  # RIGHT
        array([38, 62])

        not like:

        >>> mkl_random.multinomial(100, [1.0, 2.0])  # WRONG
        array([100,   0])

        """
        cdef cnp.npy_intp d
        cdef cnp.ndarray parr "arrayObject_parr", mnarr "arrayObject_mnarr"
        cdef double *pix
        cdef int *mnix
        cdef cnp.npy_intp i, j, sz
        cdef double Sum
        cdef int dn

        d = len(pvals)
        parr = <cnp.ndarray>cnp.PyArray_ContiguousFromObject(pvals, cnp.NPY_DOUBLE, 1, 1)
        pix = <double*>cnp.PyArray_DATA(parr)

        if kahan_sum(pix, d-1) > (1.0 + 1e-12):
            raise ValueError("sum(pvals[:-1]) > 1.0")

        shape = _shape_from_size(size, d)
        multin = np.zeros(shape, np.int32)

        mnarr = <cnp.ndarray>multin
        mnix = <int*>cnp.PyArray_DATA(mnarr)
        sz = cnp.PyArray_SIZE(mnarr)

        irk_multinomial_vec(self.internal_state, sz // d, mnix, n, d, pix)

        return multin


    def dirichlet(self, object alpha, size=None):
        """
        dirichlet(alpha, size=None)

        Draw samples from the Dirichlet distribution.

        Draw `size` samples of dimension k from a Dirichlet distribution. A
        Dirichlet-distributed random variable can be seen as a multivariate
        generalization of a Beta distribution. Dirichlet pdf is the conjugate
        prior of a multinomial in Bayesian inference.

        Parameters
        ----------
        alpha : array
            Parameter of the distribution (k dimension for sample of
            dimension k).
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray,
            The drawn samples, of shape (size, alpha.ndim).

        Notes
        -----
        .. math:: X \\approx \\prod_{i=1}^{k}{x^{\\alpha_i-1}_i}

        Uses the following property for computation: for each dimension,
        draw a random sample y_i from a standard gamma generator of shape
        `alpha_i`, then
        :math:`X = \\frac{1}{\\sum_{i=1}^k{y_i}} (y_1, \\ldots, y_n)` is
        Dirichlet distributed.

        References
        ----------
        .. [1] David McKay, "Information Theory, Inference and Learning
               Algorithms," chapter 23,
               http://www.inference.phy.cam.ac.uk/mackay/
        .. [2] Wikipedia, "Dirichlet distribution",
               http://en.wikipedia.org/wiki/Dirichlet_distribution

        Examples
        --------
        Taking an example cited in Wikipedia, this distribution can be used if
        one wanted to cut strings (each of initial length 1.0) into K pieces
        with different lengths, where each piece had, on average, a designated
        average length, but allowing some variation in the relative sizes of
        the pieces.

        >>> s = mkl_random.dirichlet((10, 5, 3), 20).transpose()

        >>> plt.barh(range(20), s[0])
        >>> plt.barh(range(20), s[1], left=s[0], color='g')
        >>> plt.barh(range(20), s[2], left=s[0]+s[1], color='r')
        >>> plt.title("Lengths of Strings")

        """
        #=================
        # Pure python algo
        #=================
        #alpha   = N.atleast_1d(alpha)
        #k       = alpha.size

        #if n == 1:
        #    val = N.zeros(k)
        #    for i in range(k):
        #        val[i]   = sgamma(alpha[i], n)
        #    val /= N.sum(val)
        #else:
        #    val = N.zeros((k, n))
        #    for i in range(k):
        #        val[i]   = sgamma(alpha[i], n)
        #    val /= N.sum(val, axis = 0)
        #    val = val.T

        #return val
        cdef cnp.npy_intp   k
        cdef cnp.npy_intp   totsize
        cdef cnp.ndarray    alpha_arr, val_arr
        cdef double     *alpha_data
        cdef double     *val_data
        cdef cnp.npy_intp   i, j
        cdef double     invacc, acc
        cdef cnp.broadcast  multi1, multi2

        alpha_arr = <cnp.ndarray>cnp.PyArray_FROM_OTF(alpha, cnp.NPY_DOUBLE, cnp.NPY_ARRAY_ALIGNED)
        if (alpha_arr.ndim != 1):
            raise ValueError("Parameter alpha is not a vector")

        k     = len(alpha)
        shape = _shape_from_size(size, k)

        diric    = self.standard_gamma(alpha_arr, shape)

        val_arr  = <cnp.ndarray>diric
        totsize = cnp.PyArray_SIZE(val_arr)

        # Use of iterators is faster than calling PyArray_ContiguousFromObject and iterating in C
        multi1 = cnp.PyArray_MultiIterNew(2, <void *>val_arr, <void *>alpha_arr)
        multi2 = cnp.PyArray_MultiIterNew(2, <void *>val_arr, <void *>alpha_arr)

        i = 0
        with self.lock, nogil:
            while i < totsize:
                acc = 0.0
                for j from 0 <= j < k:
                    val_data = <double*> cnp.PyArray_MultiIter_DATA(multi1, 0)
                    acc += val_data[0]
                    cnp.PyArray_MultiIter_NEXTi(multi1, 0)
                invacc = 1.0/acc
                for j from 0 <= j < k:
                    val_data = <double*> cnp.PyArray_MultiIter_DATA(multi2, 0)
                    val_data[0] *= invacc
                    cnp.PyArray_MultiIter_NEXTi(multi2, 0)
                i += k

        return diric

    # Shuffling and permutations:
    def shuffle(self, object x):
        """
        shuffle(x)

        Modify a sequence in-place by shuffling its contents.

        Parameters
        ----------
        x : array_like
            The array or list to be shuffled.

        Returns
        -------
        None

        Examples
        --------
        >>> arr = np.arange(10)
        >>> mkl_random.shuffle(arr)
        >>> arr
        [1 7 5 2 9 4 3 6 0 8]

        This function only shuffles the array along the first index of a
        multi-dimensional array:

        >>> arr = np.arange(9).reshape((3, 3))
        >>> mkl_random.shuffle(arr)
        >>> arr
        array([[3, 4, 5],
               [6, 7, 8],
               [0, 1, 2]])

        """
        cdef:
            cnp.npy_intp i, j, n = len(x), stride, itemsize
            char* x_ptr
            char* buf_ptr
            cdef cnp.ndarray u "arrayObject_u"
            cdef double *u_data

        if (n == 0):
            return

        u = <cnp.ndarray>self.random_sample(n-1)
        u_data = <double*>cnp.PyArray_DATA(u)

        if type(x) is np.ndarray and x.ndim == 1 and x.size:
            # Fast, statically typed path: shuffle the underlying buffer.
            # Only for non-empty, 1d objects of class ndarray (subclasses such
            # as MaskedArrays may not support this approach).
            x_ptr = <char*><size_t>x.ctypes.data
            stride = x.strides[0]
            itemsize = x.dtype.itemsize
            # As the array x could contain python objects we use a buffer
            # of bytes for the swaps to avoid leaving one of the objects
            # within the buffer and erroneously decrementing it's refcount
            # when the function exits.
            buf = np.empty(itemsize, dtype=np.int8) # GC'd at function exit
            buf_ptr = <char*><size_t>buf.ctypes.data
            with self.lock:
                # We trick gcc into providing a specialized implementation for
                # the most common case, yielding a ~33% performance improvement.
                # Note that apparently, only one branch can ever be specialized.
                if itemsize == sizeof(cnp.npy_intp):
                    self._shuffle_raw(n, sizeof(cnp.npy_intp), stride, x_ptr, buf_ptr, u_data)
                else:
                    self._shuffle_raw(n, itemsize, stride, x_ptr, buf_ptr, u_data)
        elif isinstance(x, np.ndarray) and x.ndim > 1 and x.size:
            # Multidimensional ndarrays require a bounce buffer.
            buf = np.empty_like(x[0])
            with self.lock:
                for i in reversed(range(1, n)):
                    j = <cnp.npy_intp>floor( (i + 1) * u_data[i - 1])
                    if (j < i):
                        buf[...] = x[j]
                        x[j] = x[i]
                        x[i] = buf
        else:
            # Untyped path.
            with self.lock:
                for i in reversed(range(1, n)):
                    j = <cnp.npy_intp>floor( (i + 1) * u_data[i - 1])
                    x[i], x[j] = x[j], x[i]

    cdef inline _shuffle_raw(self, cnp.npy_intp n, cnp.npy_intp itemsize,
                             cnp.npy_intp stride, char* data, char* buf, double* udata):
        cdef cnp.npy_intp i, j
        for i in reversed(range(1, n)):
            j = <cnp.npy_intp>floor( (i + 1) * udata[i - 1])
            memcpy(buf, data + j * stride, itemsize)
            memcpy(data + j * stride, data + i * stride, itemsize)
            memcpy(data + i * stride, buf, itemsize)

    def permutation(self, object x):
        """
        permutation(x)

        Randomly permute a sequence, or return a permuted range.

        If `x` is a multi-dimensional array, it is only shuffled along its
        first index.

        Parameters
        ----------
        x : int or array_like
            If `x` is an integer, randomly permute ``np.arange(x)``.
            If `x` is an array, make a copy and shuffle the elements
            randomly.

        Returns
        -------
        out : ndarray
            Permuted sequence or array range.

        Examples
        --------
        >>> mkl_random.permutation(10)
        array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6])

        >>> mkl_random.permutation([1, 4, 9, 12, 15])
        array([15,  1,  9,  4, 12])

        >>> arr = np.arange(9).reshape((3, 3))
        >>> mkl_random.permutation(arr)
        array([[6, 7, 8],
               [0, 1, 2],
               [3, 4, 5]])

        """
        if isinstance(x, (int, long, np.integer)):
            arr = np.arange(x)
        else:
            arr = np.array(x)
        self.shuffle(arr)
        return arr


def __RandomState_ctor():
    """Return a RandomState instance.
    This function exists solely to assist (un)pickling.
    Note that the state of the RandomState returned here is irrelevant, as this function's
    entire purpose is to return a newly allocated RandomState whose state pickle can set.
    Consequently the RandomState returned by this function is a freshly allocated copy
    with a seed=0.
    See https://github.com/numpy/numpy/issues/4763 for a detailed discussion
    """
    return RandomState(seed=0)

_rand = RandomState()
seed = _rand.seed
get_state = _rand.get_state
set_state = _rand.set_state
random_sample = _rand.random_sample
choice = _rand.choice
randint = _rand.randint
bytes = _rand.bytes
uniform = _rand.uniform
rand = _rand.rand
randn = _rand.randn
random_integers = _rand.random_integers
standard_normal = _rand.standard_normal
normal = _rand.normal
beta = _rand.beta
exponential = _rand.exponential
standard_exponential = _rand.standard_exponential
standard_gamma = _rand.standard_gamma
gamma = _rand.gamma
f = _rand.f
noncentral_f = _rand.noncentral_f
chisquare = _rand.chisquare
noncentral_chisquare = _rand.noncentral_chisquare
standard_cauchy = _rand.standard_cauchy
standard_t = _rand.standard_t
vonmises = _rand.vonmises
pareto = _rand.pareto
weibull = _rand.weibull
power = _rand.power
laplace = _rand.laplace
gumbel = _rand.gumbel
logistic = _rand.logistic
lognormal = _rand.lognormal
rayleigh = _rand.rayleigh
wald = _rand.wald
triangular = _rand.triangular

binomial = _rand.binomial
negative_binomial = _rand.negative_binomial
poisson = _rand.poisson
zipf = _rand.zipf
geometric = _rand.geometric
hypergeometric = _rand.hypergeometric
logseries = _rand.logseries

multivariate_normal = _rand.multivariate_normal
multinormal_cholesky = _rand.multinormal_cholesky
multinomial = _rand.multinomial
dirichlet = _rand.dirichlet

shuffle = _rand.shuffle
permutation = _rand.permutation
