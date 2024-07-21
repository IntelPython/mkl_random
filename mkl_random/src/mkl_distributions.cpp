/*
 Copyright (c) 2017-2024, Intel Corporation

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

#include <stddef.h> /* for nullptr */
#include <limits.h> /* for ULONG_MAX */
#include <assert.h>
#include <math.h>    /* fmod, fabs */
#include <cmath>     /* expm1 */
#include <algorithm> /* std::sort */

#include "mkl.h"
#include "mkl_vml.h"
#include "mkl_distributions.h"
#include "Python.h"
#include "numpy/npy_common.h" /* npy_intp */

#define MKL_INT_MAX ((npy_intp)(~((MKL_UINT)0) >> 1))

#if defined(__ICC) || defined(__INTEL_COMPILER)
#define DIST_PRAGMA_VECTOR _Pragma("vector")
#define DIST_PRAGMA_NOVECTOR _Pragma("novector")
#define DIST_ASSUME_ALIGNED(p, b) __assume_aligned((p), (b));
#elif defined(__clang__)
#define DIST_PRAGMA_VECTOR _Pragma("clang loop vectorize(enable)")
#define DIST_PRAGMA_NOVECTOR _Pragma("clang loop vectorize(disable)")
#define DIST_ASSUME_ALIGNED(p, b)
#elif defined(__GNUG__)
#define DIST_PRAGMA_VECTOR _Pragma("GCC ivdep")
#define DIST_PRAGMA_NOVECTOR
#define DIST_ASSUME_ALIGNED(p, b)
#else
#define DIST_PRAGMA_VECTOR
#define DIST_PRAGMA_NOVECTOR
#define DIST_ASSUME_ALIGNED(p, b)
#endif

void irk_double_vec(irk_state *state, npy_intp len, double *res)
{
    int err = 0;
    const double d_zero = 0.0, d_one = 1.0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, state->stream, MKL_INT_MAX, res, d_zero, d_one);
        assert(err == VSL_STATUS_OK);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, state->stream, len, res, d_zero, d_one);
    assert(err == VSL_STATUS_OK);
}

void irk_uniform_vec(irk_state *state, npy_intp len, double *res, const double low, const double high)
{
    int err = 0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, state->stream, MKL_INT_MAX, res, low, high);
        assert(err == VSL_STATUS_OK);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, state->stream, len, res, low, high);
    assert(err == VSL_STATUS_OK);
}

void irk_standard_normal_vec_ICDF(irk_state *state, npy_intp len, double *res)
{
    int err = 0;
    const double d_zero = 0.0, d_one = 1.0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, state->stream, MKL_INT_MAX, res, d_zero, d_one);
        assert(err == VSL_STATUS_OK);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, state->stream, len, res, d_zero, d_one);
    assert(err == VSL_STATUS_OK);
}

void irk_normal_vec_ICDF(irk_state *state, npy_intp len, double *res, const double loc, const double scale)
{
    int err = 0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, state->stream, MKL_INT_MAX, res, loc, scale);
        assert(err == VSL_STATUS_OK);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, state->stream, len, res, loc, scale);
    assert(err == VSL_STATUS_OK);
}

void irk_standard_normal_vec_BM1(irk_state *state, npy_intp len, double *res)
{
    int err = 0;
    const double d_zero = 0.0, d_one = 1.0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, state->stream, MKL_INT_MAX, res, d_zero, d_one);
        assert(err == VSL_STATUS_OK);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, state->stream, len, res, d_zero, d_one);
    assert(err == VSL_STATUS_OK);
}

void irk_normal_vec_BM1(irk_state *state, npy_intp len, double *res, const double loc, const double scale)
{
    int err = 0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, state->stream, MKL_INT_MAX, res, loc, scale);
        assert(err == VSL_STATUS_OK);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, state->stream, len, res, loc, scale);
    assert(err == VSL_STATUS_OK);
}

void irk_standard_normal_vec_BM2(irk_state *state, npy_intp len, double *res)
{
    int err = 0;
    const double d_zero = 0.0, d_one = 1.0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, state->stream, MKL_INT_MAX, res, d_zero, d_one);
        assert(err == VSL_STATUS_OK);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, state->stream, len, res, d_zero, d_one);
    assert(err == VSL_STATUS_OK);
}

void irk_normal_vec_BM2(irk_state *state, npy_intp len, double *res, const double loc, const double scale)
{
    int err = 0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, state->stream, MKL_INT_MAX, res, loc, scale);
        assert(err == VSL_STATUS_OK);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, state->stream, len, res, loc, scale);
    assert(err == VSL_STATUS_OK);
}

void irk_standard_exponential_vec(irk_state *state, npy_intp len, double *res)
{
    int err = 0;
    const double d_zero = 0.0, d_one = 1.0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        err = vdRngExponential(VSL_RNG_METHOD_EXPONENTIAL_ICDF_ACCURATE, state->stream, MKL_INT_MAX, res, d_zero, d_one);
        assert(err == VSL_STATUS_OK);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngExponential(VSL_RNG_METHOD_EXPONENTIAL_ICDF_ACCURATE, state->stream, len, res, d_zero, d_one);
    assert(err == VSL_STATUS_OK);
}

void irk_exponential_vec(irk_state *state, npy_intp len, double *res, const double scale)
{
    int err = 0;
    const double d_zero = 0.0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        err = vdRngExponential(VSL_RNG_METHOD_EXPONENTIAL_ICDF_ACCURATE, state->stream, MKL_INT_MAX, res, d_zero, scale);
        assert(err == VSL_STATUS_OK);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngExponential(VSL_RNG_METHOD_EXPONENTIAL_ICDF_ACCURATE, state->stream, len, res, d_zero, scale);
    assert(err == VSL_STATUS_OK);
}

void irk_standard_cauchy_vec(irk_state *state, npy_intp len, double *res)
{
    int err = 0;
    const double d_zero = 0.0, d_one = 1.0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        err = vdRngCauchy(VSL_RNG_METHOD_CAUCHY_ICDF, state->stream, MKL_INT_MAX, res, d_zero, d_one);
        assert(err == VSL_STATUS_OK);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngCauchy(VSL_RNG_METHOD_CAUCHY_ICDF, state->stream, len, res, d_zero, d_one);
    assert(err == VSL_STATUS_OK);
}

void irk_standard_gamma_vec(irk_state *state, npy_intp len, double *res, const double shape)
{
    int err = 0;
    const double d_zero = 0.0, d_one = 1.0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        err = vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, state->stream, MKL_INT_MAX, res, shape, d_zero, d_one);
        assert(err == VSL_STATUS_OK);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, state->stream, len, res, shape, d_zero, d_one);
    assert(err == VSL_STATUS_OK);
}

void irk_gamma_vec(irk_state *state, npy_intp len, double *res, const double shape, const double scale)
{
    int err = 0;
    const double d_zero = 0.0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        err = vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, state->stream, MKL_INT_MAX, res, shape, d_zero, scale);
        assert(err == VSL_STATUS_OK);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, state->stream, len, res, shape, d_zero, scale);
    assert(err == VSL_STATUS_OK);
}

/*  X ~ Z * (G*(2/df))**-0.5 */
void irk_standard_t_vec(irk_state *state, npy_intp len, double *res, const double df)
{
    int err = 0;
    const double d_zero = 0.0, d_one = 1.0;
    double shape = df / 2;
    double *sn = nullptr;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        irk_standard_t_vec(state, MKL_INT_MAX, res, df);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, state->stream, len, res, shape, d_zero, 1.0 / shape);
    assert(err == VSL_STATUS_OK);

    vmdInvSqrt(len, res, res, VML_HA);

    sn = (double *)mkl_malloc(len * sizeof(double), 64);
    assert(sn != nullptr);

    err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, state->stream, len, sn, d_zero, d_one);
    assert(err == VSL_STATUS_OK);

    vmdMul(len, res, sn, res, VML_HA);
    mkl_free(sn);
}

/* chisquare(df) ~ G(df/2, 2) */
void irk_chisquare_vec(irk_state *state, npy_intp len, double *res, const double df)
{
    int err = 0;
    const double d_zero = 0.0, d_two = 2.0;
    double shape = 0.5 * df;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        irk_chisquare_vec(state, MKL_INT_MAX, res, df);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, state->stream, len, res, shape, d_zero, d_two);
    assert(err == VSL_STATUS_OK);
}

/*    P ~ U^(-1/a) - 1 =  */
void irk_pareto_vec(irk_state *state, npy_intp len, double *res, const double alp)
{
    int err = 0;
    npy_intp i = 0;
    const double d_zero = 0.0, d_one = 1.0;
    double neg_rec_alp = -1.0 / alp;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        irk_pareto_vec(state, MKL_INT_MAX, res, alp);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, state->stream, len, res, d_zero, d_one);
    assert(err == VSL_STATUS_OK);

    /* res[i] = pow(res[i], neg_rec_alp) */
    vmdPowx(len, res, neg_rec_alp, res, VML_HA);

    DIST_PRAGMA_VECTOR
    for (i = 0; i < len; ++i)
        res[i] -= 1.0;
}

/*  W ~ E^(1/alp) */
void irk_weibull_vec(irk_state *state, npy_intp len, double *res, const double alp)
{
    int err = 0;
    const double d_zero = 0.0, d_one = 1.0;
    double rec_alp = 1.0 / alp;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        irk_weibull_vec(state, MKL_INT_MAX, res, alp);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngExponential(VSL_RNG_METHOD_EXPONENTIAL_ICDF_ACCURATE, state->stream, len, res, d_zero, d_one);
    assert(err == VSL_STATUS_OK);

    vmdPowx(len, res, rec_alp, res, VML_HA);
}

/*  pow(1 - exp(-E(1))), 1./a) == pow(U, 1./a) */
void irk_power_vec(irk_state *state, npy_intp len, double *res, const double alp)
{
    int err = 0;
    const double d_zero = 0.0, d_one = 1.0;
    double rec_alp = 1.0 / alp;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        irk_power_vec(state, MKL_INT_MAX, res, alp);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, state->stream, len, res, d_zero, d_one);
    assert(err == VSL_STATUS_OK);

    /* res[i] = pow(res[i], rec_alp) */
    vmdPowx(len, res, rec_alp, res, VML_HA);
}

/*  scale * sqrt(2.0 * E(1))  */
void irk_rayleigh_vec(irk_state *state, npy_intp len, double *res, const double scale)
{
    int err = 0;
    npy_intp i = 0;
    const double d_zero = 0.0, d_two = 2.0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        irk_rayleigh_vec(state, MKL_INT_MAX, res, scale);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngExponential(VSL_RNG_METHOD_EXPONENTIAL_ICDF_ACCURATE, state->stream, len, res, d_zero, d_two);
    assert(err == VSL_STATUS_OK);

    vmdSqrt(len, res, res, VML_HA);

    DIST_PRAGMA_VECTOR
    for (i = 0; i < len; ++i)
        res[i] *= scale;
}

void irk_beta_vec(irk_state *state, npy_intp len, double *res, const double a, const double b)
{
    int err = 0;
    const double d_zero = 0.0, d_one = 1.0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        err = vdRngBeta(VSL_RNG_METHOD_BETA_CJA_ACCURATE, state->stream, MKL_INT_MAX, res, a, b, d_zero, d_one);
        assert(err == VSL_STATUS_OK);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngBeta(VSL_RNG_METHOD_BETA_CJA_ACCURATE, state->stream, len, res, a, b, d_zero, d_one);
    assert(err == VSL_STATUS_OK);
}

/*  F(df_num, df_den) ~ G( df_num/2, 2/df_num) / G(df_den/2, 2/df_den))  */
void irk_f_vec(irk_state *state, npy_intp len, double *res, const double df_num, const double df_den)
{
    int err = 0;
    const double d_zero = 0.0;
    double shape = 0.5 * df_num, scale = 2.0 / df_num;
    double *den = nullptr;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        irk_f_vec(state, MKL_INT_MAX, res, df_num, df_den);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, state->stream, len, res, shape, d_zero, scale);
    assert(err == VSL_STATUS_OK);

    den = (double *)mkl_malloc(len * sizeof(double), 64);
    assert(den != nullptr);

    shape = 0.5 * df_den;
    scale = 2.0 / df_den;
    err = vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, state->stream, len, den, shape, d_zero, scale);
    assert(err == VSL_STATUS_OK);

    vmdDiv(len, res, den, res, VML_HA);
    mkl_free(den);
}

/*
   for df > 1, X ~ Chi2(df - 1) + ( sqrt(nonc) + Z)^2
   for df <=1, X ~ Chi2( df + 2*I), where I ~ Poisson( nonc/2.0)
*/
void irk_noncentral_chisquare_vec(irk_state *state, npy_intp len, double *res, const double df, const double nonc)
{
    int err = 0;
    npy_intp i = 0;
    const double d_zero = 0.0, d_one = 1.0, d_two = 2.0;
    double shape, loc;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        irk_noncentral_chisquare_vec(state, MKL_INT_MAX, res, df, nonc);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    if (df > 1)
    {
        double *nvec;

        shape = 0.5 * (df - 1.0);
        /* res has chi^2 with (df - 1) */
        err = vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, state->stream, len, res, shape, d_zero, d_two);

        nvec = (double *)mkl_malloc(len * sizeof(double), 64);
        assert(nvec != nullptr);

        loc = sqrt(nonc);
        err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, state->stream, len, nvec, loc, d_one);
        assert(err == VSL_STATUS_OK);

        /* squaring could result in an overflow */
        vmdSqr(len, nvec, nvec, VML_HA);
        vmdAdd(len, res, nvec, res, VML_HA);

        mkl_free(nvec);
    }
    else
    {
        if (df == 0.)
        {
            return irk_chisquare_vec(state, len, res, df);
        }
        if (df < 1)
        {
            /* noncentral_chisquare(df, nonc) ~ G( df/2 + Poisson(nonc/2), 2) */
            double lambda;
            int *pvec = (int *)mkl_malloc(len * sizeof(int), 64);

            assert(pvec != nullptr);

            lambda = 0.5 * nonc;
            err = viRngPoisson(VSL_RNG_METHOD_POISSON_PTPE, state->stream, len, pvec, lambda);
            assert(err == VSL_STATUS_OK);

            shape = 0.5 * df;

            if (0.125 * len > sqrt(lambda))
            {
                int *idx = nullptr;
                double *tmp = nullptr;

                idx = (int *)mkl_malloc(len * sizeof(int), 64);
                assert(idx != nullptr);

                DIST_PRAGMA_VECTOR
                for (i = 0; i < len; ++i)
                    idx[i] = (int)i;

                std::sort(idx, idx + len, [pvec](int i1, int i2)
                          { return pvec[i1] < pvec[i2]; });
                /* idx now contains original indexes of ordered Poisson outputs */

                /* allocate workspace to store samples of gamma, enough to hold entire output */
                tmp = (double *)mkl_malloc(len * sizeof(double), 64);
                assert(tmp != nullptr);

                for (i = 0; i < len;)
                {
                    int cv = pvec[idx[i]];
                    npy_intp k = 0, j = 0;

                    for (j = i + 1; (j < len) && (pvec[idx[j]] == cv); ++j)
                    {
                    }

                    assert(j > i);
                    err = vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, state->stream, j - i, tmp,
                                     shape + cv, d_zero, d_two);
                    assert(err == VSL_STATUS_OK);

                    DIST_PRAGMA_VECTOR
                    for (k = 0; k < j - i; ++k)
                        res[idx[k + i]] = tmp[k];

                    i = j;
                }

                mkl_free(tmp);
                mkl_free(idx);
            }
            else
            {

                for (i = 0; i < len; ++i)
                {
                    err = vdRngGamma(VSL_RNG_METHOD_GAMMA_GNORM_ACCURATE, state->stream, 1,
                                     res + i, shape + pvec[i], d_zero, d_two);
                    assert(err == VSL_STATUS_OK);
                }
            }

            mkl_free(pvec);
        }
        else
        {
            /* noncentral_chisquare(1, nonc) ~ (Z + sqrt(nonc))**2 for df == 1 */
            loc = sqrt(nonc);
            err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, state->stream, len, res, loc, d_one);
            assert(err == VSL_STATUS_OK);
            /* squaring could result in an overflow */
            vmdSqr(len, res, res, VML_HA);
        }
    }
}

void irk_laplace_vec(irk_state *state, npy_intp len, double *res, const double loc, const double scale)
{
    int err = 0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        err = vdRngLaplace(VSL_RNG_METHOD_LAPLACE_ICDF, state->stream, MKL_INT_MAX, res, loc, scale);
        assert(err == VSL_STATUS_OK);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngLaplace(VSL_RNG_METHOD_LAPLACE_ICDF, state->stream, len, res, loc, scale);
    assert(err == VSL_STATUS_OK);
}

void irk_gumbel_vec(irk_state *state, npy_intp len, double *res, const double loc, const double scale)
{
    int err = 0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        err = vdRngGumbel(VSL_RNG_METHOD_GUMBEL_ICDF, state->stream, MKL_INT_MAX, res, loc, scale);
        assert(err == VSL_STATUS_OK);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngGumbel(VSL_RNG_METHOD_GUMBEL_ICDF, state->stream, len, res, loc, scale);
    assert(err == VSL_STATUS_OK);
}

/*   Logistic(loc, scale) ~ loc + scale * log(u/(1.0 - u)) */
void irk_logistic_vec(irk_state *state, npy_intp len, double *res, const double loc, const double scale)
{
    int err = 0;
    npy_intp i = 0;
    const double d_one = 1.0, d_zero = 0.0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        irk_logistic_vec(state, MKL_INT_MAX, res, loc, scale);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, state->stream, len, res, d_zero, d_one);
    assert(err == VSL_STATUS_OK);

    /* can MKL optimize computation of the logit function  p \mapsto \ln(p/(1-p)) */
    DIST_PRAGMA_VECTOR
    for (i = 0; i < len; ++i)
        res[i] = log(res[i] / (1.0 - res[i]));

    DIST_PRAGMA_VECTOR
    for (i = 0; i < len; ++i)
        res[i] = loc + scale * res[i];
}

void irk_lognormal_vec_ICDF(irk_state *state, npy_intp len, double *res, const double mean, const double sigma)
{
    int err = 0;
    const double d_zero = 0.0, d_one = 1.0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        err = vdRngLognormal(VSL_RNG_METHOD_LOGNORMAL_ICDF_ACCURATE, state->stream, MKL_INT_MAX, res, mean, sigma, d_zero, d_one);
        assert(err == VSL_STATUS_OK);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngLognormal(VSL_RNG_METHOD_LOGNORMAL_ICDF_ACCURATE, state->stream, len, res, mean, sigma, d_zero, d_one);
    assert(err == VSL_STATUS_OK);
}

void irk_lognormal_vec_BM(irk_state *state, npy_intp len, double *res, const double mean, const double sigma)
{
    int err = 0;
    const double d_zero = 0.0, d_one = 1.0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        err = vdRngLognormal(VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2_ACCURATE, state->stream, MKL_INT_MAX, res, mean, sigma, d_zero, d_one);
        assert(err == VSL_STATUS_OK);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngLognormal(VSL_RNG_METHOD_LOGNORMAL_BOXMULLER2_ACCURATE, state->stream, len, res, mean, sigma, d_zero, d_one);
    assert(err == VSL_STATUS_OK);
}

/* direct transformation method */
void irk_wald_vec(irk_state *state, npy_intp len, double *res, const double mean, const double scale)
{
    int err = 0;
    npy_intp i = 0;
    const double d_zero = 0., d_one = 1.0;
    double *uvec = nullptr;
    double gsc = sqrt(0.5 * mean / scale);

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        irk_wald_vec(state, MKL_INT_MAX, res, mean, scale);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, state->stream, len, res, d_zero, gsc);
    assert(err == VSL_STATUS_OK);

    /* Y = mean/(2 scale) * Z^2 */
    vmdSqr(len, res, res, VML_HA);

    DIST_PRAGMA_VECTOR
    for (i = 0; i < len; ++i)
    {
        if (res[i] <= 2.0)
        {
            res[i] = 1.0 + res[i] + sqrt(res[i] * (res[i] + 2.0));
        }
        else
        {
            res[i] = 1.0 + res[i] * (1.0 + sqrt(1.0 + 2.0 / res[i]));
        }
    }

    uvec = (double *)mkl_malloc(len * sizeof(double), 64);
    assert(uvec != nullptr);

    err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, state->stream, len, uvec, d_zero, d_one);
    assert(err == VSL_STATUS_OK);

    DIST_PRAGMA_VECTOR
    for (i = 0; i < len; ++i)
    {
        if (uvec[i] * (1.0 + res[i]) <= res[i])
            res[i] = mean / res[i];
        else
            res[i] = mean * res[i];
    }

    mkl_free(uvec);
}

#ifndef M_PI
/*  128-bits worth of pi */
#define M_PI 3.141592653589793238462643383279502884197
#endif

/* Uses the rejection algorithm compared against the wrapped Cauchy
   distribution suggested by Best and Fisher and documented in
   Chapter 9 of Luc's Non-Uniform Random Variate Generation.
   http://cg.scs.carleton.ca/~luc/rnbookindex.html
   (but corrected to match the algorithm in R and Python)
*/
static void
irk_vonmises_vec_small_kappa(irk_state *state, npy_intp len, double *res, const double mu, const double kappa)
{
    int err = 0;
    npy_intp n = 0, i = 0, size = 0;
    double rho_over_kappa, rho, r, s_kappa, Z, W, Y, V;
    double *Uvec = nullptr, *Vvec = nullptr;
    float *VFvec = nullptr;
    const double d_zero = 0.0, d_one = 1.0;

    assert(0. < kappa <= 1.0);

    r = 1 + sqrt(1 + 4 * kappa * kappa);
    rho_over_kappa = (2) / (r + sqrt(2 * r));
    rho = rho_over_kappa * kappa;

    /* s times kappa */
    s_kappa = (1 + rho * rho) / (2 * rho_over_kappa);

    Uvec = (double *)mkl_malloc(len * sizeof(double), 64);
    assert(Uvec != nullptr);
    Vvec = (double *)mkl_malloc(len * sizeof(double), 64);
    assert(Vvec != nullptr);

    for (n = 0; n < len;)
    {
        size = len - n;
        err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, size, Uvec, d_zero, M_PI);
        assert(err == VSL_STATUS_OK);
        err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, state->stream, size, Vvec, d_zero, d_one);
        assert(err == VSL_STATUS_OK);

        for (i = 0; i < size; ++i)
        {
            Z = cos(Uvec[i]);
            V = Vvec[i];
            W = (kappa + s_kappa * Z) / (s_kappa + kappa * Z);
            Y = s_kappa - kappa * W;
            if ((Y * (2 - Y) >= V) || (log(Y / V) + 1 >= Y))
            {
                res[n++] = acos(W);
            }
        }
    }

    mkl_free(Uvec);

    VFvec = (float *)Vvec;
    err = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, VFvec, (float)d_zero, (float)d_one);
    assert(err == VSL_STATUS_OK);

    DIST_PRAGMA_VECTOR
    for (i = 0; i < len; ++i)
    {
        double mod, resi;

        resi = (VFvec[i] < 0.5) ? mu - res[i] : mu + res[i];
        mod = fabs(resi);
        mod = (fmod(mod + M_PI, 2 * M_PI) - M_PI);
        res[i] = (resi < 0) ? -mod : mod;
    }

    mkl_free(Vvec);
}

static void
irk_vonmises_vec_large_kappa(irk_state *state, npy_intp len, double *res, const double mu, const double kappa)
{
    int err = 0;
    npy_intp i = 0, n = 0, size = 0;
    double r_over_two_kappa, recip_two_kappa;
    double s_minus_one, hpt, r_over_two_kappa_minus_one, rho_minus_one;
    double *Uvec = nullptr, *Vvec = nullptr;
    float *VFvec = nullptr;
    const double d_zero = 0.0, d_one = 1.0;

    assert(kappa > 1.0);

    recip_two_kappa = 1 / (2 * kappa);

    /* variables here are dwindling to zero as kappa grows */
    hpt = sqrt(1 + recip_two_kappa * recip_two_kappa);
    r_over_two_kappa_minus_one = recip_two_kappa * (1 + recip_two_kappa / (1 + hpt));
    r_over_two_kappa = 1 + r_over_two_kappa_minus_one;
    rho_minus_one = r_over_two_kappa_minus_one - sqrt(2 * r_over_two_kappa * recip_two_kappa);
    s_minus_one = rho_minus_one * (0.5 * rho_minus_one / (1 + rho_minus_one));

    Uvec = (double *)mkl_malloc(len * sizeof(double), 64);
    assert(Uvec != nullptr);
    Vvec = (double *)mkl_malloc(len * sizeof(double), 64);
    assert(Vvec != nullptr);

    for (n = 0; n < len;)
    {
        size = len - n;
        err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, size, Uvec, d_zero, 0.5 * M_PI);
        assert(err == VSL_STATUS_OK);
        err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, state->stream, size, Vvec, d_zero, d_one);
        assert(err == VSL_STATUS_OK);

        for (i = 0; i < size; ++i)
        {
            double sn, cn, sn2, cn2;
            double neg_W_minus_one, V, Y;

            sn = sin(Uvec[i]);
            cn = cos(Uvec[i]);
            V = Vvec[i];
            sn2 = sn * sn;
            cn2 = cn * cn;

            neg_W_minus_one = s_minus_one * sn2 / (0.5 * s_minus_one + cn2);
            Y = kappa * (s_minus_one + neg_W_minus_one);

            if ((Y * (2 - Y) >= V) || (log(Y / V) + 1 >= Y))
            {
                Y = neg_W_minus_one * (2 - neg_W_minus_one);
                if (Y < 0)
                    Y = 0.;
                else if (Y > 1.0)
                    Y = 1.0;

                res[n++] = asin(sqrt(Y));
            }
        }
    }

    mkl_free(Uvec);

    VFvec = (float *)Vvec;
    err = vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, VFvec, (float)d_zero, (float)d_one);
    assert(err == VSL_STATUS_OK);

    DIST_PRAGMA_VECTOR
    for (i = 0; i < len; ++i)
    {
        double mod, resi;

        resi = (VFvec[i] < 0.5) ? mu - res[i] : mu + res[i];
        mod = fabs(resi);
        mod = (fmod(mod + M_PI, 2 * M_PI) - M_PI);
        res[i] = (resi < 0) ? -mod : mod;
    }

    mkl_free(Vvec);
}

void irk_vonmises_vec(irk_state *state, npy_intp len, double *res, const double mu, const double kappa)
{
    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        irk_vonmises_vec(state, MKL_INT_MAX, res, mu, kappa);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    if (kappa > 1.0)
        irk_vonmises_vec_large_kappa(state, len, res, mu, kappa);
    else
        irk_vonmises_vec_small_kappa(state, len, res, mu, kappa);
}

void irk_noncentral_f_vec(irk_state *state, npy_intp len, double *res, const double df_num, const double df_den, const double nonc)
{
    npy_intp i;
    double *den = nullptr, fctr;

    if (len < 1)
        return;

    if (nonc == 0.)
        return irk_f_vec(state, len, res, df_num, df_den);

    while (len > MKL_INT_MAX)
    {
        irk_noncentral_f_vec(state, MKL_INT_MAX, res, df_num, df_den, nonc);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    irk_noncentral_chisquare_vec(state, len, res, df_num, nonc);

    den = (double *)mkl_malloc(len * sizeof(double), 64);

    if (den == nullptr)
        return;

    irk_noncentral_chisquare_vec(state, len, den, df_den, nonc);

    vmdDiv(len, res, den, res, VML_HA);

    mkl_free(den);
    fctr = df_den / df_num;

    DIST_PRAGMA_VECTOR
    for (i = 0; i < len; ++i)
        res[i] *= fctr;
}

void irk_triangular_vec(irk_state *state, npy_intp len, double *res, const double x_min, const double x_mode, const double x_max)
{
    int err = 0;
    npy_intp i = 0;
    const double d_zero = 0.0, d_one = 1.0;
    double ratio, lpr, rpr;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        irk_triangular_vec(state, MKL_INT_MAX, res, x_min, x_mode, x_max);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, state->stream, len, res, d_zero, d_one);
    assert(err == VSL_STATUS_OK);

    {
        double wtot, wl, wr;

        wtot = x_max - x_min;
        wl = x_mode - x_min;
        wr = x_max - x_mode;

        ratio = wl / wtot;
        lpr = wl * wtot;
        rpr = wr * wtot;
    }

    assert(0 <= ratio && ratio <= 1);

    if (ratio <= 0)
    {
        DIST_PRAGMA_VECTOR
        for (i = 0; i < len; ++i)
        {
            /* U and 1 - U are equal in distribution */
            res[i] = x_max - sqrt(res[i] * rpr);
        }
    }
    else if (ratio >= 1)
    {
        DIST_PRAGMA_VECTOR
        for (i = 0; i < len; ++i)
        {
            res[i] = x_min + sqrt(res[i] * lpr);
        }
    }
    else
    {
        DIST_PRAGMA_VECTOR
        for (i = 0; i < len; ++i)
        {
            double ui = res[i];
            res[i] = (ui > ratio) ? x_max - sqrt((1.0 - ui) * rpr) : x_min + sqrt(ui * lpr);
        }
    }
}

void irk_binomial_vec(irk_state *state, npy_intp len, int *res, const int n, const double p)
{
    int err = 0;

    if (len < 1)
        return;

    if (n == 0)
    {
        memset(res, 0, len * sizeof(int));
    }
    else
    {
        while (len > MKL_INT_MAX)
        {
            err = viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, state->stream, MKL_INT_MAX, res, n, p);
            assert(err == VSL_STATUS_OK);
            res += MKL_INT_MAX;
            len -= MKL_INT_MAX;
        }

        err = viRngBinomial(VSL_RNG_METHOD_BINOMIAL_BTPE, state->stream, len, res, n, p);
        assert(err == VSL_STATUS_OK);
    }
}

void irk_multinomial_vec(irk_state *state, npy_intp len, int *res, const int n, const int k, const double *pvec)
{
    int err = 0;

    if (len < 1)
        return;

    if (n == 0)
    {
        memset(res, 0, len * k * sizeof(int));
    }
    else
    {
        while (len > MKL_INT_MAX)
        {
            err = viRngMultinomial(VSL_RNG_METHOD_MULTINOMIAL_MULTPOISSON, state->stream, MKL_INT_MAX, res, n, k, pvec);
            assert(err == VSL_STATUS_OK);
            res += k * MKL_INT_MAX;
            len -= k * MKL_INT_MAX;
        }

        err = viRngMultinomial(VSL_RNG_METHOD_MULTINOMIAL_MULTPOISSON, state->stream, len, res, n, k, pvec);
        assert(err == VSL_STATUS_OK);
    }
}

void irk_geometric_vec(irk_state *state, npy_intp len, int *res, const double p)
{
    int err = 0;

    if (len < 1)
        return;

    if ((0.0 < p) && (p < 1.0))
    {
        while (len > MKL_INT_MAX)
        {
            err = viRngGeometric(VSL_RNG_METHOD_GEOMETRIC_ICDF, state->stream, MKL_INT_MAX, res, p);
            assert(err == VSL_STATUS_OK);

            res += MKL_INT_MAX;
            len -= MKL_INT_MAX;
        }

        err = viRngGeometric(VSL_RNG_METHOD_GEOMETRIC_ICDF, state->stream, len, res, p);
        assert(err == VSL_STATUS_OK);
    }
    else
    {
        if (p == 1.0)
        {
            npy_intp i;
            for (i = 0; i < len; ++i)
                res[i] = 0;
        }
        else
        {
            assert(p >= 0.0);
            assert(p <= 1.0);
        }
    }
}

void irk_negbinomial_vec(irk_state *state, npy_intp len, int *res, const double a, const double p)
{
    int err = 0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        err = viRngNegbinomial(VSL_RNG_METHOD_NEGBINOMIAL_NBAR, state->stream, MKL_INT_MAX, res, a, p);
        assert(err == VSL_STATUS_OK);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = viRngNegbinomial(VSL_RNG_METHOD_NEGBINOMIAL_NBAR, state->stream, len, res, a, p);
    assert(err == VSL_STATUS_OK);
}

void irk_hypergeometric_vec(irk_state *state, npy_intp len, int *res, const int lot_s,
                            const int sampling_s, const int marked_s)
{
    int err = 0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        err = viRngHypergeometric(VSL_RNG_METHOD_HYPERGEOMETRIC_H2PE, state->stream, MKL_INT_MAX, res,
                                  lot_s, sampling_s, marked_s);
        assert(err == VSL_STATUS_OK);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = viRngHypergeometric(VSL_RNG_METHOD_HYPERGEOMETRIC_H2PE, state->stream, len, res,
                              lot_s, sampling_s, marked_s);
    assert(err == VSL_STATUS_OK);
}

void irk_poisson_vec_PTPE(irk_state *state, npy_intp len, int *res, const double lambda)
{
    int err = 0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        err = viRngPoisson(VSL_RNG_METHOD_POISSON_PTPE, state->stream, MKL_INT_MAX, res, lambda);
        assert(err == VSL_STATUS_OK);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = viRngPoisson(VSL_RNG_METHOD_POISSON_PTPE, state->stream, len, res, lambda);
    assert(err == VSL_STATUS_OK);
}

void irk_poisson_vec_POISNORM(irk_state *state, npy_intp len, int *res, const double lambda)
{
    int err = 0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        err = viRngPoisson(VSL_RNG_METHOD_POISSON_POISNORM, state->stream, MKL_INT_MAX, res, lambda);
        assert(err == VSL_STATUS_OK);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = viRngPoisson(VSL_RNG_METHOD_POISSON_POISNORM, state->stream, len, res, lambda);
    assert(err == VSL_STATUS_OK);
}

void irk_poisson_vec_V(irk_state *state, npy_intp len, int *res, double *lambdas)
{
    int err = 0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        err = viRngPoissonV(VSL_RNG_METHOD_POISSONV_POISNORM, state->stream, MKL_INT_MAX, res, lambdas);
        assert(err == VSL_STATUS_OK);

        res += MKL_INT_MAX;
        lambdas += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = viRngPoissonV(VSL_RNG_METHOD_POISSONV_POISNORM, state->stream, len, res, lambdas);
    assert(err == VSL_STATUS_OK);
}

void irk_zipf_long_vec(irk_state *state, npy_intp len, long *res, const double a)
{
    int err = 0;
    npy_intp i = 0, n_accepted = 0, batch_size = 0;
    double T, U, V, am1, b;
    double *Uvec = nullptr, *Vvec = nullptr;
    long X;
    const double d_zero = 0.0, d_one = 1.0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        irk_zipf_long_vec(state, MKL_INT_MAX, res, a);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    am1 = a - d_one;
    b = pow(2.0, am1);

    Uvec = (double *)mkl_malloc(len * sizeof(double), 64);
    assert(Uvec != nullptr);
    Vvec = (double *)mkl_malloc(len * sizeof(double), 64);
    assert(Vvec != nullptr);

    for (n_accepted = 0; n_accepted < len;)
    {
        batch_size = len - n_accepted;
        err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, state->stream, batch_size, Uvec, d_zero, d_one);
        assert(err == VSL_STATUS_OK);
        err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, batch_size, Vvec, d_zero, d_one);
        assert(err == VSL_STATUS_OK);

        DIST_PRAGMA_VECTOR
        for (i = 0; i < batch_size; ++i)
        {
            U = d_one - Uvec[i];
            V = Vvec[i];
            X = (long)floor(pow(U, (-1.0) / am1));
            /* The real result may be above what can be represented in a signed
             * long. It will get casted to -sys.maxint-1. Since this is
             * a straightforward rejection algorithm, we can just reject this value
             * in the rejection condition below. This function then models a Zipf
             * distribution truncated to sys.maxint.
             */
            T = pow(d_one + d_one / X, am1);
            if ((X > 0) && ((V * X) * (T - d_one) / (b - d_one) <= T / b))
            {
                res[n_accepted++] = X;
            }
        }
    }

    mkl_free(Vvec);
    mkl_free(Uvec);
}

void irk_logseries_vec(irk_state *state, npy_intp len, int *res, const double theta)
{
    int err = 0;
    npy_intp i = 0, n_accepted = 0, batch_size = 0;
    double q, r, V;
    double *Uvec = nullptr, *Vvec = nullptr;
    int result;
    const double d_zero = 0.0, d_one = 1.0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        irk_logseries_vec(state, MKL_INT_MAX, res, theta);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    r = log(d_one - theta);

    Uvec = (double *)mkl_malloc(len * sizeof(double), 64);
    assert(Uvec != nullptr);
    Vvec = (double *)mkl_malloc(len * sizeof(double), 64);
    assert(Vvec != nullptr);

    for (n_accepted = 0; n_accepted < len;)
    {
        batch_size = len - n_accepted;
        err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, batch_size, Uvec, d_zero, d_one);
        assert(err == VSL_STATUS_OK);
        err = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, state->stream, batch_size, Vvec, d_zero, d_one);
        assert(err == VSL_STATUS_OK);

        DIST_PRAGMA_VECTOR
        for (i = 0; i < batch_size; ++i)
        {
            V = Vvec[i];
            if (V >= theta)
            {
                res[n_accepted++] = 1;
            }
            else
            {
#if __cplusplus > 199711L
                q = -expm1(r * Uvec[i]);
#else
                /*  exp(x) - 1 == 2 * exp(x/2) * sinh(x/2)  */
                q = r * Uvec[i];
                if (q > 1.)
                {
                    q = 1.0 - exp(q);
                }
                else
                {
                    q = 0.5 * q;
                    q = -2.0 * exp(q) * sinh(q);
                }
#endif
                if (V <= q * q)
                {
                    result = (int)floor(1 + log(V) / log(q));
                    if (result > 0)
                    {
                        res[n_accepted++] = result;
                    }
                }
                else
                {
                    res[n_accepted++] = (V < q) ? 2 : 1;
                }
            }
        }
    }

    mkl_free(Vvec);
}

/* samples discrete uniforms from [low, high) */
void irk_discrete_uniform_vec(irk_state *state, npy_intp len, int *res, const int low, const int high)
{
    int err = 0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        err = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, MKL_INT_MAX, res, low, high);
        assert(err == VSL_STATUS_OK);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    err = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, res, low, high);
    assert(err == VSL_STATUS_OK);
}

void irk_discrete_uniform_long_vec(irk_state *state, npy_intp len, long *res, const long low, const long high)
{
    int err = 0;
    unsigned long max;
    npy_intp i = 0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        irk_discrete_uniform_long_vec(state, MKL_INT_MAX, res, low, high);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    max = ((unsigned long)high) - ((unsigned long)low) - 1UL;
    if (max == 0)
    {
        DIST_PRAGMA_VECTOR
        for (i = 0; i < len; ++i)
            res[i] = low;

        return;
    }

    if (max <= (unsigned long)INT_MAX)
    {
        int *buf = (int *)mkl_malloc(len * sizeof(int), 64);
        assert(buf != nullptr);

        err = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, buf, -1, (int)max);
        assert(err == VSL_STATUS_OK);

        DIST_PRAGMA_VECTOR
        for (i = 0; i < len; ++i)
            res[i] = low + ((long)buf[i]) + 1L;

        mkl_free(buf);
    }
    else
    {
        unsigned long mask = max;
        unsigned long *buf = nullptr;
        int n_accepted;

        /* Smallest bit mask >= max */
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
#if ULONG_MAX > 0xffffffffUL
        mask |= mask >> 32;
#endif

        buf = (unsigned long *)mkl_malloc(len * sizeof(long), 64);
        assert(buf != nullptr);
        n_accepted = 0;

        while (n_accepted < len)
        {
            int k, batchSize = len - n_accepted;

            err = viRngUniformBits64(VSL_RNG_METHOD_UNIFORM_STD, state->stream, batchSize, (unsigned MKL_INT64 *)buf);
            assert(err == VSL_STATUS_OK);

            for (k = 0; k < batchSize; ++k)
            {
                unsigned long value = buf[k] & mask;
                if (value <= max)
                {
                    res[n_accepted++] = low + value;
                }
            }
        }

        mkl_free(buf);
    }
}

void irk_ulong_vec(irk_state *state, npy_intp len, unsigned long *res)
{
    int err = 0;

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        irk_ulong_vec(state, MKL_INT_MAX, res);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

#if ULONG_MAX <= 0xffffffffUL
    err = viRngUniformBits32(VSL_RNG_METHOD_UNIFORMBITS32_STD, state->stream, len, (unsigned int *)res);
#else
    err = viRngUniformBits64(VSL_RNG_METHOD_UNIFORMBITS64_STD, state->stream, len, (unsigned MKL_INT64 *)res);
#endif

    assert(err == VSL_STATUS_OK);
}

void irk_long_vec(irk_state *state, npy_intp len, long *res)
{
    npy_intp i = 0;
    unsigned long *ulptr = (unsigned long *)res;

    irk_ulong_vec(state, len, ulptr);

    DIST_PRAGMA_VECTOR
    for (i = 0; i < len; ++i)
        res[i] = (long)(ulptr[i] >> 1);
}

void irk_rand_bool_vec(irk_state *state, npy_intp len, npy_bool *res, const npy_bool lo, const npy_bool hi)
{
    int err = 0;
    npy_intp i = 0;
    int *buf = nullptr;

    if (len < 1)
        return;

    if (len > MKL_INT_MAX)
    {
        irk_rand_bool_vec(state, MKL_INT_MAX, res, lo, hi);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    if (lo == hi)
    {
        DIST_PRAGMA_VECTOR
        for (i = 0; i < len; ++i)
            res[i] = lo;

        return;
    }

    assert((lo == 0) && (hi == 1));
    buf = (int *)mkl_malloc(len * sizeof(int), 64);
    assert(buf != nullptr);

    err = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, buf, (int)lo, (int)hi + 1);
    assert(err == VSL_STATUS_OK);

    DIST_PRAGMA_VECTOR
    for (i = 0; i < len; ++i)
        res[i] = (npy_bool)buf[i];

    mkl_free(buf);
}

void irk_rand_uint8_vec(irk_state *state, npy_intp len, npy_uint8 *res, const npy_uint8 lo, const npy_uint8 hi)
{
    int err = 0;
    npy_intp i = 0;
    int *buf = nullptr;

    if (len < 1)
        return;

    if (len > MKL_INT_MAX)
    {
        irk_rand_uint8_vec(state, MKL_INT_MAX, res, lo, hi);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    if (lo == hi)
    {
        DIST_PRAGMA_VECTOR
        for (i = 0; i < len; ++i)
            res[i] = lo;

        return;
    }

    assert(lo < hi);
    buf = (int *)mkl_malloc(len * sizeof(int), 64);
    assert(buf != nullptr);

    err = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, buf, (int)lo, (int)hi + 1);
    assert(err == VSL_STATUS_OK);

    DIST_PRAGMA_VECTOR
    for (i = 0; i < len; ++i)
        res[i] = (npy_uint8)buf[i];

    mkl_free(buf);
}

void irk_rand_int8_vec(irk_state *state, npy_intp len, npy_int8 *res, const npy_int8 lo, const npy_int8 hi)
{
    int err = 0;
    npy_intp i = 0;
    int *buf = nullptr;

    if (len < 1)
        return;

    if (len > MKL_INT_MAX)
    {
        irk_rand_int8_vec(state, MKL_INT_MAX, res, lo, hi);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    if (lo == hi)
    {
        DIST_PRAGMA_VECTOR
        for (i = 0; i < len; ++i)
            res[i] = lo;

        return;
    }

    assert(lo < hi);
    buf = (int *)mkl_malloc(len * sizeof(int), 64);
    assert(buf != nullptr);

    err = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, buf, (int)lo, (int)hi + 1);
    assert(err == VSL_STATUS_OK);

    DIST_PRAGMA_VECTOR
    for (i = 0; i < len; ++i)
        res[i] = (npy_int8)buf[i];

    mkl_free(buf);
}

void irk_rand_uint16_vec(irk_state *state, npy_intp len, npy_uint16 *res, const npy_uint16 lo, const npy_uint16 hi)
{
    int err = 0;
    npy_intp i = 0;
    int *buf = nullptr;

    if (len < 1)
        return;

    if (len > MKL_INT_MAX)
    {
        irk_rand_uint16_vec(state, MKL_INT_MAX, res, lo, hi);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    if (lo == hi)
    {
        DIST_PRAGMA_VECTOR
        for (i = 0; i < len; ++i)
            res[i] = lo;

        return;
    }

    assert(lo < hi);
    buf = (int *)mkl_malloc(len * sizeof(int), 64);
    assert(buf != nullptr);

    err = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, buf, (int)lo, (int)hi + 1);
    assert(err == VSL_STATUS_OK);

    DIST_PRAGMA_VECTOR
    for (i = 0; i < len; ++i)
        res[i] = (npy_uint16)buf[i];

    mkl_free(buf);
}

void irk_rand_int16_vec(irk_state *state, npy_intp len, npy_int16 *res, const npy_int16 lo, const npy_int16 hi)
{
    int err = 0;
    npy_intp i = 0;
    int *buf = nullptr;

    if (len < 1)
        return;

    if (len > MKL_INT_MAX)
    {
        irk_rand_int16_vec(state, MKL_INT_MAX, res, lo, hi);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    if (lo == hi)
    {
        DIST_PRAGMA_VECTOR
        for (i = 0; i < len; ++i)
            res[i] = lo;

        return;
    }

    assert(lo < hi);
    buf = (int *)mkl_malloc(len * sizeof(int), 64);
    assert(buf != nullptr);

    err = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, buf, (int)lo, (int)hi + 1);
    assert(err == VSL_STATUS_OK);

    DIST_PRAGMA_VECTOR
    for (i = 0; i < len; ++i)
        res[i] = (npy_int16)buf[i];

    mkl_free(buf);
}

void irk_rand_uint32_vec(irk_state *state, npy_intp len, npy_uint32 *res, const npy_uint32 lo, const npy_uint32 hi)
{
    int err = 0;
    unsigned int intm = INT_MAX;

    if (len < 1)
        return;

    if (len > MKL_INT_MAX)
    {
        irk_rand_uint32_vec(state, MKL_INT_MAX, res, lo, hi);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    /* optimization for lo = 0 and hi = 2**32-1 */
    if (!(lo || ~hi))
    {
        err = viRngUniformBits32(VSL_RNG_METHOD_UNIFORMBITS32_STD, state->stream, len, (unsigned int *)res);
        assert(err == VSL_STATUS_OK);

        return;
    }

    if (hi >= intm)
    {

        npy_int32 shft = ((npy_uint32)intm) + ((npy_uint32)1);
        int i;

        /* if lo is non-zero, shift one more to accommodate possibility of hi being ULONG_MAX */
        if (lo)
            shft++;

        err = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, (int *)res, (int)(lo - shft), (int)(hi - shft + 1U));
        assert(err == VSL_STATUS_OK);

        DIST_PRAGMA_VECTOR
        for (i = 0; i < len; ++i)
            res[i] += shft;
    }
    else
    {
        err = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, (int *)res, (int)lo, (int)hi + 1);
        assert(err == VSL_STATUS_OK);
    }
}

void irk_rand_int32_vec(irk_state *state, npy_intp len, npy_int32 *res, const npy_int32 lo, const npy_int32 hi)
{
    int err = 0;
    int intm = INT_MAX;

    if (len < 1)
        return;

    if (len > MKL_INT_MAX)
    {
        irk_rand_int32_vec(state, MKL_INT_MAX, res, lo, hi);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    if (hi >= intm)
    {
        int i;

        irk_rand_uint32_vec(state, len, (npy_uint32 *)res, 0U, (npy_uint32)(hi - lo));

        DIST_PRAGMA_VECTOR
        for (i = 0; i < len; ++i)
            res[i] += lo;
    }
    else
    {
        err = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, (int *)res, (int)lo, (int)hi + 1);
        assert(err == VSL_STATUS_OK);
    }
}

void irk_rand_uint64_vec(irk_state *state, npy_intp len, npy_uint64 *res, const npy_uint64 lo, const npy_uint64 hi)
{
    npy_uint64 rng;
    int err = 0;
    npy_intp i = 0;

    if (len < 1)
        return;

    if (len > MKL_INT_MAX)
    {
        irk_rand_uint64_vec(state, MKL_INT_MAX, res, lo, hi);

        res += MKL_INT_MAX;
        len -= MKL_INT_MAX;
    }

    /* optimization for lo = 0 and hi = 2**64-1 */
    if (!(lo || ~hi))
    {
        err = viRngUniformBits64(VSL_RNG_METHOD_UNIFORMBITS64_STD, state->stream, len, (unsigned MKL_INT64 *)res);
        assert(err == VSL_STATUS_OK);

        return;
    }

    rng = hi - lo;
    if (!rng)
    {
        DIST_PRAGMA_VECTOR
        for (i = 0; i < len; ++i)
            res[i] = lo;

        return;
    }

    rng++;

    if (rng <= (npy_uint64)INT_MAX)
    {
        int *buf = (int *)mkl_malloc(len * sizeof(int), 64);
        assert(buf != nullptr);

        err = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->stream, len, buf, 0, (int)rng);
        assert(err == VSL_STATUS_OK);

        DIST_PRAGMA_VECTOR
        for (i = 0; i < len; ++i)
            res[i] = lo + ((npy_uint64)buf[i]);

        mkl_free(buf);
    }
    else
    {
        npy_uint64 mask = rng;
        npy_uint64 *buf = nullptr;
        npy_intp n_accepted = 0;

        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        mask |= mask >> 32;

        buf = (npy_uint64 *)mkl_malloc(len * sizeof(npy_uint64), 64);
        assert(buf != nullptr);

        while (n_accepted < len)
        {
            npy_intp k = 0;
            npy_intp batchSize = len - n_accepted;

            err = viRngUniformBits64(VSL_RNG_METHOD_UNIFORM_STD, state->stream, batchSize, (unsigned MKL_INT64 *)buf);
            assert(err == VSL_STATUS_OK);

            for (k = 0; k < batchSize; ++k)
            {
                npy_uint64 value = buf[k] & mask;
                if (value <= rng)
                {
                    res[n_accepted++] = lo + value;
                }
            }
        }

        mkl_free(buf);
    }
}

void irk_rand_int64_vec(irk_state *state, npy_intp len, npy_int64 *res, const npy_int64 lo, const npy_int64 hi)
{
    npy_uint64 rng = 0;
    npy_intp i = 0;

    if (len < 1)
        return;

    rng = ((npy_uint64)hi) - ((npy_uint64)lo);

    irk_rand_uint64_vec(state, len, (npy_uint64 *)res, 0, rng);

    DIST_PRAGMA_VECTOR
    for (i = 0; i < len; ++i)
        res[i] = res[i] + lo;
}

const MKL_INT cholesky_storage_flags[3] = {
    VSL_MATRIX_STORAGE_FULL,
    VSL_MATRIX_STORAGE_PACKED,
    VSL_MATRIX_STORAGE_DIAGONAL};

void irk_multinormal_vec_ICDF(irk_state *state, npy_intp len, double *res, const int dim, double *mean_vec, double *ch,
                              const ch_st_enum storage_flag)
{
    int err = 0;
    const MKL_INT storage_mode = cholesky_storage_flags[storage_flag];

    err = vdRngGaussianMV(VSL_RNG_METHOD_GAUSSIANMV_ICDF, state->stream, len, res, dim, storage_mode, mean_vec, ch);
    assert(err == VSL_STATUS_OK);
}

void irk_multinormal_vec_BM1(irk_state *state, npy_intp len, double *res, const int dim, double *mean_vec, double *ch,
                             const ch_st_enum storage_flag)
{
    int err = 0;
    const MKL_INT storage_mode = cholesky_storage_flags[storage_flag];

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        err = vdRngGaussianMV(VSL_RNG_METHOD_GAUSSIANMV_BOXMULLER, state->stream, MKL_INT_MAX, res, dim, storage_mode, mean_vec, ch);
        assert(err == VSL_STATUS_OK);

        res += MKL_INT_MAX * dim;
        len -= MKL_INT_MAX;
    }

    err = vdRngGaussianMV(VSL_RNG_METHOD_GAUSSIANMV_BOXMULLER, state->stream, len, res, dim, storage_mode, mean_vec, ch);
    assert(err == VSL_STATUS_OK);
}

void irk_multinormal_vec_BM2(irk_state *state, npy_intp len, double *res, const int dim, double *mean_vec, double *ch,
                             const ch_st_enum storage_flag)
{
    int err = 0;
    const MKL_INT storage_mode = cholesky_storage_flags[storage_flag];

    if (len < 1)
        return;

    while (len > MKL_INT_MAX)
    {
        err = vdRngGaussianMV(VSL_RNG_METHOD_GAUSSIANMV_BOXMULLER2, state->stream, MKL_INT_MAX, res, dim, storage_mode, mean_vec, ch);
        assert(err == VSL_STATUS_OK);

        res += MKL_INT_MAX * dim;
        len -= MKL_INT_MAX;
    }

    err = vdRngGaussianMV(VSL_RNG_METHOD_GAUSSIANMV_BOXMULLER2, state->stream, len, res, dim, storage_mode, mean_vec, ch);
    assert(err == VSL_STATUS_OK);
}

/* This code is taken from distribution.c, and is currently unused. It is retained here for
   possible future optimization of sampling from multinomial */

static double irk_double(irk_state *state)
{
    double res;

    irk_double_vec(state, 1, &res);

    return res;
}
