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
/*
 Adopted from NumPy's Random kit 1.3,
 Copyright (c) 2003-2005, Jean-Sebastien Roy (js@jeannot.org)
 */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <assert.h>

#ifdef _WIN32
/*
 * Windows
 * XXX: we have to use this ugly defined(__GNUC__) because it is not easy to
 * detect the compiler used in distutils itself
 */
#if (defined(__GNUC__) && defined(NEED_MINGW_TIME_WORKAROUND))

/*
 * FIXME: ideally, we should set this to the real version of MSVCRT. We need
 * something higher than 0x601 to enable _ftime64 and co
 */
#define __MSVCRT_VERSION__ 0x0700
#include <time.h>
#include <sys/timeb.h>

/*
 * mingw msvcr lib import wrongly export _ftime, which does not exist in the
 * actual msvc runtime for version >= 8; we make it an alias to _ftime64, which
 * is available in those versions of the runtime
 */
#define _FTIME(x) _ftime64((x))
#else
#include <time.h>
#include <sys/timeb.h>
#define _FTIME(x) _ftime((x))
#endif

#ifndef RK_NO_WINCRYPT
/* Windows crypto */
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0400
#endif
#include <windows.h>
#include <wincrypt.h>
#endif

#else
/* Unix */
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#endif

#include "randomkit.h"

#ifndef RK_DEV_URANDOM
#define RK_DEV_URANDOM "/dev/urandom"
#endif

#ifndef RK_DEV_RANDOM
#define RK_DEV_RANDOM "/dev/random"
#endif

const char *irk_strerror[RK_ERR_MAX] =
    {
        "no error",
        "random device unvavailable"};

/* static functions */
static unsigned long irk_hash(unsigned long key);

void irk_dealloc_stream(irk_state *state)
{
    VSLStreamStatePtr stream = state->stream;

    if (stream)
    {
        vslDeleteStream(&stream);
    }
}

const MKL_INT brng_list[BRNG_KINDS] = {
    VSL_BRNG_MT19937,
    VSL_BRNG_SFMT19937,
    VSL_BRNG_WH,
    VSL_BRNG_MT2203,
    VSL_BRNG_MCG31,
    VSL_BRNG_R250,
    VSL_BRNG_MRG32K3A,
    VSL_BRNG_MCG59,
    VSL_BRNG_PHILOX4X32X10,
    VSL_BRNG_NONDETERM,
    VSL_BRNG_ARS5};

/* Mersenne-Twister 2203 algorithm and Wichmann-Hill algorithm
 * each have a parameter which produces a family of BRNG algorithms,
 * MKL identifies individual members of these families by VSL_BRNG_ALGO + family_id
 */
#define SIZE_OF_MT2203_FAMILY 6024
#define SIZE_OF_WH_FAMILY 273

int irk_get_brng_mkl(irk_state *state)
{
    int i, mkl_brng_id = vslGetStreamStateBrng(state->stream);

    if ((VSL_BRNG_MT2203 <= mkl_brng_id) && (mkl_brng_id < VSL_BRNG_MT2203 + SIZE_OF_MT2203_FAMILY))
        mkl_brng_id = VSL_BRNG_MT2203;
    else if ((VSL_BRNG_WH <= mkl_brng_id) && (mkl_brng_id < VSL_BRNG_WH + SIZE_OF_WH_FAMILY))
        mkl_brng_id = VSL_BRNG_WH;

    for (i = 0; i < BRNG_KINDS; i++)
        if (mkl_brng_id == brng_list[i])
            return i;

    return -1;
}

int irk_get_brng_and_stream_mkl(irk_state *state, unsigned int *stream_id)
{
    int i, mkl_brng_id = vslGetStreamStateBrng(state->stream);

    if ((VSL_BRNG_MT2203 <= mkl_brng_id) && (mkl_brng_id < VSL_BRNG_MT2203 + SIZE_OF_MT2203_FAMILY))
    {
        *stream_id = (unsigned int)(mkl_brng_id - VSL_BRNG_MT2203);
        mkl_brng_id = VSL_BRNG_MT2203;
    }
    else if ((VSL_BRNG_WH <= mkl_brng_id) && (mkl_brng_id < VSL_BRNG_WH + SIZE_OF_WH_FAMILY))
    {
        *stream_id = (unsigned int)(mkl_brng_id - VSL_BRNG_WH);
        mkl_brng_id = VSL_BRNG_WH;
    }

    for (i = 0; i < BRNG_KINDS; i++)
        if (mkl_brng_id == brng_list[i])
        {
            *stream_id = (unsigned int)(0);
            return i;
        }

    return -1;
}

void irk_seed_mkl(irk_state *state, const unsigned int seed, const irk_brng_t brng, const unsigned int stream_id)
{
    VSLStreamStatePtr stream_loc;
    int err = VSL_STATUS_OK;
    const MKL_INT mkl_brng = brng_list[brng];

    if (NULL == state->stream)
    {
        err = vslNewStream(&(state->stream), mkl_brng + stream_id, seed);

        assert(err == VSL_STATUS_OK);
    }
    else
    {
        err = vslNewStream(&stream_loc, mkl_brng + stream_id, seed);
        assert(err == VSL_STATUS_OK);

        err = vslDeleteStream(&(state->stream));
        assert(err == VSL_STATUS_OK);

        state->stream = stream_loc;
    }
    if (err)
    {
        printf(
            "irk_seed_mkl: encountered error when calling Intel(R) MKL\n");
    }
}

void irk_seed_mkl_array(irk_state *state, const unsigned int seed_vec[], const int seed_len,
                        const irk_brng_t brng, const unsigned int stream_id)
{
    VSLStreamStatePtr stream_loc;
    int err = VSL_STATUS_OK;
    const MKL_INT mkl_brng = brng_list[brng];

    if (NULL == state->stream)
    {

        err = vslNewStreamEx(&(state->stream), mkl_brng + stream_id, (MKL_INT)seed_len, seed_vec);

        assert(err == VSL_STATUS_OK);
    }
    else
    {

        err = vslNewStreamEx(&stream_loc, mkl_brng + stream_id, (MKL_INT)seed_len, seed_vec);
        if (err == VSL_STATUS_OK)
        {

            err = vslDeleteStream(&(state->stream));
            assert(err == VSL_STATUS_OK);

            state->stream = stream_loc;
        }
    }
}

irk_error
irk_randomseed_mkl(irk_state *state, const irk_brng_t brng, const unsigned int stream_id)
{
#ifndef _WIN32
    struct timeval tv;
#else
    struct _timeb tv;
#endif
    int no_err;
    unsigned int *seed_array;
    size_t buf_size = 624;
    size_t seed_array_len = buf_size * sizeof(unsigned int);

    seed_array = (unsigned int *)malloc(seed_array_len);
    no_err = irk_devfill(seed_array, seed_array_len, 0) == RK_NOERR;

    if (no_err)
    {
        /* ensures non-zero seed */
        seed_array[0] |= 0x80000000UL;
        irk_seed_mkl_array(state, seed_array, buf_size, brng, stream_id);
        free(seed_array);

        return RK_NOERR;
    }
    else
    {
        free(seed_array);
    }

#ifndef _WIN32
    gettimeofday(&tv, NULL);
    irk_seed_mkl(state, irk_hash(getpid()) ^ irk_hash(tv.tv_sec) ^ irk_hash(tv.tv_usec) ^ irk_hash(clock()), brng, stream_id);
#else
    _FTIME(&tv);
    irk_seed_mkl(state, irk_hash(tv.time) ^ irk_hash(tv.millitm) ^ irk_hash(clock()), brng, stream_id);
#endif

    return RK_ENODEV;
}

/*
 *  Python needs this to determine the amount memory to allocate for the buffer
 */
int irk_get_stream_size(irk_state *state)
{
    return vslGetStreamSize(state->stream);
}

void irk_get_state_mkl(irk_state *state, char *buf)
{
    int err = vslSaveStreamM(state->stream, buf);

    if (err != VSL_STATUS_OK)
    {
        assert(err == VSL_STATUS_OK);
        printf(
            "irk_get_state_mkl encountered error when calling Intel(R) MKL\n");
    }
}

int irk_set_state_mkl(irk_state *state, char *buf)
{
    int err = vslLoadStreamM(&(state->stream), buf);

    return (err == VSL_STATUS_OK) ? 0 : 1;
}

int irk_leapfrog_stream_mkl(irk_state *state, const MKL_INT k, const MKL_INT nstreams)
{
    int err;

    err = vslLeapfrogStream(state->stream, k, nstreams);

    switch (err)
    {
    case VSL_STATUS_OK:
        return 0;
    case VSL_RNG_ERROR_LEAPFROG_UNSUPPORTED:
        return 1;
    default:
        return -1;
    }
}

int irk_skipahead_stream_mkl(irk_state *state, const long long int nskip)
{
    int err;

    err = vslSkipAheadStream(state->stream, nskip);

    switch (err)
    {
    case VSL_STATUS_OK:
        return 0;
    case VSL_RNG_ERROR_SKIPAHEAD_UNSUPPORTED:
        return 1;
    default:
        return -1;
    }
}

/* Thomas Wang 32 bits integer hash function */
static unsigned long
irk_hash(unsigned long key)
{
    key += ~(key << 15);
    key ^= (key >> 10);
    key += (key << 3);
    key ^= (key >> 6);
    key += ~(key << 11);
    key ^= (key >> 16);
    return key;
}

void irk_random_vec(irk_state *state, const int len, unsigned int *res)
{
    viRngUniformBits(VSL_RNG_METHOD_UNIFORMBITS_STD, state->stream, len, res);
}

void irk_fill(void *buffer, size_t size, irk_state *state)
{
    unsigned int r;
    unsigned char *buf = reinterpret_cast<unsigned char *>(buffer);
    int err, len;

    /* len = size / 4 */
    len = (size >> 2);
    err = viRngUniformBits32(VSL_RNG_METHOD_UNIFORMBITS32_STD, state->stream, len, (unsigned int *)buf);
    assert(err == VSL_STATUS_OK);

    /* size = size % 4 */
    size &= 0x03;
    if (!size)
    {
        return;
    }

    buf += (len << 2);
    err = viRngUniformBits32(VSL_RNG_METHOD_UNIFORMBITS32_STD, state->stream, 1, &r);
    assert(err == VSL_STATUS_OK);

    for (; size; r >>= 8, size--)
    {
        *(buf++) = (unsigned char)(r & 0xFF);
    }
    if (err)
        printf("irk_fill: error encountered when calling Intel(R) MKL \n");
}

irk_error
irk_devfill(void *buffer, size_t size, int strong)
{
#ifndef _WIN32
    FILE *rfile;
    int done;

    if (strong)
    {
        rfile = fopen(RK_DEV_RANDOM, "rb");
    }
    else
    {
        rfile = fopen(RK_DEV_URANDOM, "rb");
    }
    if (rfile == NULL)
    {
        return RK_ENODEV;
    }
    done = fread(buffer, size, 1, rfile);
    fclose(rfile);
    if (done)
    {
        return RK_NOERR;
    }
#else

#ifndef RK_NO_WINCRYPT
    HCRYPTPROV hCryptProv;
    BOOL done;

    if (!CryptAcquireContext(&hCryptProv, NULL, NULL, PROV_RSA_FULL,
                             CRYPT_VERIFYCONTEXT) ||
        !hCryptProv)
    {
        return RK_ENODEV;
    }
    done = CryptGenRandom(hCryptProv, size, (unsigned char *)buffer);
    CryptReleaseContext(hCryptProv, 0);
    if (done)
    {
        return RK_NOERR;
    }
#endif

#endif
    return RK_ENODEV;
}

irk_error
irk_altfill(void *buffer, size_t size, int strong, irk_state *state)
{
    irk_error err;

    err = irk_devfill(buffer, size, strong);
    if (err)
    {
        irk_fill(buffer, size, state);
    }
    return err;
}
