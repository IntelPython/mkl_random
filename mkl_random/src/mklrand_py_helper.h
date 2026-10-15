// clang-format off
#ifndef _MKLRAND_PY_HELPER_H_
#define _MKLRAND_PY_HELPER_H_

#include <Python.h>
// clang-format on

static PyObject *empty_py_bytes(npy_intp length, void **bytesVec)
{
    PyObject *b;
    b = PyBytes_FromStringAndSize(NULL, length);
    if (b) {
        *bytesVec = PyBytes_AS_STRING(b);
    }
    return b;
}

static char *py_bytes_DataPtr(PyObject *b) { return PyBytes_AS_STRING(b); }

static int is_bytes_object(PyObject *b) { return PyBytes_Check(b); }

#endif /* _MKLRAND_PY_HELPER_H_ */
