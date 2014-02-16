/*
 * src/pl/plpython/plpy_numpy.h
 */

#ifndef PLPY_NUMPY_H
#define PLPY_NUMPY_H

#include "utils/palloc.h"
#include "utils/resowner.h"

extern PyObject *
PLy_spi_execute_query_to_matrix(char *query, long limit);
PyObject *
PLy_spi_execute_toarray(PyObject *self, PyObject *args);

#endif   /* PLPY_NUMPY_H */
