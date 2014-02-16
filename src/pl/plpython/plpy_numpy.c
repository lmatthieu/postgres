#include "Python.h"
#include "numpy/npy_math.h"
#include "numpy/ndarrayobject.h"

#include "postgres.h"

#include "access/htup_details.h"
#include "access/xact.h"
#include "catalog/pg_type.h"
#include "executor/spi.h"
#include "mb/pg_wchar.h"
#include "parser/parse_type.h"
#include "utils/memutils.h"
#include "utils/syscache.h"

#include "plpython.h"

#include "plpy_spi.h"

#include "plpy_elog.h"
#include "plpy_main.h"
#include "plpy_planobject.h"
#include "plpy_plpymodule.h"
#include "plpy_procedure.h"
#include "plpy_resultobject.h"

#include "plpy_numpy.h"

static PyObject *
PLy_spi_execute_fetch_result_to_matrix(SPITupleTable *tuptable, int rows, int status);

PyObject *
PLy_spi_execute_query_to_matrix(char *query, long limit)
{
	int			rv;
	volatile MemoryContext oldcontext;
	volatile ResourceOwner oldowner;
	PyObject   *ret = NULL;

	oldcontext = CurrentMemoryContext;
	oldowner = CurrentResourceOwner;

	PLy_spi_subtransaction_begin(oldcontext, oldowner);

	PG_TRY();
	{
		PLyExecutionContext *exec_ctx = PLy_current_execution_context();

		pg_verifymbstr(query, strlen(query), false);
		rv = SPI_execute(query, exec_ctx->curr_proc->fn_readonly, limit);
		ret = PLy_spi_execute_fetch_result_to_matrix(SPI_tuptable, SPI_processed, rv);

		PLy_spi_subtransaction_commit(oldcontext, oldowner);
	}
	PG_CATCH();
	{
		PLy_spi_subtransaction_abort(oldcontext, oldowner);
		return NULL;
	}
	PG_END_TRY();

	if (rv < 0)
	{
		Py_XDECREF(ret);
		PLy_exception_set(PLy_exc_spi_error,
						  "SPI_execute failed: %s",
						  SPI_result_code_string(rv));
		return NULL;
	}

	return ret;
}

static PyObject *
PLy_spi_execute_fetch_result_to_matrix(SPITupleTable *tuptable, int rows, int status)
{
	PLyResultObject *result;
	volatile MemoryContext oldcontext;

	result = (PLyResultObject *) PLy_result_new();
	Py_DECREF(result->status);
	result->status = PyInt_FromLong(status);

	if (status > 0 && tuptable == NULL)
	{
		Py_DECREF(result->nrows);
		result->nrows = PyInt_FromLong(rows);
	}
	else if (status > 0 && tuptable != NULL)
	{
		PLyTypeInfo args;
		int			i;

		Py_DECREF(result->nrows);
		result->nrows = PyInt_FromLong(rows);
		PLy_typeinfo_init(&args);

		oldcontext = CurrentMemoryContext;
		PG_TRY();
		{
			MemoryContext oldcontext2;

			if (rows)
			{
				import_array();
				Py_DECREF(result->rows);
				int n, m;
				double *data = 0;

				PLy_input_tuple_funcs(&args, tuptable->tupdesc);
				n = rows;
				m = args.in.r.natts;
				npy_intp dims[2] = {n, m};
				result = PyArray_SimpleNew(2, dims, NPY_DOUBLE);

				for (i = 0; i < rows; i++)
				{
					PLyTypeInfo *info = &args;
					HeapTuple tuple = tuptable->vals[i];
					TupleDesc desc = tuptable->tupdesc;
					bool is_null;

					int j;

					for (j = 0; j < info->in.r.natts; j++)
					{
						Datum		vattr;
						bool		is_null;
						double *obj = PyArray_GETPTR2(result, i, j);

						if (desc->attrs[j]->attisdropped)
							continue;
						vattr = heap_getattr(tuple, (j + 1), desc, &is_null);

						if (is_null || info->in.r.atts[j].func == NULL)
							*obj = NPY_NAN;
						else
							*obj = DatumGetFloat8(vattr);
					}
				}

			}

/*
 * Save tuple descriptor for later use by result set metadata
 * functions.  Save it in TopMemoryContext so that it survives
 * outside of an SPI context.  We trust that PLy_result_dealloc()
 * will clean it up when the time is right.  (Do this as late as
 * possible, to minimize the number of ways the tupdesc could get
 * leaked due to errors.)
 */
			oldcontext2 = MemoryContextSwitchTo(TopMemoryContext);
			result->tupdesc = CreateTupleDescCopy(tuptable->tupdesc);
			MemoryContextSwitchTo(oldcontext2);
		}
		PG_CATCH();
		{
			MemoryContextSwitchTo(oldcontext);
			PLy_typeinfo_dealloc(&args);
			Py_DECREF(result);
			PG_RE_THROW();
		}
		PG_END_TRY();

		PLy_typeinfo_dealloc(&args);
		SPI_freetuptable(tuptable);
	}

	return (PyObject *) result;
}

/* execute(query="select * from foo", limit=5, ismatrix=False)
 * execute(plan=plan, values=(foo, bar), limit=5)
 */
PyObject *
PLy_spi_execute_toarray(PyObject *self, PyObject *args)
{
	char	   *query;
	PyObject   *list = NULL;
	long		limit = 0;

	if (PyArg_ParseTuple(args, "s|l", &query, &limit))
	{
		return PLy_spi_execute_query_to_matrix(query, limit);
	}

	PLy_exception_set(PLy_exc_error, "plpy.execute expected a query or a plan");
	return NULL;
}
