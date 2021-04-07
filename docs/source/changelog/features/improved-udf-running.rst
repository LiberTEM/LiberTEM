[Feature] Improve UDF running
=============================
* Allow running multiple UDFs "at once" on a single `DataSet`
* Allow usage from an asynchronous context
* Allow getting results iteratively as a generator (both sync and async)
* Simple plotting, including live-updates (`Context.run_udf(..., plots=True)` for very simple usage)
* See :pr:`1011` for details
