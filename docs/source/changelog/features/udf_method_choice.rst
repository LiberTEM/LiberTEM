[Feature] UDF process_ method choice at runtime
===============================================
* A UDF can now implement multiple types of processing routine
  (:code:`process_frame`, :code:`process_tile`, :code:`process_partition`)
  and signal to the :code:`UDFRunner` which to use based on runtime
  information via the :code:`UDF.get_method()` endpoint.
  (:issue:`1508`, :pr:`1509`).
