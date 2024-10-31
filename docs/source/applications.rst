.. _`applications`:

Applications
============

LiberTEM itself is primarily a framework for efficient data access and
compute parallelisation. Through the :ref:`UDF interface<user-defined functions>`,
the user is able to define their own analyses and benefit from the optimisations
that LiberTEM provides on any dataset in a way which scales to any machine.

Some application-specific code developed using LiberTEM has been
spun out as sub-packages that can be installed independent of
LiberTEM core. See :ref:`packages` for the current overview of sub-packages.

A number of standalone applications are provided as example
Jupyter notebooks, available for download from
<`the examples page <https://github.com/LiberTEM/LiberTEM/tree/master/examples>`_>
and also integrated into this documentation via the links below:

.. toctree::
   :maxdepth: 2

   app/phasecontrast
   app/amorphous
   app/strain
   app/phasechange
   app/processing

* `Ptychography (external) <https://ptychography-4-0.github.io/ptychography/>`_