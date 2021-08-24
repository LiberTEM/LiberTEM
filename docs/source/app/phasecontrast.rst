.. `phasecontrast app`:

Phase contrast
==============

Center of mass / first moment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can get a first moment :cite:`ROSE1976251` visualization of your data set by
selecting "Center of mass" in the "Add analysis" dropdown:

..  figure:: ../images/app/com-example.png

Take note of the "Channel" drop down under the right image, where you can select
different visualizations derived from the vector field.

Center of mass is also available in the :class:`~libertem.api.Context` API.
Please see `this example
<https://github.com/LiberTEM/LiberTEM/blob/master/examples/center_of_mass.ipynb>`_
and the reference documentation for
:meth:`~libertem.api.Context.create_com_analysis`.

Ptychographic reconstruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please see `the ptychography 4.0 project
<https://ptychography-4-0.github.io/ptychography/algorithms.html>`_ for
ptychography algorithms implemented as LiberTEM UDFs.
