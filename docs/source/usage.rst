.. _`usage documentation`:

GUI usage
=========

LiberTEM ships with a convenient web-based interface for quickly understanding a dataset
through a selection of common analyses (virtual detectors, centre-of-mass, clustering etc).

.. note::
    Note that the GUI is currently limited to 2D visualizations, which makes it adapted
    4D-STEM imagery but not spectrum imaging such as STEM-EELS. If you have a need for
    display of 1D signals please `file an issue <https://github.com/liberTEM/LiberTEM/issues>`_.

Starting the GUI
----------------

You can the start the interface from the command line after 
activating the virtualenv or conda environment where LiberTEM is installed.

.. code-block:: shell

    (libertem) $ libertem-server

In most situations this will automatically launch a webpage in your browser to start
loading data and running analyses. If it didn't open automatically, you can access
it by default at http://localhost:9000.

.. note::

   The GUI is tested to work on Firefox and Chromium-based browsers for now. If you
   cannot use a compatible browser for some reason, please
   `file an issue <https://github.com/liberTEM/LiberTEM/issues>`_.

It is also possible to connect to a :code:`libertem-server` on a remote machine
though this requires special configuration and depends on your network environment
(see :ref:`server config`).

Create the local cluster
------------------------

The first page in the GUI allows you to specify the compute resources
the system will use to process your data (number of CPUs, GPUs etc). By default
all available resources are selected, though you can reduce this if needed.

..  figure:: ./images/use/create.png

.. note::

   Use of GPUs requires a working CuPy installation, see :ref:`CuPy install <cupy install>`
   for more information.

Opening data
------------

Next, LiberTEM shows a button to start browsing for
available files. On a local machine that's simply the local filesystem.

.. note:: See :ref:`sample data` for publicly available datasets.

..  figure:: ./images/use/browse.png

This opens the file browser dialogue. On top it shows the current directory,
below it lists all files and subdirectories in that directory. You select an
entry by clicking once on it. You can move up one directory with the ".." entry
on top of the list. The file browser is still very basic. Possible improvements
are discussed in `Issue #83 <https://github.com/LiberTEM/LiberTEM/issues/83>`_.
Contributions are highly appreciated! This example opens an HDF5 file :cite:`Zeltmann2019`.

..  figure:: ./images/use/open.png

You can also bookmark locations you frequently need to access, using the
star icon. The bookmarks are then found under "Go to...".

..  figure:: ./images/use/star.png

After selecting a file, you set the type in the drop-down menu at the top of the
dialogue above the file name. After that you set the appropriate parameters that
depend on the file type. Clicking on "Load Dataset" will open the file with the
selected parameters. The interface and internal logic to find good presets based
on file type and available metadata, validate the inputs and display helpful
error messages is still work in progress. Contributions are highly appreciated!

See :ref:`Loading using the GUI` for more detailed instructions and
format-specific information.

..  figure:: ./images/use/type.png

Running analyses
----------------

Once a dataset is loaded, you can add analyses to it. As an example we choose a
"Ring" analysis, which implements a ring-shaped virtual detector.

..  figure:: ./images/use/add_analysis.png

..  figure:: ./images/use/adjust.png


This analysis shows two views on your data: the two detector dimensions on
the left, the scanning dimensions on the right, assuming a 4D-STEM dataset.
For the general case, we also call the detector dimensions the *signal
dimensions*, and the scanning dimensions the *navigation dimensions*.
See also :ref:`concepts` for more information on axes and coordinate system.

Directly after
adding the analysis, LiberTEM starts calculating an average of all the detector
frames. The average is overlaid with the mask representing the virtual detector. The view on the right
will later show the result of applying the mask to the data. In the beginning it
is empty. The first processing might take a while depending on file size and I/O
performance. Fast SSDs and enough RAM to keep the working files in the file
system cache are highly recommended for a good user experience.

You can adjust the virtual detector by dragging the handles in the GUI. Below it
shows the parameters in numerical form. This is useful to extract positions, for
example for scripting.

After clicking "Apply", LiberTEM performs the calculation and shows the result
in scan coordinates on the right side.

..  figure:: ./images/use/apply.png

Instead of average, you can select "Standard Deviation". This calculates
standard deviation of all detector frames.

..  figure:: ./images/use/std_dev.png

If you are interested in individual frames rather than the average, you can
switch to "Pick" mode in the "Mode" drop-down menu directly below the detector
window.

..  figure:: ./images/use/pick.png

In "Pick" mode, a selector appears in the result frame on the right. You can
drag it around with the mouse to see the frames live in the left window. The
picked coordinates are displayed along with the virtual detector parameters
below the frame window on the left.

..  figure:: ./images/use/pick_frame.png

If you are interested in a limited region, the ROI dropdown provides the option
to select a rectangular region. For example if you select "Rect", the
average/standard deviation is calculated over all images that lie inside selected
rectangle. You can adjust the rectangle by dragging the handles in the GUI.

..  figure:: ./images/use/rect.png

Some analyses, such as the Center of Mass (COM) analysis, can render the result
in different ways. You can select different result channels in the "Channel" drop-down menu
below the right window.

..  figure:: ./images/use/image.png

.. _`download results`:

Downloading results
-------------------

After an analysis has finished running, you can download the results. Clicking the download button
below the analysis will open a dialog:

..  figure:: ./images/use/download-btn.png

In the download dialog, you can choose between different file formats, and separately
download the available results.

..  figure:: ./images/use/download-modal.png

You can also download a Jupyter notebook corresponding to the analysis and
continue working with the same parameters using scripting.

.. figure:: ./images/use/download-jupyter.png

It's also possible to copy individual cells of Jupyter notebook directly from GUI, with an option
to copy the complete source code.

.. figure:: ./images/use/copy-jupyter.png

Keyboard controls
~~~~~~~~~~~~~~~~~

You can use arrow keys to change the coordinate parameters of any analysis. To
do this, click on the handle you want to modify, and then use the arrow keys to
move the handle. Hold shift to move in larger steps.

Application-specific documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more applications, like strain mapping and crystallinity analysis, please
see the :doc:`Applications <applications>` section.

.. _`server config`:

Configuring the LiberTEM server (Advanced)
------------------------------------------

The LiberTEM GUI is based on a client-server architecture. To use the LiberTEM GUI, you need to
have the server running on the machine where your data is available. For using LiberTEM from
Python scripts, this is not necessary, see :ref:`api documentation`.

By default, this starts the server on http://localhost:9000, which you can verify by the
log output::

    [2018-08-08 13:57:58,266] INFO [libertem.web.server.main:886] listening on localhost:9000

To modify the configuration of the server (address, port, autorization etc.), the
:code:`libertem-server` has a number of options available:

.. include:: autogenerated/libertem-server.help
    :literal:

.. versionadded:: 0.4.0
    :code:`--browser` / :code:`--no-browser` option was added, open browser by default.
.. versionadded:: 0.6.0
    :code:`-l, --log-level` was added.
.. versionadded:: 0.8.0
    :code:`-t, --token-path` was added and :code:`-h, --host` was re-enabled.
.. versionadded:: 0.9.0
    :code:`--preload` and :code:`--insecure` were added.

To access LiberTEM remotely, you can use :ref:`use SSH forwarding <ssh forwarding>`
or our :ref:`jupyter integration`, if you already have JupyterHub or JupyterLab
set up on a server.