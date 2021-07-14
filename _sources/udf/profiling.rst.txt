.. _`udf profiling`:

.. testsetup:: *

    from libertem.api import Context


Profiling UDFs
==============

To bring your UDF to production, it may be necessary to look into the performance and
efficiency of your function. There may be some low-hanging fruit to optimize away, but
they only really become visible once you profile your code. Just looking at code and guessing
which parts are expensive can be misleading and cause you to spend time optimizing the wrong part
of your code. Always measure!

Profiling means instrumenting the execution of a program and finding out which parts
of your code use how much of a given resource (for example, CPU time, or memory). 

Prerequisite: :code:`InlineJobExecutor`
---------------------------------------

By default, your UDF will be run in a multi-process or multi-threaded
environment. This makes profiling a challenge, since the usual profiling tools
will only capture information on the main process that performs the final
reduction and not the processes that do most of the work.

So, in order to use the default Python profiling mechanisms, all parts of the UDF
should be executed in the same single thread. LiberTEM provides what it calls
:class:`~libertem.executor.inline.InlineJobExecutor` for this purpose. Improving
the execution time in a single-threaded executor will almost always improve the
multi-process execution time. Nevertheless, you should confirm the impact of
changes in your production environment.

To use the :class:`~libertem.executor.inline.InlineJobExecutor`, pass it to the
:class:`~libertem.api.Context` on construction:

.. testcode::
   
   from libertem.executor.inline import InlineJobExecutor
   ctx = Context(executor=InlineJobExecutor())

Then, you can continue as usual, loading data, executing your UDF, etc.

Using progress bar
---------------------

A progress bar is a graphical tool that shows a far a process has progressed.
It can be used to assess the partitioning of data, since the progress is
indicated on a per partition basis in the following way:

.. code-block:: text
     
    res = ctx.run_udf(udf=fit_udf, dataset=ds, roi=roi, progress=True)
    > 100%|██████████| 18/18 [01:17<00:00,  4.29s/it]

This run is using 18 partitions, and took 4.29s per partition. 

Line profiling using `line_profiler`
------------------------------------------

Profiling comes in different forms, some will show you how much time each function
of a whole program took, others can give you a line-by-line overview for functions
you are interested in, like `line_profiler`. For most UDFs this will be sufficient
for performance analysis.

First, you need to install the `line_profiler` package via pip, having your
conda environment or virtualenv activated:

.. code-block:: shell

   (libertem) $ python -m pip install line_profiler

Then the easiest way to get started is to use the ipython/jupyter integration of
`line_profiler`. Put :code:`%load_ext line_profiler` somewhere in your notebook,
then you can use the :code:`%lprun` magic command to run your UDF while profiling:

.. code-block:: text

   %lprun -f YourUDF.get_task_data -f YourUDF.process_frame ctx.run_udf(dataset=dataset, YourUDF())

Note the repeated :code:`-f` arguments - you can list any methods of your UDF you are
interested in, or even other Python functions you are directly or indirectly using. If you
had some complicated code in :code:`YourUDF.merge`, you would include :code:`-f YourUDF.merge`
in the :code:`%lprun` call.

This is how the output could look like:

.. code-block:: text

   Timer unit: 1e-06 s

   Total time: 0.002283 s
   File: /home/clausen/source/libertem/src/libertem/udf/holography.py
   Function: get_task_data at line 145

   Line #      Hits         Time  Per Hit   % Time  Line Contents
   ==============================================================
      145                                               def get_task_data(self):
      146                                                   """
      147                                                   Updates `task_data`
      148                                           
      149                                                   Returns
      150                                                   -------
      151                                                   kwargs : dict
      152                                                   A dictionary with the following keys:
      153                                                       kwargs['aperture'] : array-like
      154                                                       Side band filter aperture (mask)
      155                                                       kwargs['slice'] : slice
      156                                                       Slice for slicing FFT of the hologram
      157                                                   """
      158                                           
      159         2         48.0     24.0      2.1          out_shape = self.params.out_shape
      160         2         51.0     25.5      2.2          sy, sx = self.meta.partition_shape.sig
      161         2          5.0      2.5      0.2          oy, ox = out_shape
      162         2          7.0      3.5      0.3          f_sampling = (1. / oy, 1. / ox)
      163         2        292.0    146.0     12.8          sb_size = self.params.sb_size * np.mean(f_sampling)
      164         2        261.0    130.5     11.4          sb_smoothness = sb_size * self.params.sb_smoothness * np.mean(f_sampling)
      165                                           
      166         2       1172.0    586.0     51.3          f_freq = freq_array(out_shape)
      167         2        263.0    131.5     11.5          aperture = aperture_function(f_freq, sb_size, sb_smoothness)
      168                                           
      169         2         64.0     32.0      2.8          y_min = int(sy / 2 - oy / 2)
      170         2         37.0     18.5      1.6          y_max = int(sy / 2 + oy / 2)
      171         2         32.0     16.0      1.4          x_min = int(sx / 2 - ox / 2)
      172         2         30.0     15.0      1.3          x_max = int(sx / 2 + oy / 2)
      173         2          8.0      4.0      0.4          slice_fft = (slice(y_min, y_max), slice(x_min, x_max))
      174                                           
      175                                                   kwargs = {
      176         2          4.0      2.0      0.2              'aperture': aperture,
      177         2          6.0      3.0      0.3              'slice': slice_fft
      178                                                   }
      179         2          3.0      1.5      0.1          return kwargs

   Total time: 63.748 s
   File: /home/clausen/source/libertem/src/libertem/udf/holography.py
   Function: process_frame at line 181

   Line #      Hits         Time  Per Hit   % Time  Line Contents
   ==============================================================
      181                                               def process_frame(self, frame):
      182                                                   """
      183                                                   Reconstructs holograms outputting results into 'wave'
      184                                           
      185                                                   Parameters
      186                                                   ----------
      187                                                   frame
      188                                                      single frame (hologram) of the data
      189                                                   """
      190        16        154.0      9.6      0.0          if not self.params.precision:
      191                                                       frame = frame.astype(np.float32)
      192                                                   # size_x, size_y = self.params.out_shape
      193        16         81.0      5.1      0.0          frame_size = self.meta.partition_shape.sig
      194        16         58.0      3.6      0.0          sb_pos = self.params.sb_position
      195        16         66.0      4.1      0.0          aperture = self.task_data.aperture
      196        16         52.0      3.2      0.0          slice_fft = self.task_data.slice
      197                                           
      198        16   59291808.0 3705738.0     93.0          fft_frame = fft2(frame) / np.prod(frame_size)
      199        16    2189960.0 136872.5      3.4          fft_frame = np.roll(fft_frame, sb_pos, axis=(0, 1))
      200                                           
      201        16    2258700.0 141168.8      3.5          fft_frame = fftshift(fftshift(fft_frame)[slice_fft])
      202                                           
      203        16        816.0     51.0      0.0          fft_frame = fft_frame * aperture
      204                                           
      205        16       5957.0    372.3      0.0          wav = ifft2(fft_frame) * np.prod(frame_size)
      206        16        364.0     22.8      0.0          self.results.wave[:] = wav

Things to note:

 * :code:`get_task_data` takes a very small amount of time, compared to :code:`process_frame`. It does
   not make sense to concentrate on optimizing :code:`get_task_data` at all, in this case!
 * In :code:`process_frame`, the :code:`fft2` call takes up most time, so that is where
   we should direct our efforts. Improving, for example, the calls to :code:`fftshift` would give us
   a max speed-up of a few percent - and only, if we manage to dramatically improve their execution time!
 * `line_profiler` doesn't give information about individual expressions - sometimes you have to
   put expressions on their own line to see their individual contributions to the execution time. See
   the :code:`fft2` and :code:`np.prod` calls on the hottest line in the profile!
 * After successfully improving on the profiled times, always re-run with profiling disabled and without
   :class:`~libertem.executor.inline.InlineJobExecutor` and measure the total time, for example using
   :code:`%%time`. This makes sure that your optimizations actually work in a production environment!
 * The usual benchmarking rules apply - for example, try to run the profiling on an otherwise idle system,
   otherwise you can get noisy results.
 * Single-threaded execution can be quite slow compared to using LiberTEM in production - if it is too slow
   for your taste, you can run your UDF on a subset of your data using a :ref:`region of interest <udf roi>`.

.. seealso::

   `Python Data Science Handbook <https://jakevdp.github.io/PythonDataScienceHandbook/01.07-timing-and-profiling.html#Line-By-Line-Profiling-with-%lprun>`_
      The Python Data Science Handbook has a section on profiling and timing, including `line_profiler`.

   `Official documentation for line_profiler <https://github.com/rkern/line_profiler>`_
      All information on how to use `line_profiler`, including using it from different contexts.

   :ref:`Profiling long-running tests <profiling tests>`
      Information on how to profile the execution time of test cases.

   :ref:`Debugging UDFs`
      Using the :code:`InlineJobExecutor` to debug problems in your UDF.
