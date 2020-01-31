LiberTEM user-defined functions for distributed reduction operations on streaming detector data
===============================================================================================

Authors
-------

TODO establish full author list like in JOSS paper following authorship policy

Alexander Clausen
.................

ORCID 0000-0002-9555-7455

Main developer of the architecture and implementation, critical review of the paper

Karina Ruzaeva
..............

ORCID 0000-0003-3610-0989

Optimized the correlation-based peak finding and ported it to the UDF API, wrote
sections of this paper related to correlation-based peak finding, critical
review

Dieter Weber
............

ORCID 0000-0001-6635-9567 

Corresponding author, d.weber@fz-juelich.de

Defined requirements and applications, testing, feedback and improvements for
UDF API, wrote most of the paper text.

Knut Müller-Caspary
...................

ORCID 0000-0002-2588-7993

Provided reference implementation for correlation-based peak finding, scientific
guidance for applications, critical review

Rafal Dunin-Borkowski
.....................

ORCID 0000-0001-8082-0647

Overall guidance, problem definition and feedback, critical review

Affiliation
-----------

Forschungszentrum Jülich
Ernst Ruska-Centre for Microscopy and Spectroscopy with Electrons
Jülich, Germany

Abstract
--------

The data rate of detectors for electron microscopy has surpassed the performance
improvements of essential IT components such as CPU, memory, mass storage and
network by two orders of magnitude over the last ten years :cite:`Weber2018`. In
response to these changing requirements we have developed a versatile API to
define high-performance distributed reduction operations on large-scale binary
data that is suitable for processing with interactive low-latency GUI
integration for both offline and live data.

Introduction and motivation
---------------------------

The data rate of detectors for electron microscopy has grown by three
orders of magnitude between 2009 and 2019, while the throughput of IT components
such as CPU, memory, mass storage and network grew by roughly one order of
magnitude, following exponential scaling laws like Moore’s law [@Weber2018].

PC-based solutions that were perfectly adequate in 2009 are no longer suitable
since the aggregate data rate from modern detectors can even exceed the memory
bandwidth of a typical PC and data analysis routines have evolved into numerical
analysis of complex multidimensional datasets [@Ophus2019].

Modern detectors consist of arrays of sub-detectors, of which each has their own
read-out electronics and connectors in order to process and transmit data in
parallel. Live feedback from the acquisition is a common requirement from
electron microscopists since exploring a sample with electron microscopy is a
visual and interactive process. Live data processing for detectors with parallel
readout streams is a challenge with conventional software since the data
stream is not available as whole frames, but distributed over several systems as
partial frames. The data rate is so high that combining the streams into one
single stream is not practical, and current distributed detectors like the Gatan
K2IS camera can only dump data to storage for subsequent offline analysis
because of these difficulties in implementation.

For live display and human interaction, as well as many postprocessing steps,
the data will have to be reduced. For a significant number of reduction
algorithms, it is in principle possible to split up the main reduction operation
into separate reductions for each partial frame and then merge the partial
results. That includes, but is not limited to, any linear operation on the data.

That means that such an algorithm can, in principle, generate a reduced live
data stream from a distributed acquisition system that is suitable for live
display. Other algorithms can’t easily be split, or splitting would incur
significant performance penalties. That includes algorithms that calculate
Fourier transforms of whole frames, since it interleaves the data of a whole
frame with itself.

A practical system to process such detector therefore requires means to express
operations in terms of partial reductions. At the same time it should include a
back-end that performs a shuffle operation to construct whole frames, i.e.
re-distribute the partial frames on the processing nodes in such a way that the
data of whole frames is combined in order to support algorithms that can’t
work efficiently on partial frames.

Algorithms that support processing of partial frames can
provide very smooth live data processing, while algorithms that require a
shuffle step may only work on larger units for block-wise processing.

Furthermore, an implementation that can reduce data in smaller portions and then
combine the partial reduction results is ideally suited for distributed offline
data processing to boost throughput. This concept was popularized with the
MapReduce programming model that has found widespread application in processing
tabular data TODO cite Spark and Hadoop.

These established systems can, in principle, process binary data, too. However,
their API provides only a limited choice of processing routines at the time of
writing, their IO capabilities for binary data are limited and their internal
data paths are not optimized to handle the extremely high throughput that modern
CPUs and GPUs can reach when processing numerical data.

In the case of Spark, the issue was deeply rooted in the way how the Java
virtual machine (JVM) passed numerical data between input files and external
libraries such as BLAS implementations at the time when we made the fundamental
design choices for LiberTEM. The JVM made several internal copies of input data,
which resulted in a more than four-fold performance penalty compared to a
zero-copy Python-based solution, which sadly excluded Spark from our application
[@Weber2020].

Solution
--------

The LiberTEM user-defined function (UDF) API was
developed to implement algorithms in such a way that they can process partial
data streams and data that is distributed over many nodes in a user-friendly
way. In particular, it separates the implementation of the algorithm from many
details of the back-end and data set, including the shape of the data and the
way how it is split up.

At the same time, it allows to implement very efficient
data paths and processing routines for numerical data that are a key requirement
to keep up with the immense data rates. A key focus in the development was
practicality, i.e. providing convenient interfaces to define both simple as well as
highly complex real-world reductions that gives users a lot of freedom to
implement their algorithms as needed. That includes offering opportunities for
performance optimization such as re-using intermediate data and buffers,
ensuring locality of reference and allowing loop nest optimization.

As a result, a LiberTEM UDF can run efficiently on a laptop, workstation or
cluster, and it can process both distributed live data streams and distributed
offline data. In each case, it can produce fine-grained live-updating results to
visualize a progressing calculation or display live data.

Optimized strain mapping as a lead application
----------------------------------------------

Since practicality and performance for real-world applications was a key
requirement, we co-developed the UDF API with an optimized implementation for
strain mapping, which is a major application in 4D STEM data analysis.

TODO merge with Karina's text

In a common approach for strain mapping, the positions of diffraction disks or
spots are refined with subpixel accuracy in each frame, and  a best fit for an
affine transformation from average to actual position of the spot is determined
that is an indication for the strain of the material.

The key reduction operation in this process is determining the position of
diffraction disks or spots in each frame with subpixel accuracy and precision,
leading to a data reduction by two to six orders of magnitude. We choose
convergent beam electron diffraction (CBED) as an application where the
convergent beam results in larger disks with internal intensity variations
rather than sharp diffraction peaks. Cross-correlation showed a favorable
combination of quality, robustness and performance in comparison to Hough
transforms and TODO in preliminary tests with real-world CBED patterns.

The performance of cross-correlation-based refinement can be boosted by a number
of optimizations:

  since the shifts are usually small.
* Use fast correlation based on fast Fourier transforms and the correlation
  theorem TODO reference.
* Re-use the Fourier transform of the template.
* Limit the correlation to tight regions around the peak position from an averaged frame
  since strain only leads to small peak shifts.
* Limit the analysis to tight regions of interest in the navigation space.
* Use optimized FFT implementations and ensure an optimal input data layout for that implementation.

Since LiberTEM is based on Python, a number of Python-specific optimizations were applied as well:

* Optimized result buffer handling using larger arrays for many frames to avoid frequent allocation
  and garbage collection of small units of memory
* Minimize overheads by processing several peaks at a time in each step using array programming
  techniques, in combination with a block size that is optimized for L3 CPU cache efficiency.
* Targeted optimization of bottlenecks with Numba-based implementations where array programming
  is inefficient or complicated.

On the LiberTEM back-end side we handle parallelization, distribution and
optimized input/output, which were already in place before developing the UDF
API TODO cite JOSS paper. Since some types of input data has to be decoded, for example the packed 12
bit integers of the K2 IS raw format, the data should be processed in chunks
that fit the CPU cache. A size of 1 MB has proven effective since many CPUs that
are used in numerical processing have at least 1 MB of L3 cache per core.

API design
----------

LiberTEM divides input data into partitions that can be processed independently
on many CPU cores and processing nodes. In a MapReduce context, this implies a
two-stage reduction: First, the data of a partition is reduced into a partial
result buffer on individual worker processes, and then the partial result
buffers are transferred to the central node and merged there.

Furthermore, the API should allow both very simple and highly complex
applications. That means many of its features are optional and will not
complicate any code that doesn't make use of them.

The API for LiberTEM UDFs is class-based to allow composition and extension
through object-oriented programming and to combine the various aspects of a UDF
into a self-contained package. Furthermore, using attributes of a UDF class to
pass parameters and meta information allows to make a rich portfolio of optional
features available to member functions without cluttering the interface for
simple applications that don't require them.

In order to allow the optimizations and features described above, the API offers
the following interfaces:

Context.map()
.............

Many basic operations such as calculating sums or other statistics on the data
can be expressed calling a function for each frame and creating a result array
with the same shape as the navigation dimension that contains the individual
results for each frame. The merge function is a simple assignment in this case.
LiberTEM UDFs support such an interface by determining the shape and type of
result buffers automatically by calling the function with a mock-up frame. This
interface is exposed through a simple map function that accepts a dataset and a
function as parameters and returns an array that matches the navigation
dimension of the dataset. The Blobfinder uses more advanced features of the
LiberTEM UDF API that go beyond a simple map().

get_result_buffers()
....................

The LiberTEM UDF API offers convenience functions for creating and assigning to
buffers with dimensions that match navigation or signal space. Furthermore,
buffers with custom dimensions that are not tied to the dataset are possible.
Buffers are defined by implementing the get_result_buffers() method that returns
a descriptor for all the required buffers. Based on this declaration, the UDF
back-end can allocate the appropriate buffers on the worker processes and on the
merging node(s). Strain mapping uses these capabilities, preserving signal space
for creating a sum or standard deviation map for finding approximate peak
positions, and preserving navigation space for the per-frame refinement of these
peak positions positions.

process_frame()
...............

Frame-by-frame processing is the simplest interface and allows to implement
algorithms that require full frames. It works by implementing the
process_frame() function (UDFFrameMixin) in an UDF class.  Behind the scenes,
the UDF back-end can assemble full frames from tiled datasets as required.
Single frames from smaller detectors often fit into the L3 cache, which means
this interface can be reasonably efficient. This is used for the default
Blobfinder implementation.

process_tile()
..............

This interface offers tiled processing, i.e. processing stacks of partial frames
from a contiguous region of navigation and signal space by implementing the
process_tile() method (UDFTileMixin) in an UDF class. This allows to benefit
from CPU caches through loop nest optimization, in particular when applying
signal space masks to large frames. This is used for an alternative
implementation of the correlation engine that relies on sparse matrix products.

merge()
.......

Overriding merge() in a UDF class allows to implement custom merge functions
that merge partial result buffers into the complete result. Reductions that
preserve navigation space can use the default merge() implementation, which is a
simple assignment. Reductions that preserve signal space or use buffers of type
"single" require a merge implementation that is appropriate for the application.

* Allow passing parameters for an analysis. The Blobfinder uses this to pass the list of
  expected peak positions and parameters of the template.
* 
* Allocate only those parts of navigation space buffers that are required for each partition.
* Allocate result buffers for a whole partition and assign output data directly
  to these buffers using views for the currently processed data portion. This
  avoids creating and discarding many small data structures.
* Allow specifying task data that is re-used for each data portion of a partition.
  The blobfinder keeps the Fourier transform of the correlation template and an
  intermediate buffer with special alignment for optimal PyFFTw performance.
* Allow specifying a ROI in navigation space where only values set to True in a
  masking Boolean array are calculated. The blobfinder uses this feature to restrict calculations only to relevant portions of a dataset.
* Offer preprocessing functions that are executed once for each partition.
  This can be used to allocate custom data structures in result buffers, such as
  lists for ragged arrays.
* Offer postprocessing functions that are executed once for each partition.
  This allows to implement two-stage reductions where the per-frame or per-tile
  function aggregates data into some of the result buffers, and postprocessing
  then performs an additional analysis or transformation step. This is
  particularly useful for per-tile processing if the second step requires
  complete data from all parts of each frame. The blobfinder uses this for an
  additional lattice refinement step after correlation.
* Allow extension and composition through object-oriented programming. The
  blobfinder allows arbitrary combinations of correlation and refinement
  implementation through this feature.
* Allow passing auxiliary data in navigation or signal space. This allows to set parameters
  for each frame or for each pixel in signal space individually. Advanced
  blobfinder applications use this feature to first cluster frames into separate
  classes and the aggregate results individually for each cluster class. Other
  applications can include per-frame intensity normalization using a beam
  current monitor when processing data from synchrotrons, or applying a detector
  dark frame and gain map "on the fly" without creating an intermediate
  corrected dataset.



The functions to allocate result buffers 
