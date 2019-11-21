LiberTEM user-defined functions for distributed reduction operations on streaming detector data
===============================================================================================

Authors
-------

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

The data rate of detectors for electron microscopy has grown by TODO three
orders of magnitude between 2009 and 2019, while the throughput of IT components
such as CPU, memory, mass storage and network grew by only TODO one order of
magnitude, following exponential scaling laws like Moore’s law [@Weber2018].

PC-based solutions that were perfectly adequate in 2009 are no longer suitable
since the aggregate data rate from modern detectors can even exceed the memory
bandwidth of a typical PC and data analysis routines have evolved into numerical
analysis of complex multidimensional datasets [@Ophus2019].

Modern detectors consist of arrays of sub-detectors, of which each has their own
read-out electronics and connectors in order to process and transmit data in
parallel. Live feedback from the acquisition that they are currently performing
is a common requirement from electron microscopists since exploring a sample
with electron microscopy is a visual and interactive process. Live data
processing for detectors with parallel readout streams is a challenge with
conventional software because the data stream is not available as whole frames,
but distributed over several systems as partial frames. The data rate is so high
that combining the streams into one single stream is not practical, and current
distributed detectors like the Gatan K2IS camera can only dump data to storage
for subsequent offline analysis because of these difficulties in implementation.

For live display and human interaction, as well as many postprocessing steps,
the data will have to be reduced. For a significant number of reduction
algorithms, it is in principle possible to split up the main reduction operation
into separate reductions for each partial frame and then merge the partial
results. That includes, but is not limited to, any linear operation on the data.

That means that such an algorithm can, in principle, generate a reduced live
data stream from a distributed acquisition system that is suitable for live
display. Other algorithms can’t easily be split, or splitting would incur
significant performance penalties. That includes algorithms that calculate
Fourier transforms of whole frames, because it interleaves the data of a whole
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

Development process
-------------------

Since practicality and performance for real-world applications was a key
requirement, we co-developed the UDF API with an optimized implementation for
strain mapping, which is a major application in 4D STEM data analysis.

Optimized Strain mapping
........................

TODO merge with Karina's text

In a common approach for strain mapping, the approximate position of diffraction
disks or spots is determined for an entire dataset, then their actual position
is refined with subpixel accuracy in each frame, and finally a best fit for an
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

The performance of cross-correlation-based refinement can be boosted by a number of optimizations:

* Use fast correlation based on fast Fourier transforms and the correlation theorem TODO reference
* Re-use the Fourier transform of the template
* Limit the correlation to tight regions around the approximate peak positions since strain only leads to small shifts
* Use optimized FFT implementations and ensure a data layout with optimal alignment.

Since LiberTEM is based on Python, a number of Python-specific optimizations were applied as well:

* Optimized result buffer handling using larger arrays for many frames to avoid frequent allocation and garbage collection of small units of memory
* Minimize overheads by processing several peaks at a time in each step using array programming techniques, in combination with a block size that is optimized for L3 CPU cache efficiency.
* Targeted optimization of bottlenecks with Numba-based implementations where array programming is inefficient or complicated.

On the LiberTEM back-end side we handle parallelization, distribution and optimized input/output, which were already in place before developing the UDF API.


