---
title: 'LiberTEM: Software platform for scalable multidimensional data processing in transmission electron microscopy'
tags:
  - Python
  - transmission electron microscopy
  - distributed
  - big data
  - MapReduce
authors:
  - name: Alexander Clausen
    orcid: 0000-0002-9555-7455
    affiliation: 1
  - name: Dieter Weber
    orcid: 0000-0001-6635-9567
    affiliation: 1
  - name: Karina Ruzaeva
    affiliation: 1
    orcid: 0000-0003-3610-0989
  - name: Vadim Migunov
    affiliation: "1, 3"
    orcid: 0000-0002-6296-4492
  - name: Jan Caron
    affiliation: 1
    orcid: 0000-0002-0873-889X
  - name: Rahul Chandra
    affiliation: 2
    orcid: 0000-0003-2079-5368
  - name: Magnus Nord
    affiliation: 4
    orcid: 0000-0001-7981-5293
  - name: Knut Müller-Caspary
    affiliation: 1
    orcid: 0000-0002-2588-7993
  - name: Rafal E. Dunin-Borkowski
    affiliation: 1
    orcid: 0000-0001-8082-0647
affiliations:
 - name: Forschungszentrum Jülich, Ernst Ruska-Centre for Microscopy and Spectroscopy with Electrons
   index: 1
 - name: Chandigarh University
   index: 2
 - name: Central Facility for Electron Microscopy (GFE), RWTH Aachen University
   index: 3
 - name: University of Antwerp, EMAT
   index: 4

date: 12 December 2019
bibliography: paper.bib
---

# Summary

Increases in the data rates of detectors for electron microscopy (EM) have
outpaced increases in network, mass storage and memory bandwidth by two orders
of magnitude between 2009 and 2019 [@Weber2018]. The LiberTEM open source
platform [@Clausen2020] is designed to match the growing performance
requirements of EM data processing [@Weber2020].

# Motivation

The data rate of the fastest detectors for electron microscopy that are
available in 2019 exceeds 50 GB/s, which is faster than the memory bandwidth of
typical personal computers (PCs) at this time. Applications from ten years
before that ran smoothly on a typical PC have evolved into numerical analysis of
complex multidimensional datasets [@Ophus2019] that require distributed
processing on high-performance systems. Furthermore, electron microscopy is
interactive and visual, and experiments performed inside electron microscopes
(so-called in situ experiments) often rely on fast on-line data processing as
the experimental parameters need to be adjusted based on the observation
results. As a consequence, modern data processing systems for electron
microscopy should be designed for very high throughput in combination with short
response times for interactive GUI use and closed-loop feedback. That requires
fundamental changes in the architecture and programming model, and consequently
in the implementation of algorithms and user interfaces for electron microscopy
applications.

# Description

The LiberTEM open source platform for high-throughput distributed processing of
large-scale binary data sets is developed to fulfill these demanding
requirements: Very high throughput on distributed systems, in combination with a
responsive, interactive interface. The current focus for application development
is electron microscopy. Nevertheless, LiberTEM is suitable for any kind of
large-scale binary data that has a hyper-rectangular array layout, notably data
from synchrotrons and neutron sources.

LiberTEM uses a simplified MapReduce [@Dean2008] programming model. It is
designed to run and perform well on PCs, single server nodes, clusters and cloud
services. On clusters it can use fast distributed local storage on
high-performance SSDs. That way it achieves very high aggregate IO performance
on a compact and cost-efficient system built from stock components. On a cluster
with eight microblade nodes we could show a mass storage throughput of 46 GB/s
for a virtual detector calculation.

LiberTEM is supported on Linux, Mac OS X and Windows. Other platforms that allow
installation of Python 3 and the required packages will likely work as well. The
GUI is running in a web browser.

Based on its processing architecture, LiberTEM offers implementations for
various applications of electron microscopy data. That includes basic
capabilities such as integrating over ranges of the input data (virtual
detectors and virtual darkfield imaging, for example), and advanced applications
such as data processing for strain mapping, amorphous materials and phase change
materials. More applications will be added as development progresses.

Compared to established MapReduce-like systems like Apache Spark [@Zaharia2016]
or Apache Hadoop [@Patel2012], it offers a data model that is similar to NumPy
[@Walt2011], suitable for typical binary data from area detectors, as opposed to
tabular data in the case of Spark and Hadoop. It includes interfaces to the
established Python-based numerical processing tools, supports a number of
relevant file formats for EM data, and features optimized data paths for
numerical data that eliminate unnecessary copies and allow cache-efficient
processing.

Compared to tools such as Dask arrays for NumPy-based distributed computations
[@Rocklin2015], LiberTEM is developed towards low-latency interactive feedback
for GUI display as well as future applications for high-throughput distributed
live data processing. As a consequence, data reduction operations in LiberTEM
are not defined as top-down operations like in the case of Dask arrays that are
then broken down into a graph of lower-level operations, but as explicitly
defined bottom-up streaming operations that work on small portions of the input
data. That way, LiberTEM can work efficiently on smaller data portions that fit
into the L3 cache of typical CPUs.

When compared to Dask arrays that try to emulate NumPy arrays as closely as
possible, the data and programming model of LiberTEM is more rigid and closely
linked to the way how the data is structured and how reduction operations are
performed in the back-end. That places restrictions on the implementation of
operations, but at the same time it is easier to understand, control and
optimize how a specific operation is executed, both in the back-end and in
user-defined operations. In particular, it is easier to implement complex
reductions in such a way that they are performed efficiently with a single pass
over the data.

The main focus in LiberTEM has been achieving very high throughput, responsive
GUI interaction and scalability for both offline and live data processing. These
requirements resulted in a distinctive way of handling data in combination with
a matching programming model. Ensuring compatibility and interoperability with
other solutions like [Gatan Microscopy Suite
(GMS)](http://www.gatan.com/products/tem-analysis/gatan-microscopy-suite-software),
Dask, HyperSpy [@Pena2019], pyXem [@Johnstone2019],
[pixStem](https://pixstem.org/) and others is work in progress. They use a
similar data model, which makes integration possible, in principle. As an
example, LiberTEM can be run from within development versions of an upcoming GMS
release that includes an embedded Python interpreter, and it can already generate
efficient Dask.distributed arrays from the data formats it supports.

The online documentation with more details on installation, use, architecture,
applications, supported formats, performance benchmarks and more can be found at
<https://libertem.github.io/LiberTEM/>.

Live data processing for interactive microscopy and automation of experiments is
currently under development. The architecture and programming model of LiberTEM
are already developed in such a way that current applications will work on live
data streams without modification as soon as back-end support is implemented.

# Acknowledgements

This project has received funding from the European Research Council (ERC) under
the European Union’s Horizon 2020 research and innovation programme (grant
agreements No 780487 - VIDEO and No 856538 - 3D MAGiC).

This project has received funding from the European Union’s Horizon 2020
research and innovation programme under grant agreements No 686053 - CritCat and
No 823717 – ESTEEM3.

We gratefully acknowledge funding from the Initiative and Networking Fund of the
Helmholtz Association within the Helmholtz Young Investigator Group moreSTEM
under Contract No. VH-NG-1317 at Forschungszentrum Jülich in Germany.

We gratefully acknowledge funding from the Information & Data Science Pilot
Project "Ptychography 4.0" of the Helmholtz Association.

We kindly acknowledge funding from Google Summer of Code 2019 under the umbrella
of the Python software foundation.

STEMx equipment and software for 4D STEM data acquisition with K2 IS camera
courtesy of Gatan Inc.

Forschungszentrum Jülich is supporting LiberTEM with funding for personnel,
access to its infrastructure and administrative support.

Furthermore, we wish to thank a large number of people who contributed in
various ways to LiberTEM. [We maintain a full and continuously updated list of creators and
contributors online.](https://libertem.github.io/LiberTEM/acknowledgments.html)

# References
