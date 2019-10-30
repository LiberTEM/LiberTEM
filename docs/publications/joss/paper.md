---
title: 'LiberTEM: Open platform for pixelated scanning transmission electron microscopy'
tags:
  - Python
  - scanning transmission electron microscopy
  - distributed
  - big data
  - MapReduce
authors:
  - name: Dieter Weber
    orcid: 0000-0001-6635-9567
    affiliation: 1
  - name: Alexander Clausen
    orcid: 0000-0002-9555-7455
    affiliation: 1
  - name: Karina Ruzaeva
    affiliation: 1
    orcid: 0000-0003-3610-0989
  - name: Knut Müller-Caspary
    affiliation: 1
    orcid: 0000-0002-2588-7993
  - name: Rafal Dunin-Borkowski
    affiliation: 1
    orcid: 0000-0001-8082-0647
affiliations:
 - name: Forschungszentrum Jülich, Ernst Ruska-Centre for Microscopy and Spectroscopy with Electrons
   index: 1
date: 30 October 2019
bibliography: paper.bib
---

# Summary

LiberTEM [@Clausen2019] is an open source platform for high-throughput distributed processing
of large-scale binary data sets using a simplified MapReduce programming model.
The current focus is pixelated scanning transmission electron microscopy (STEM)
[@MacLaren2016] and scanning electron beam diffraction data.

It is designed for high throughput and scalability on PCs, single server nodes,
clusters and cloud services. On clusters it can use fast distributed local
storage on high-performance SSDs. That way it achieves very high aggregate IO
performance on a compact and cost-efficient system built from stock components.
Benchmarking results can be found at
https://libertem.github.io/LiberTEM/performance.html

LiberTEM is supported on Linux, Mac OS X and Windows. Other platforms that allow
installation of Python 3 and the required packages will likely work as well. The
GUI is running in a web browser.

LiberTEM offers implementations to process diffraction data for a number of
applications. That includes basic capabilities such as integrating over ranges
of the input data (virtual detectors and virtual darkfield imaging, for
example), and advanced applications such as data processing for strain mapping,
amorphous materials and phase change materials. More details can be found at
https://libertem.github.io/LiberTEM/applications.html

Compared to established MapReduce-like systems like Apache Spark [@Zaharia2016]
or Apache Hadoop [@Patel2012], it offers a data model that is more similar to
NumPy targeted at typical binary data from area detectors, as opposed to
tabular data in the case of Spark and Hadoop. It includes interfaces to the
established Python-based numerical processing tools, supports a number of
relevant file formats for EM data, and features optimized data paths for
numerical data that eliminate unnecessary copies and allow cache-efficient
processing. As a result, it can reach more than four times the throughput of
Spark-based processing for the same operations on typical data sets.

Compared to tools like Dask and Dask.Distributed for NumPy-based distributed
computations [@Rocklin2015], LiberTEM is developed towards low-latency
interactive feedback for GUI display as well as future applications for
high-throughput distributed live data processing. As a consequence, data
reduction operations in LiberTEM are not defined as top-down operations like in
the case of Dask arrays, but as bottom-up streaming operations that work on
small portions of the input data. Since LiberTEM can work efficiently on smaller
data portions that fit into the L3 cache of typical CPUs, it can be more than a factor of
two faster than equivalent implementations based on Dask arrays.

When compared to Dask arrays that try to emulate NumPy arrays as closely as
possible, the data and programming model of LiberTEM is more rigid and closer to
the way how the data is structured and how reduction operations are performed in
the back-end. That places restrictions on the implementation of reductions, but
at the same time it is easier to understand, control and optimize how a specific
operation is executed, both in the back-end and in user-defined operations. In
particular, it is easier to implement complex reductions in such a way that they
are performed efficiently with a single pass over the data.

The main focus in LiberTEM has been achieving very high throughput, responsive
GUI interaction and scalability for both offline and live data processing. These
requirements resulted in a distinctive way of handling data in combination with
an adapted programming model. Ensuring compatibility and interoperability with
other solutions like [Gatan Microscopy Suite
(GMS)](http://www.gatan.com/products/tem-analysis/gatan-microscopy-suite-software),
Hyperspy [@Pena2019], PyXem [@Johnstone2019], [Pixstem](https://pixstem.org/)
and others is work in progress. LiberTEM can be run from within development
versions of an upcoming GMS release that includes an embedded Python
interpreter. Furthermore, LiberTEM can generate efficient Dask.Distributed
arrays from the data formats it supports. That way, software packages that can
process Dask arrays such as Hyperspy, PyXem and Pixstem can, in principle, use
LiberTEM's high-performance distributed IO capabilities.

# Acknowledgements

This project has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No 780487).

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 686053.

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 823717 – ESTEEM3.

We kindly acknowledge funding from Google Summer of Code 2019 under the umbrella of the Python software foundation.

Forschungszentrum Jülich is supporting LiberTEM with funding for personnel, access to its infrastructure and administrative support.

Furthermore, we wish to thank a large number of people who contributed in various ways to LiberTEM. A full list of creators and contributors can be found at https://libertem.github.io/LiberTEM/acknowledgments.html

# References