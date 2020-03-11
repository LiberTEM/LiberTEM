LiberTEM
========

**LiberTEM creators**: Alexander Clausen¹, Dieter Weber¹, Karina Ruzaeva¹, Vadim
Migunov¹², Jan Caron¹, Rahul Chandra³, Magnus Nord⁴, Colin Ophus⁵, Simon Peter,
Jay van Schyndel⁶, Jaeweon Shin⁷, Knut Müller-Caspary¹, Rafal Dunin-Borkowski¹

**Presentation**: Alexander Clausen¹, Dieter Weber¹

¹Jülich Research Centre, ²RWTH Aachen University, ³Chandigarh University,
⁴University of Antwerp, ⁵Lawrence Livermore National Lab, ⁶Monash University
eResearch Centre, ⁷ETH Zürich

Starting from 2009, increases in the data rates of TEM cameras have outpaced
increases in network, mass storage and memory bandwidth by two orders of
magnitude. This immense increase in performance opens new doors for applications
and, at the same time, requires adequate IT systems to handle acquisition,
storage and processing that go well beyond solutions based on personal
computers.

The LiberTEM open source platform is developed to address these challenges. It
allows high-throughput distributed processing of large-scale binary data sets
using a simplified MapReduce programming model. This programming model allows to
process live data streams and offline data with the same algorithms while
generating a stream of intermediate results for interactive GUI display. At the
same time, it scales to very high throughput. In a benchmark we have already
shown 46 GB/s on a microcloud cluster with eight nodes.

The current focus for application development is pixelated scanning transmission
electron microscopy (STEM) and scanning electron beam diffraction data.
Nevertheless, LiberTEM is suitable for processing any type of n-dimensional
numerical data.

LiberTEM can be used with a web-based GUI, via Python scripting, or embedded in
other applications as a data processing back-end. It is supported on Windows,
Linux and Mac OS X. Several typical TEM file formats are supported.

This contribution will first introduce the general architecture and interfaces,
demonstrate how it can be applied to typical tasks in electron microscopy, and
show examples for embedding in an application as a processing back-end.

More information on LiberTEM is available at
https://libertem.github.io/LiberTEM/index.html

Full continuously updated list of acknowledgments:
https://libertem.github.io/LiberTEM/acknowledgments.html
