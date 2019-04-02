GSoC 2019 ideas
===============

LiberTEM is participating in the `Google Summer of Code <https://summerofcode.withgoogle.com/>`_ as a sub-organization of the `Python Software Foundation <http://python-gsoc.org/>`_. As a student, you can get paid by Google for three months, have fun working on an interesting open source software project, gain real-world development experience, and do something that looks nice on your CV!

* Check out our description and project ideas below
* Contact us if you'd like to work on LiberTEM
* Prepare a `proposal <http://python-gsoc.org/index.html#apply>`_ together with us
* You submit your application at the Google Summer of Code homepage to the Python Software Foundation organization, naming LiberTEM as the sub-organization.

Why LiberTEM
--------------

`LiberTEM <.>`_ is an open source platform for high-throughput distributed processing of pixelated scanning transmission electron microscopy (STEM) data. It is created to deal with the terabytes of data that modern high-speed high-resolution detectors for electron microscopy can produce. Our `architecture <architecture.html>`_ page describes in more detail how exactly it works.

..  figure:: ./images/Principle.png
    :scale: 50%
    :alt: In pixelated STEM, a full diffraction image is recorded for each scan position.

    *In pixelated STEM, a sample is scanned with a focused electron beam, and a full image of the transmitted beam is recorded for each scan position. The result is a four-dimensional data hypercube. This application can generate tremendous amounts of data from high-resolution scans with a high-speed high-resolution detector.*

The project started in the beginning of 2018 and is currently attracting more and more users because it is orders of magnitude faster and more scalable than established solutions, and at the same time convenient to use. In our field, LiberTEM is seen as the up and coming technology to tap the full scientific potential of modern high-speed detectors. Such detectors can already reach a data rate of 8.5 GB/s and will soon produce more than 50 GB/s of raw data. The conventional established PC-based solutions cannot keep up with such data rates, and distributed systems like LiberTEM are required for progress in this field.

Working on LiberTEM will give you experience in developing distributed systems for high-performance data processing with Python. You can learn how to profile an application and optimize performance in a targeted way. Our current records are an aggregate of 49 GB/s on `eight low-end microblade nodes <https://www.supermicro.com/products/system/3U/5038/SYS-5038MD-H8TRF.cfm>`_ reading from mass storage, and 21 GB/s on a single `high-end <https://ark.intel.com/content/www/us/en/ark/products/126793/intel-xeon-w-2195-processor-24-75m-cache-2-30-ghz.html>`_ workstation reading from the file system cache. LiberTEM has its roots in electron microscopy, but can be adopted for other tasks that involve high-throughput data-parallel processing of very large binary data sets.

..  figure:: ./images/Future.png
    :alt: Envisioned future architecture of LiberTEM

    *LiberTEM currently implements distributed offline data processing as shown on the right of this figure, and is designed to be extended to high-throughput distributed live data processing as illustrated on the left.*

If you work on our GUI, you'll learn how a responsive web application for big data analytics can be built with a front-end based on TypeScript, React and Redux, and an asynchronous Python back-end based on Tornado and dask.distributed.

Working on the application side will give you experience in Python-based big data analytics of large-scale binary data sets with a focus on imaging, physics and materials science with industry-leading throughput and efficiency.

About us
--------

Alex is an experienced software engineer, systems administrator, Python programmer, web developer, and expert on profiling and performance optimization. He focuses on the implementation side of LiberTEM. 

* https://github.com/sk1p


Dieter has an interdisciplinary background in materials science, computer science, product development, product management and business administration. He is mostly taking care of the application and business side of LiberTEM. 

* https://github.com/uellue
* https://www.facebook.com/uellue
* https://www.linkedin.com/in/uellue/

We'd be happy to share our experience with you!

How to reach us
---------------

The easiest path is our Gitter channel: https://gitter.im/LiberTEM/Lobby

E-Mail: `Dieter Weber <mailto:d.weber@fz-juelich.de>`_ `Alexander Clausen <mailto:a.clausen@fz-juelich.de>`_

Just drop a message! We are based in Germany (UTC+1 / UTC+2) and are generally active during the day.

Getting started
---------------

If you have questions, please ask freely: Supporting users and contributors has a high priority for us and your questions help us improve our documentation. Our development is currently moving very quickly. We are planning to complete our documentation when a few major features have left the prototype stage. For that reason it is always a good idea to be in touch directly.

Installation
~~~~~~~~~~~~

Please see `our documentation <https://libertem.github.io/LiberTEM/install.html>`_ for details! Forking our repository, cloning the fork and `installing the clone <https://libertem.github.io/LiberTEM/install.html#installing-from-a-git-clone>`_ are the recommended setup if you will be contributing significant amounts of code. Our `page on contributing <contributing.html>`_ has some -- still incomplete -- information that can help you get started with development. 

Currently, we are still working on getting suitable sample files online. Please contact us to get interesting sample data to work on!

What to work on
~~~~~~~~~~~~~~~

Our `issue tracker can give you a broad overview <https://github.com/LiberTEM/LiberTEM/issues>`_ of what we have on our plate. We've marked a number of `Good first issues <https://github.com/LiberTEM/LiberTEM/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22>`_ that might be relatively easy to solve and can help you get introduced to our code base. Further below we list a few specific ideas.


Writing your GSoC application
-----------------------------

The PYTHON GSOC website has a good overview for the application process: http://python-gsoc.org/index.html#apply http://python-gsoc.org/students.html with links to additional resources. Please remember to name the sub-organization *LiberTEM* in your proposal! 

To write your application, `Mailman <https://turnbull.sk.tsukuba.ac.jp/Blog/SPAM.txt>`_ gives a few general ideas. For us it is most important to know why you'd like to contribute to LiberTEM, what your experience level is and what you'd like to learn in order to find a good match for a project. Generally, we like targeted applications and applicants who contact us directly early-on.

We are happy to work with you for writing up a project idea! For Python GSoC it is a general requirement to already contribute a pull request to a sub-organization before submitting a project idea. Please contact us if you'd like some help with that! `Improving our examples <https://github.com/LiberTEM/LiberTEM/tree/master/examples>`_ in code, description and presentation would be both relatively easy and quite useful for us. The examples are currently lagging a bit behind recent changes in the `API <https://github.com/LiberTEM/LiberTEM/blob/master/src/libertem/api.py>`_. You could hunt down discrepancies and suggest updates. Please contact us for the corresponding data to run the examples!

Project Ideas
-------------

These are somewhat larger work items. Some of them can keep you busy for the entire time. Please feel free to suggest your own ideas as well! Just working on a number of smaller features and getting a good cross-sectional experience of LiberTEM can work as well.

1. Beginner/Intermediate/Advanced: Implement new analysis workflows or improve an existing one.
    We have a number of them from easy to hard on our waiting list. This can give you experience with the product development, design and application side of software engineering, and applied data science. A major part of the work is first figuring out *what* to implement together with our users, and then *how* to implement it. You can decide how far you take it: A detailed requirements document, a technical specification, a prototype, or a full production-grade implementation? All of that is useful for us.

    *Skills:* Communication, software development methodology, Python and numpy programming.
    
    *Domain knowledge:* Math, statistics, image processing and physics are of advantage.

    *Primary contact:* Dieter (@uellue)

2. Beginner/Intermediate/Advanced: Compression survey.
    Analyze high-throughput compression techniques, dive into lz4/zstd, blosc etc., compare against existing file formats. With this project you can improve your low-level programming experience: Instruction sets, CPU caches, optimizing compilers, auto-vectorization, and so on. Our favorite technology to do work in this area with Python is `numba <http://numba.pydata.org/>`_. Can be done basically independent of the LiberTEM codebase. For a beginner project you can compare existing implementations of common compression algorithms for our kind of data. For an advanced project you could test `autoencoders <https://en.wikipedia.org/wiki/Autoencoder>`_.

    *Skills:* Programming in C and Python, profiling.
    
    *Domain knowledge:* Good understanding how computers work in detail; neural networks for autoencoder.

    *Contact:* Dieter (@uellue), Alex (@sk1p)

3. Intermediate: `Explore automated benchmarks in detail <https://github.com/LiberTEM/LiberTEM/issues/198>`_.
    This will help us to catch performance regressions. In our experience, running a benchmark requires a reproducible, undisturbed environment and comparison to good reference data. For that reason we see it as more challenging than automated tests for functionality and correctness. You could run benchmarks in CI and observe variance, and record and present benchmark results over time.

    *Skills:* Programming, profiling, visualization.
    
    *Domain knowledge:* Continuous integration and automation tools.

    *Primary contact:* Alex (@sk1p)

4. Intermediate: `Editor for masks <https://github.com/LiberTEM/LiberTEM/issues/47>`_.
    Currently, the masks in the GUI are limited to a few simple shapes, while the back-end allows arbitrary masks. You could implement an online mask editor to give users more flexibility on designing masks. Part of the task would be a requirements analysis with experts for the scientific application, and an analysis if any existing code like http://fatiherikli.github.io/react-designer/ https://two.js.org/examples/ or http://fabricjs.com/controls-customization can possibly be used. This project would be mostly implemented in TypeScript.

    *Skills:* Programming in TypeScript, GUI development.
    
    *Domain knowledge:* --

    *Contact:* Dieter (@uellue), Alex (@sk1p)

5. Intermediate: Deploy LiberTEM with kubernetes.
    Help us set up a helm chart and documentation to deploy a LiberTEM cluster with kubernetes. The subject is fairly new to us and we'd appreciate your help, in particular if you already have experience with kubernetes.

    *Skills:* Systems administration and automation.
    
    *Domain knowledge:* kubernetes

    *Primary contact:* Alex (@sk1p)

6. Intermediate/Advanced: `Cloud caching layer <https://github.com/LiberTEM/LiberTEM/issues/136>`_.
    Since LiberTEM can achieve a staggering throughput with its standard analysis, reading data from network can quickly become a major bottleneck and create heavy load on any network-based storage system. We have started with the Hadoop File System for local storage on the nodes to avoid sending data through the network repeatedly, but that comes with a number of disadvantages. For that reason we'd like to include a transparent caching layer on the nodes that keeps portions of a data set in local SSD storage.

    *Skills:* Python and numpy programming, profiling. 
    
    *Domain knowledge:* --

    *Contact:* Dieter (@uellue), Alex (@sk1p)

7. Intermediate/Advanced: Proper schemas, validation and automatic form generation for analysis parameters.
    This feature will make it easier to implement new types of analysis in the GUI. This is a cross-section through Python and TypeScript, though we could also split off the more react-y part. Does not require numpy knowledge, or domain knowledge. Python/TypeScript required. General WebDev experience could help.

    *Skills:* Systematic thinking and abstraction, Python and TypeScript programming, web development. 
    
    *Domain knowledge:* --

    *Primary contact:* Alex (@sk1p)

8. Advanced: `Live visualization of large binary data <https://github.com/LiberTEM/LiberTEM/issues/134>`_.
    Basically an efficient/zoomable/user-friendly/fully-featured replacement for our visualization. Requires a cross-section of different technologies from Python/numpy/threading over HTTP/websockets to Canvas/WebGL. Could be spun off into its own project if it is successful!

    *Skills:* Python and TypeScript programming, web development, asynchronous and parallel programming, numerical processing, visualization. 
    
    *Domain knowledge:* Experience with similar projects and frameworks like for example `GR <https://gr-framework.org/>`_ desirable. Knowledge of `GIS <https://en.wikipedia.org/wiki/Geographic_information_system>`_ could potentially be useful.

    *Contact:* Dieter (@uellue), Alex (@sk1p)