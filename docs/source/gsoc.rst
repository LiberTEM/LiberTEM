GSoC 2019 ideas
===============

Why LiberTEM
--------------

`LiberTEM <https://libertem.github.io/LiberTEM/>`_ LiberTEM is an open source platform for high-throughput distributed processing of pixelated scanning transmission electron microscopy (STEM) data. It is created to deal with the terabytes of data that modern high-speed high-resolution detectors for electron microscopy can produce. 

It started in the beginning of 2018 and is currently attracting more and more users because it is orders of magnitude faster and more scalable than established solutions, and at the same time convenient to use. In our field, LiberTEM is seen as the leading technology to tap the full scientific potential modern high-speed detectors. Such detectors can already reach 8.5 GB/s and will soon produce more than 50 GB/s of raw data. LiberTEM has its roots in electron microscopy, but can be adopted for other tasks that involve high-throughput data-parallel processing of very large binary data sets.

Working on LiberTEM will give you experience in developing distributed systems for high-performance data processing with Python. You can learn how to do profiling and targeted performance optimization: Our current records are an aggregate of 49 GB/s on eight low-end microblade nodes reading from mass storage, and 21 GB/s on a single high-end workstation reading from the file system cache. 

If you work on our GUI, you'll learn how a responsive web application for big data analytics can be built with a front-end based on TypeScript, React and Redux, and an asynchronous Python back-end based on on Tornado and dask.distributed. 

Our team is experienced in software development methodology and tools, product management, product development, open source business models and strategy, and start-ups. We'd be happy to share our experience with you!

How to reach us
---------------

The easiest path is our Gitter channel: https://gitter.im/LiberTEM/Lobby

E-Mail: `Dieter Weber d.weber@fz-juelich.de <mailto:d.weber@fz-juelich.de>`_

Just drop a message! We are based in Germany and are generally active during the day. :-)

Getting Started
---------------

If you have questions, please ask freely: Supporting users and constributors has a high priority for us. Our development is currently moving very quickly. We are planning to complete our documentation when a few major features have left the prototype stage. Until then it is always a good idea to just ask if you are missing information.

Installation
~~~~~~~~~~~~

Please see `our documentation <https://libertem.github.io/LiberTEM/install.html>`_ for details! Forking our repository, cloning the fork and `installing the clone <https://libertem.github.io/LiberTEM/install.html#installing-from-a-git-clone>`_ are the recommended setup if you will be contributing significant amounts of code.

Currently, we are still working on getting suitable sample files online. Please contact us to get sample data!



What's the process for submitting your first bug fix?
Where should students look to find easy bugs to try out?
Writing your GSoC application
Links to advice about applications and the application template goes here.

Remind your students that your sub-org name must be in the title of their applications!
Here's a link to the student application information for Python
Project Ideas
You should usually have a couple of project ideas, ranging in difficulty from beginner to expert. Please do try to have at least one, preferably several beginner tasks: GSoC gets a lot of students with minimal open source experience who feel very discouraged (and sometimes even complain to Google) if orgs don't any have projects at their level.

1. Project name
Project description: Make sure you have a high-level description that any student can understand, as well as deeper details
Skills: programming languages? specific domain knowledge?
Difficulty level: Easy/Intermediate/Hard classification (students ask for this info frequently to help them narrow down their choices. Difficulty levels are something Google wants to see, so they aren't optional; make your best guess.)
Related Readings/Links: was there a mailing list discussion about this topic? standards you want the students to read first? bugs/feature requests?
Potential mentors: A list of mentors likely to be involved with this project, so students know who to look for on IRC/mailing lists if they have questions. (If you've had trouble with students overwhelming specific mentors, feel free to re-iterate here if students should contact the mailing list to reach all mentors.)
2. Project name
As above. etc. Unless there's a compelling reason to sort in some other order, ideas should be ordered approximately from easiest to hardest.