Workflow as a back-end
======================

General goal: Working should follow the familiar way of the software where LiberTEM is embedded. The tools of the SW package are available where possible, perhaps additional capabilities of LibertEM are exposed. Main difference to the user: Processing is FAST.

User connects to back-end, logs in with credentials --> LiberTEM cluster with IFF login

The user maps cloud storage into the back-end, possibly with other credentials or through a unified authentication mechanism --> IFFCloud with Linux credentials or Australian cloud

The user browses files in the various repositories. The user sees metadata (type, date, size, shape, format, application-specific info) and a preview of file contents, for example HAADF, BF, ADF or COM. The user gets an information about access speed -- for example traffic light, green for storage in Hadoop FS of cluster? One of the repositories can be the file system of the computer running the front-end -- ingest source, in particular!

The user can move data between repositories, in particular ingest data into the cluster (this could be slow and should be possible to kick off a day before working) --> Operations continue even if the session ends. That includes time-consuming calculations.

World of the future: If a repository is "remote processing enlightened", the user can deploy LiberTEM processing containers at the repository, i.e. set up a temporary LiberTEM cluster deployment "on demand" with a few clicks, so-to-speak.

The user can manage ongoing operations -- in particular cancel, perhaps pause. The user sees a progress bar and status info, such as expected duration, throughput.

The user opens a data set. The user sees a form of preview, similar to current web GUI? The user can browse the file contents, like current web GUI.

The user can modify the display with the familiar tools of the application, i.e. color map, scale bar etc. The metadata is available as if the application was working on a small native file locally.

The user can use as much as possible of the familiar tools of the SW. Likely candidates: Fourier transform, filter in Fourier space, cropping (rectangular or free-form?), line profile. Virtual detectors. What else? Long list! The user obtains a window that shows the result of the transformation. Only a reduced version is sent to the GUI! Possible reductions: Similar to current LibertEM Web GUI -- frame at position, averages, integrations. The user has GUI or menu controls for some of the reductions.

The user can step by step apply more transformations to the data, following the familiar pattern of the SW.

The user can save and open intermediate results as files, with full information of what when how who. The user can save in all available repositories, including the computer running the front-end. Formats include LibertEM-native cluster file and native formats of the front-end application.

If the result is small enough, the user can transition from working remotely to working locally. The user gets a hint when working locally makes sense, i.e. local processing would be so fast that round-trip times and cluster overhead start to dominate. Possibly, this transition is automatic. No matter if local or remote, the user can use save and load through local and remote storage.

The user can use the familiar scripting tools of the SW. Operations are executed remotely where it makes sense, ideally automatically. Possibly, the scripting facilities should be changed or get an additional extension to support building efficient pipelines. --> Possibly "Hyperspy 2.0" with REAL support for remote out-of-core processing with a future dask-numba-polly hybrid.This would be the "correct" way to implement Hyperspy functionality.

The user is informed if and why certain operations are not available as remote operations (YET). Including a contact to perhaps help with the implementation? If the remote operation has significantly less capabilities than local operation, the user is clearly informed that he/she is working remotely with reduced capabilities, but increased speed. Main job of the remote back-end: Get the data reduced to a size that can be handled locally; front-end for selected operations that are supported remotely.

The user is aware if processing incurs costs, in particular on rented infrastructure. The user has some form of budget control -- being aware of CPU and storage quotas, resource use etc. The user is alerted if an operation will likely exceed some form of quota -- not enough storage space, taking very long, etc. The user knows how much storage, how much time is expected.

Consequences
------------

There are likely three basic building blocks: Transformations that change the data, but result in a large result that should stay remote. Examples. Cropping, moderate binning, Fourier transform, coordinate transforms, Karinas blob tracing correlation, ...

Reductions process the data in such a way that it can be handled and displayed on the front-end. Examples: Virtual detectors, Karinas blob tracing coordinates or crystallinity figure of merit, PCA coloring, ...

Renderings transform the result of a reduction to a bitmap for display, i.e. color maps etc.

Types of application
--------------------

Extension of traditional software
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Replicate familiar behavior, use existing facilities --> rendering is definitely done by the front-end for the beginning. The back-end works towards emulating the same operations that the front-end would do on smaller local files. The user can work without the remote back-end altogether resp. transition between the two.

New style cloud application
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Re-think and optimize workflows for remote operation. Rendering likely done remotely because the transmitted images are smaller than the data, usually.

Ideas on implementation plan
----------------------------

Stage 1
~~~~~~~

Basic capabilities are remote file browser and exposing existing LiberTEM capabilities, namely applying masks for data reduction.

Workflow: User applies reduction step with desired parameters, continues working locally. File handling -- copy, ingest etc -- are done with other tools.

Stage 2
~~~~~~~

More and more complex capabilities are exposed, work towards a full set of available tools in the SW mirroring capabilities for small local files. Possibly implemented through sending processing functions to the back-end? Perhaps optimized dedicated versions where it makes sense?

Stage 3
~~~~~~~

Full-blown hybrid system with full capabilities to work remotely -- same or better as local version. Includes data management, upload to repositories and such