Workflow for acquisition
========================

Full-blown version
------------------

LiberTEM works as an integral part of the detector system. Data storage like normal file storage, just with additional options where and how.

Connection to acquisition system is pre-configured, same as connection to raw detector -- authentication for acquisition system enough.

The user connects and authenticates to additional storage and processing facilities -- LiberTEM Hadoop cluster (iiflogin), IFFClouf (ifflongin), Australian cloud, ... through the microscope GUI

Fast mode for setting up microscope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

Minimal resolution, emulate STEM detectors or calculate COM absolute displacement -- works well for focusing. Support setting up beam by capturing frames of the beam as needed -- ronchigram, wobbling, ...

--> User interface can select to see high-speed virtual STEM or low-speed full camera stream on demand, as needed by the user workflow.

Two modes: Continuous or single frame (i.e. single full scan or single full frame.)

Workflow for setting up: User selects settings. SW sets up detector and LiberTEM according to settings. User hits "start scanning" or "single shot". LiberTEM handles the detector output and sends the reduced data stream to the front-end for display without saving anything. User modifies settings -- navigation, magnification, focus, astigmatism, ... until the result is right, while continuing acquiring resp. repeatedly requesting a new shot, switching modes as needed.

--> Requirement: Very fast mode change for detector and LiberTEM; very fast turn-around, possibly real-time, i.e.below 100 ms. Frame rate 60 Hz without flicker.

Summary: LiberTEM and detector emulate STEM detector and normal slow camera operation. USP LiberTEM: VERY fast mode change between STEM detector and detector frame, even both together. Possibility: high-speed detector scan 10,000-100,000 Hz stream together with averaged or sample frame 60 Hz stream.

Workflow for ad-hoc acquisition with pixelated STEM.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

User selects scan area, and perhaps slower mode, hits "acquire". (--> front-end sends metadata to LiberTEM: Scan resolution, area, microscope and detector settings, ...) While the data is coming in, the user can apply mask-based analysis and possibly other methods. --> LibertEM saves the file to a temporary file that allows re-analysis. Analysis similar as described in "Workflow as processing back-end.rst"? As a minimum, virtual STEM detectors and browsing the frames. Real-time feedback, real-time display of incoming data. 

The user can decide at any time to give the file a name and save it permanently. This should kick off transfer and possibly ingestion to the permanent storage location. After that seamless transition to "Workflow as processing back-end.rst"

Workflow for dedicated acquisition with pixelated STEM.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The user selects scan area, settings, storage location and file name schema. After hitting "acquire", the data is streamed directly to the storage location, possibly using the local system as a buffer. After that seamless transition to "Workflow as processing back-end.rst"

Additional requirements
~~~~~~~~~~~~~~~~~~~~~~~

Usually, there are buttons with scan presets available, at least in SEM: top speed continuous for navigation, medium or slow continuous for focusing, slow or top quality single scan for acquisition.

The ROI is changed frequently -- scan only on a small region for focusing to achieve high frame rate at good quality.

Reduced version
---------------

LiberTEM handles the camera only for acquisition. Focusing, navigation etc are done with the existing other detectors. Authentication and such are handled through LiberTEM Web GUI.

User sets up instrument and scan in the usual way. Selects ROI for acquisition. Moves pixelated detector in the beam. selects file name and such. Hits 'acquire', front-end sends info to LiberTEM. LiberTEM saves the data as requested and feeds requested monitoring stream to the front-end -- virtual detector and sample frames, possibly.

The user can browse and analyze the incoming file with the LiberTEM GUI, while the microscope GUI shows only the selected monitoring signal without possibility to analyze live.

Basic version
-------------

User sets up microscope and scan the usual way.

User selects ROI. User uses LibertEM web GUI to enter relevant parameters and clicks button for detector to wait for trigger. Authentication and such are handled through LiberTEM Web GUI.

User clicks "acquire" in microscope front-end. This triggers the detector acquisition. LiberTEM starts acquiring, saves the data and displays it in the web GUI based on entered data. User can browse, analyze, cancel data acquisition.

Scan is canceled in the microscope front-end.


Comments regarding implementation
---------------------------------

Two ways to handle detector-LiberTEM-Front-end relations:

1. The front-end controls microscope and camera. LiberTEM handles the data, mostly. Advantage: LiberTEM stays compact. Disadvantage: Detector support depends on front-end and LiberTEM.
2. The front-end controls microscope and LiberTEM. LiberTEM handles the camera and data. Advantage: Detector support depends only on LiberTEM, front-end implements universal LiberTEM driver and is done. Disadvantage: More code and complexity in LiberTEM, might have to replicate application-specific SW.

Likely solution: LiberTEM supports both ways. QD, for example, could be fully supported. K3 could be only data handling.

To keep in mind: Detector, scanning system and microscope need *hardware* synchronization through trigger signal. There are safety-related functions to handle. In particular moving detector in and out (danger of collision), and protecting detector from excessive beam intensity. Ways to misuse -- have phosphor screen in the way, for example. That should be immediately apparent if the user is in fromt od the microscope, but not necessarily for remote automated operation.

Metadata handling: Send data to LiberTEM in standardized format, save data directly in standardized format, save metadata in application-specific format?