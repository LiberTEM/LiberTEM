Workflow crystallographic characterization and phase identification
===================================================================

Look at frames, possibly identify and mark ROI

Determine parameters like camera length, wavelength  and convergence angle to determine size of diffraction blobs and transform pixel coordinates to lattice parameters in absolute units.

Single frame (point analysis), or application to whole scan!

Good idea to test on single frame first, or small ROI, because processing can take a bit longer

Run Karina's algo on it. Show where and with which quality blobs were identified.

Input value: Size of blobs, to be determined from illumination and projection.

Option: Shape of correlation mask.

Raw output: correlation function. Probably one sometimes wants to see that for a frame? Does it make sense to store or cache it? Or better cache the Fourier transform?

Derived output: Position, intensity and "position error bar" for list of peaks 
* better than user-defined threshold for some or all of these values.
* Fixed number of peak, in order of quality?

Identify and mark zero order peak. Should be nearly the same on each frame, except for descan error and small field-related shifts

Reduce grid to parallelogram lattice

Try to identify automatically. Allow manual intervention, i.e. selection of two base peaks. Snap to blob positions using correlation map

A frame can contain n different lattices

In case of image distortion: Determine distortion from position of peaks -- near misses. One frame is ALWAYS self-consistent (physics), and all frames should have nearly the same distortion. Main difference between frames is different zero order position from descan. Descan should be linear gradient across FOV (?) --> zero order peak shifts first fitted to linear gradient, or corrected with reference map. Extract differences for local fields.

Correction of distortions is very important, because otherwise the derived lattice parameters are not precise AND depend on peak order -- higher-order peaks are more outside, and therefore distorted differently.

Question: How stable are descan error, image distortion across settings and over time?

For phase identification, i.e. precise lattice parameters in absolute coordinates: Determine "true" calibrated coordinates for diffraction blobs. Check, for example, with silicon? Use external reference or internal reference within the scan?

Show lattice parameters within a matched frame, possibly with link to crystallography database.

Identify and color regions within the FOV with very similar lattice parameters -- populations. PCA? Set thresholds for "likeness?"

List view of populations with parameters. Clicking on an item highlights it. Populations can be blanked out, i.e. only intensity of populations that are not blanked out is shown in order to extract images of populations of special interest. 

Assign names to regions, for example within list view

Assign crystal structure and Burgers vectors to regions, for example within list view

Assign arbitrary colors (rsp. hues) to populations, for example within list view

Export the list as CSV

Identify populations by sending parameters to crystallography database -- ideally integrated by URL, or through copy and paste or manual entry.

Allow to re-calculate match by adjusting input parameters? I.e. keep populations, names, etc, just re-calculate their coordinates with a different calibration?

Allow color map of angle, rotation, abs(smaller vector), abs(larger vector), abs(small) / abs(large) to visualize stress fields, domain boundaries and such. Critical to limit coloring to population(s) or ROI to extract the subtle differences within a phase or grain which are small compared to differences between phases or grains.

Idea: Color to identify population, darkness/lightness to identify parameters? Plot in HSL?

Other workflow: Identify crystallinity of region
------------------------------------------------

Much simpler. Don't match positions, just presence of peaks!

Implementation plan
-------------------

Stage one:
~~~~~~~~~~

GUI: Show selected frame, raw convolution map on selected frame overlaid with visualized matched parameters. Purpose: Determine good parameters for fit, investigate problems with fitting, i.e. bad results in the list view

Data export (positions, intensity, quality) by scripting, with appropriate parameters

Then take it from there, unless we get a clearer picture. Having the list output through scripting, and the debug facilities, will help with developing the more challenging steps.

ALTERNATIVE
~~~~~~~~~~~

Is there a SW package that can already do good work with the list of peaks? In that case develop some form of transfer or integration!

