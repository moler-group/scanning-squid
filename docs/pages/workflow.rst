Typical Workflow
================

Preliminary Steps
-----------------

- Align SQUID at room temperature.
- Move the SQUID vertically far from the sample using the Attocubes.
- Cool down your fridge.
- Test and tune the SQUID once it is cold.
- Measure and record the scanner and Attocube capacitances.
- Check the cantilever capacitance.
- Create a directory on the data acquisition computer to hold all of the data and documentation for the cooldown.
- In this directory, create :ref:`microscopeconfig` and :ref:`measurementconfig` JSON files, then launch a Jupyter Notebook.
- In the Notebook, import any modules you'll need during the cooldown and add the scanning-squid repository to your path.

Initialize the Microscope
-------------------------

- Initialize the microscope from the :ref:`microscopeconfig` file.
- If something goes wrong, you can always restart the Jupyter Notebook kernel and/or re-initialize the microscope.

Load the Measurement Configuration
----------------------------------

- Load the :ref:`measurementconfig` file using :meth:`utils.load_json_ordered`.
- When you make changes to this file, be sure to re-load it.

Approach the Sample
-------------------

- See: :ref:`approach`.

  .. note::
      If the initial touchdown occurs at negative z scanner voltage, consider using the Attocubes to move such that touchdown occurs at a positive voltage. This way, if something goes wrong and the DAQ analog outputs go to 0 V, the SQUID will not be slammed into the sample.

  .. warning::
      This is the most dangerous/uncertain part of most measurements. If the capacitance is very noisy or the cantilever is not well-constructed, you risk not detecting the touchdown and crashing the SQUID into the sample using the Attocubes.

Acquire a Plane
---------------

- See: :ref:`getplane`

Scan Over the Plane
-------------------

- Define your scan parameters in the :ref:`measurementconfig` then reload the file using :meth:`utils.load_json_ordered`.
- Start :ref:`scanning`, sit back, and enjoy the :ref:`scanplot`!

Move Around the Sample
----------------------

- Use the :class:`instruments.atto.AttocubeController` to move around the sample, keeping in mind the angle between SQUID and sample so as not to accidentally crash.
- Unless the sample is very flat, it will be necessary to acquire a new plane after moving the Attocubes.
- If the sample is very flat and you still trust the old plane after moving the Attocubes, you can perform a single :ref:`capacitivetouchdown` at the origin and manually set :code:`atto.surface_is_current = True` to update the plane.