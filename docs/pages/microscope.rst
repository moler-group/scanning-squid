Microscope 
==========

A physical scanning SQUID microscope is represented by an instance of the :class:`microscope.Microscope` class (or liklely one of its subclasses, like :class:`microscope.SusceptometerMicroscope`). A :class:`Microscope` is a :class:`qcodes.station.Station`, to which we can attach components (instances of :class:`qcodes.Instrument` or its subclasses) whose metadata we would like to save during a measurement.

	.. figure:: ../images/microscope.png
		:scale: 35%

See :ref:`/examples/ScanPlaneExample.ipynb` for a demonstration of scanning a plane with a :class:`microscope.SusceptometerMicroscope`.

    .. toctree::
        :maxdepth: 2

        components

    .. automodule:: microscope
        :members: Microscope, SusceptometerMicroscope

    .. TODO::
    	- Capacitive touchdowns.
    	- Susceptibility touchdowns.
    	- Add room temperature preamps (SR560).
    	- :class:`microscope.SamplerMicroscope`
    	- :class:`microscope.DispersiveMicroscope`
