Microscope 
==========

A physical scanning SQUID microscope is represented by an instance of the :class:`microscope.microscope.Microscope` class (or liklely one of its subclasses, like :class:`microscope.susceptometer.SusceptometerMicroscope`). A :class:`Microscope` is a :class:`qcodes.station.Station`, to which we can attach components (instances of :class:`qcodes.Instrument` or its subclasses) whose metadata we would like to save during a measurement.

    .. figure:: ../images/microscope.jpg
        :scale: 35%

During a typical measurment (scan or capacitive touchdown), all settings/parameters of all instruments attached to the microscope are automatically queried and recorded, forming a "snapshot" of the microscope at the time of the measurement. This snapshot is saved along with a raw data file and a MATLAB .mat file containing data converted to real units. See :ref:`/examples/ScanSurfaceExample.ipynb` for a demonstration of scanning a sample surface with a :class:`microscope.susceptometer.SusceptometerMicroscope`.

    .. toctree::
        :maxdepth: 2

        components

    .. automodule:: microscope.microscope
        :members: Microscope

    .. automodule:: microscope.susceptometer
        :members: SusceptometerMicroscope

    .. automodule:: microscope.sampler
        :members: SamplerMicroscope
