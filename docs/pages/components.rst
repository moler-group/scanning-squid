Microscope Components
=====================

A :class:`Microscope` is made up of several :class:`qcodes.Instrument` objects used to control and acquire data from physical instruments.

Attocubes
---------

The :class:`instruments.atto.AttocubeController` interfaces via GPIB with the Attocube hardware (e.g. an `ANC300 controller <http://www.attocube.com/attocontrol/motion-controllers/anc300/>`_). It enforces stepping voltage limits based on the current temperature mode (either :code:`'LT'` or :code:`'RT'`).

	.. automodule:: instruments.atto
		:members: AttocubeController, ANC300

Scanner
-------

The :class:`scanner.Scanner` represents the x, y, z scanner that controls the relative motion between the sample and the SQUID. It enforces voltage limits based on the current temperature mode (either :code:`'LT'` or :code:`'RT'`). A :class:`scanner.Scanner` instance creates and closes `nidaqmx <https://nidaqmx-python.readthedocs.io/en/latest/>`_ DAQ analog ouput and input tasks as needed to drive the scanner and sense its current position.

    .. automodule:: scanner
        :members:

SQUID
-----

The :class:`squids.SQUID` and subclasses like :class:`squids.Susceptometer` record SQUID parameters and metadata.

    .. automodule:: squids
        :members: SQUID, Susceptometer

DAQ
---

Instances of the :class:`instruments.daq.DAQAnalogInputs` instrument are created only as needed for a measurement, and removed once the measurement is completed. This ensures that the DAQ hardware resources are available when needed. A :class:`instruments.daq.DAQAnalogInputs` instrument has a single gettable parameter, :class:`instruments.daq.DAQAnalogInputVoltages`, which aqcuires a given number of samples from the requested DAQ analog input channels.

    .. automodule:: instruments.daq
        :members:

Others Instruments
------------------
Lockins
~~~~~~~
- SR830 driver courtesy of `QCoDeS <http://qcodes.github.io/Qcodes/>`_.

	.. automodule:: qcodes.instrument_drivers.stanford_research.SR830
		:members: SR830

- Driver for a single `Zurich Instruments HF2LI <https://www.zhinst.com/products/hf2li>`_ "lockin channel".

    .. automodule:: instruments.hf2li
        :members: HF2LI

Temperature Controllers
~~~~~~~~~~~~~~~~~~~~~~~
Lakeshore temperature controllers.

    .. automodule:: instruments.lakeshore
        :members: Model_331, Model_335, Model_340, Model_372

SourceMeters
~~~~~~~~~~~~
Keithley SourceMeters.

    .. automodule:: instruments.keithley
        :members: Keithley_2400

Arbitrary Function Generators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tektronix AFG3000 series.

    .. automodule:: instruments.afg3000
        :members: AFG3000

Digital Delay Generators
~~~~~~~~~~~~~~~~~~~~~~~~
Stanford Research DG645.

    .. automodule:: instruments.dg645
        :members: DG645

Heater Power Supply
~~~~~~~~~~~~~~~~~~~
AIM & Thurlby Thandar PSU (BlueFors warmup heater).

    .. automodule:: instruments.heater
        :members: EL320P