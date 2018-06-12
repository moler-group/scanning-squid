Microscope Components
=====================

A :class:`Microscope` is made up of several :class:`qcodes.Instrument` objects used to control and acquire data from physical instruments.

Attocubes
---------

The :class:`atto.AttocubeController` interfaces via GPIB with the Attocube hardware (e.g. an `ANC300 controller <http://www.attocube.com/attocontrol/motion-controllers/anc300/>`_).

	.. automodule:: atto
		:members: AttocubeController, ANC300

Scanner
-------

    .. automodule:: scanner
        :members:

SQUID
-----

    .. automodule:: squids
        :members: SQUID, Susceptometer

DAQ
---

    .. automodule:: daq
        :members:

Others
------
Lockins
~~~~~~~
SR830 driver courtesy of `QCoDeS <http://qcodes.github.io/Qcodes/>`_.

	.. automodule:: qcodes.instrument_drivers.stanford_research.SR830
		:members: SR830

Pre-amps
~~~~~~~~
SR560 driver courtesy of `QCoDeS <http://qcodes.github.io/Qcodes/>`_.

	.. automodule:: qcodes.instrument_drivers.stanford_research.SR560
		:members: SR560