Measurements
============

The measurements that have so far been implemented for :class:`microscope.Microscope` and :class:`microscope.SusceptometerMicroscope` include

- :ref:`capacitivetouchdowns`
- :ref:`approach`
- :ref:`scanning` at a given height over the plane of a sample

Touchdowns
----------

.. _capacitivetouchdowns:

Capacitive touchdowns
~~~~~~~~~~~~~~~~~~~~~
In a capacitive touchdown, we sweep the scanner height (:code:`scanner.position_z`) and measure the cantilever capacitance via capacitance bridge and the :code:`CAP_lockin`. If there is a change in the slope of capacitance as a function of DAQ AO voltage above some prescribed threshold (e.g. 20 fF/V)

.. _approach:

Approaching the sample
~~~~~~~~~~~~~~~~~~~~~~

Acquiring a Plane
~~~~~~~~~~~~~~~~~

.. TODO::
    Implement and document :code:`get_plane()`.

Susceptibility Touchdowns
~~~~~~~~~~~~~~~~~~~~~~~~~

.. TODO::
    Implement and document :code:`tc_susc()`.

.. _scanning:

Scanning
--------

See :ref:`/examples/ScanPlaneExample.ipynb` for a demonstration of scanning a plane with a :class:`microscope.SusceptometerMicroscope`.