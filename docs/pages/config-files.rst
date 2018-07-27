Configuration Files
===================

Parameters of both the microscope itself and of measurements it will perform are defined in `JSON <https://realpython.com/python-json/>`_ files. These parameters are loaded into memory as an `OrderedDict <https://docs.python.org/3/library/collections.html#collections.OrderedDict>`_ using :meth:`utils.load_json_ordered` so that they are accessible to the :class:`Microscope`.

.. _microscopeconfig:

Microscope Configuration
------------------------
Example configuration file for a :class:`microscope.susceptometer.SusceptometerMicroscope`.

    .. literalinclude:: ../examples/config_susceptometer.json
        :language: json

.. _measurementconfig:

Measurement Configuration
-------------------------
Example configuration file for :class:`microscope.susceptometer.SusceptometerMicroscope` measurements.

    .. literalinclude:: ../examples/config_measurements_susc.json
        :language: json