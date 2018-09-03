Plots
=====

.. _scanplot:

ScanPlot
--------

This is the plot that is displayed during the course of a scan. It shows magnetometry, susceptibility (in and out of phase), and cantilever capacitance data as a function of x,y scanner voltage in the units requested in the :ref:`measurementconfig` file. The plot is saved as a png file to the DataSet location after each line of the scan. The last five lines of data are displayed below the colorplot, with the most recent line in red.

  .. figure:: ../images/scanplot_example.png
      :scale: 70%

  .. automodule:: plots
      :members: ScanPlot

.. _tdcplot:

TDCPlot
--------

This is the plot that is displayed during a touchdown. It shows cantilever capacitance and susceptibility (in and out of phase) as a function of z scanner voltage in the units requested in the :ref:`measurementconfig` file. The plot is saved as a png file to the DataSet location at the end of the measurement.

  .. figure:: ../images/tdcplot_example.png
      :scale: 70%

  .. automodule:: plots
      :members: TDCPlot