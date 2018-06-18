Measurements
============

The measurements that have so far been implemented for :class:`microscope.Microscope` and :class:`microscope.SusceptometerMicroscope` include

- :ref:`capacitivetouchdowns`
- :ref:`approach`
- :ref:`scanning` at a given height over the plane of a sample

Touchdowns
----------

    .. seealso:: :ref:`/examples/ApproachGetPlaneExample.ipynb`.

.. _capacitivetouchdowns:

Capacitive touchdowns
~~~~~~~~~~~~~~~~~~~~~

    .. seealso:: :meth:`microscope.Microscope.td_cap()`, :meth:`scanner.Scanner.check_for_td()`, :meth:`scanner.Scanner.get_td_height()`, and :class:`plots.TDCPlot`.

In a capacitive touchdown, we sweep the scanner height (:class:`scanner.Scanner.position_z`) and measure the cantilever capacitance via a capacitance bridge and the :code:`CAP_lockin`. If there is a change in the slope of capacitance as a function of DAQ AO voltage above some prescribed threshold (e.g. 20 fF/V), a touchdown has been detected. The measurement parameters (usually loaded into a dict called :code:`tdc_params`) for a capacitive touchdown are defined in the :ref:`measurementconfig` file as follows:

    .. code-block:: json

        {
            "td_cap": {
                "fname": "td_cap",
                "dV": "0.05 V",
                "range": ["-1.9 V","1.9 V"],
                "channels": {
                    "CAP": {
                        "lockin": {
                            "name": "CAP",
                            "amplitude": "1 V",
                            "frequency": "6.821 kHz"
                        },
                        "label": "Capacitance",
                        "gain": 1,
                        "unit": "fF",
                        "unit_latex": "fF"
                    }
                },
                "constants": {
                    "max_slope": "20 fF/V",
                    "max_delta_cap": "30 pF",
                    "initial_cap":"0 pF",
                    "nfitmin":10,
                    "nwindow":20,
                    "ntest":5,
                    "wait_factor":3
                }
            }
        }

The algorithm for performing a capacitive touchdown is as follows:

    1. Sweep :code:`scanner.position_z` through :code:`tdc_params['range']` with DAQ voltage steps given by :code:`tdc_params['dV']` and use the DAQ to measure the X output of :code:`CAP_lockin`. After each change in DAQ AO voltage, allow the lockin to settle for :code:`CAP_lockin.time_constant() * tdc_params['constants']['wait_factor']`.
    2. If at any point the capacitance is greater than :code:`tdc_params['constants']['max_delta_cap']` (i.e. if the capacitance bridge is very unbalanced), or if the pre-touchdown slope is greater than :code:`tdc_params['constants']['max_slope']`, something has gone wrong, so abort the touchdown.
    3. Once :code:`tdc_params['constants']['nwindow']` points have been acquired, partition the last :code:`tdc_params['constants']['nwindow']` points into two subsets (with the boundary not lying within :code:`tdc_params['constants']['nfitmin']` of either end of the window). For each allowed partition boundary point, fit a line to each of the two subsets, and select the boundary point that minimizes the RMS of the fit residuals.
    4. If the absolute value of the difference in slope between the two best-fit lines exceeds :code:`tdc_params['constants']['max_slope']`, a touchdown has occurred.
    5. If a touchdown is detected, repeat the fitting routine in step 4 to find the touchdown point, and exit the loop.
    6. If no touchdown is detected over the whole :code:`tdc_params['range']`, exit the loop.

The :meth:`microscope.Microscope.td_cap` will break its :class:`qcodes.Loop` if either :code:`scanner.Scanner.break_loop` or :code:`scanner.Scanner.td_has_occurred` is :code:`True`. The former is set to :code:`True` if: any of the safety limits are exceeded, the touchdown is interrupted by the user, or a touchdown is detected. The latter is only set to :code:`True` if a touchdown is detected.

    .. note::
        Whenever :code:`scanner.Scanner.break_loop` is set to :code:`True`, the scanner will be retracted to the voltage prescribed by the microscope's temperature mode (:code:`'LT'` or :code:`'RT'`).

    .. note::
        It is very important to find a low-noise regime for the capacitance measurment in order to avoid false touchdowns or not detecting a real touchdown. It seems the most effective knob to turn in order fix noise problems is :code:`CAP_lockin.frequency`. In the Bluefors 3K system, scatter of a few fF is typical and acceptable.

.. _approach:

Approaching the sample
~~~~~~~~~~~~~~~~~~~~~~

    .. seealso:: :meth:`microscope.Microscope.approach` and :ref:`capacitivetouchdowns`.

The initial approach of the sample is done by iteratively performing capacitive touchdowns and :meth:`atto.AttocubeController.step` towards the sample in the z direction until a touchdown is detected. The basic flow of :meth:`microscope.Microscope.approach` goes as follows:

    - Run :meth:`microscope.microscope.td_cap` to see if the SQUID is alread close to the sample.
    - If no touchdown is detected, while the :meth:`microscope.microscope.td_cap` loop is not broken:

        - Perform the requested number of z Attocube steps towards the sample
        - Run :meth:`microscope.microscope.td_cap`
    - If the loop was broken because a touchdown was detected, run :meth:`microscope.microscope.td_cap` to confirm that a touchdown occurred.

Acquiring a Plane
~~~~~~~~~~~~~~~~~

    .. seealso:: :meth:`utils.make_scan_grids`, :meth:`utils.make_xy_grids`, and :ref:`capacitivetouchdowns`.

In order to scan, we must know in what plane the sample lies. To acquire a plane, we perform capacitive touchdowns on a grid of x, y positions and fit a plane to the measured touchdown heights. The resulting fit coefficients are stored in the dictionary :code:`scanner.Scanner.metadata['plane']`, which has keys :code:`'x'`, :code:`'y'`, and :code:`'z'`. The sample plane for given x and y grids is then given by:

    .. code-block:: python

        coeffs = scanner.Scanner.metadata['plane']
        plane_grid = x_grid * coeffs['x'] + y_grid * coeffs['y'] + coeffs['z']

This means that :code:`coeffs['z']` is the touchdown height at the origin :code:`[x_position, y_position] == [0, 0]`. To scan, say, :code:`0.5 V` above the sample surface, the z-axis scan grid is simply :code:`plane_grid - 0.5`.

    .. note:: This plane is trusted until the Attocubes are moved by :meth:`atto.AttocubeController.step`, at which point :class:`atto.AttocubeController.plane_is_current` is set to :code:`False`, and you will not be able to scan until you've acquired a new plane.

Susceptibility Touchdowns
~~~~~~~~~~~~~~~~~~~~~~~~~

.. TODO::
    Implement and document :code:`td_susc()`.

.. _scanning:

Scanning
--------

    .. seealso:: :class:`plots.ScanPlot`

See :ref:`/examples/ScanPlaneExample.ipynb` for a demonstration of scanning a plane with a :class:`microscope.SusceptometerMicroscope`.