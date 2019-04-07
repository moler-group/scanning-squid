Measurements
============

The primary measurements for :class:`microscope.microscope.Microscope` and :class:`microscope.susceptometer.SusceptometerMicroscope` include

- :ref:`capacitivetouchdown`
- :ref:`approach`
- :ref:`getsurface`
- :ref:`scanning` at a given height over the surface of a sample

.. _capacitivetouchdown:

Capacitive touchdown
--------------------

    .. seealso:: :meth:`microscope.microscope.Microscope.td_cap`, :meth:`scanner.Scanner.check_for_td`, :meth:`scanner.Scanner.get_td_height`, and :class:`plots.TDCPlot`.

In a capacitive touchdown, we sweep the scanner z position (:class:`scanner.Scanner.position_z`) and measure the cantilever capacitance via a capacitance bridge and the :code:`CAP_lockin`. If there is a change in the slope of capacitance as a function of DAQ AO voltage above some prescribed threshold (e.g. 1 fF/V), a touchdown has been detected. The measurement parameters (usually loaded into an OrderedDict called :code:`tdc_params`) for a capacitive touchdown are defined in the :ref:`measurementconfig` file as follows:

    .. code-block:: json

        {
            "td_cap": {
                "fname": "td_cap",
                "dV": "0.1 V",
                "range": ["-9.5 V","9.5 V"],
                "channels": {
                    "CAP": {
                        "lockin": {
                            "name": "CAP",
                            "amplitude": "1 V",
                            "frequency": "6.281 kHz"
                        },
                        "label": "Capacitance",
                        "gain": 1,
                        "unit": "fF",
                        "unit_latex": "fF"
                    },
                    "SUSCX": {
                        "lockin": {
                            "name": "SUSC",
                            "amplitude": "1 V",
                            "frequency": "131.79 Hz"
                        },
                        "label": "Susceptibility",
                        "gain": 10,
                        "r_lead": "1 kOhm",
                        "unit": "Phi0/A",
                        "unit_latex": "$\\Phi_0$/A"
                    },
                    "SUSCY": {
                        "lockin": {
                            "name": "SUSC"
                        },
                        "label": "Susceptibility (out of phase)",
                        "gain": 10,
                        "r_lead": "1 kOhm",
                        "unit": "Phi0/A",
                        "unit_latex": "$\\Phi_0$/A"
                    }
                },
                "constants": {
                    "max_slope": "0.8 fF/V",
                    "max_delta_cap": "5 fF",
                    "initial_cap":"0 pF",
                    "nfitmin":10,
                    "nwindow":30,
                    "ntest":8,
                    "wait_factor":2
                }
            }
        }

The algorithm for performing a capacitive touchdown is as follows:

    1. Sweep :code:`scanner.position_z` through :code:`tdc_params['range']` with DAQ voltage steps given by :code:`tdc_params['dV']` and use the DAQ to measure the X output of :code:`CAP_lockin`. After each change in DAQ AO voltage, allow the lockin to settle for :code:`max(CAP_lockin.time_constant(), SUSC_lockin.time_constant()) * tdc_params['constants']['wait_factor']`.
    2. If at any point the capacitance is greater than :code:`tdc_params['constants']['max_delta_cap']` (i.e. if the capacitance bridge is very unbalanced), or if the pre-touchdown slope is greater than :code:`tdc_params['constants']['max_slope']`, something has gone wrong, so abort the touchdown.
    3. Once :code:`tdc_params['constants']['nwindow']` points have been acquired, partition the last :code:`tdc_params['constants']['nwindow']` points into two subsets (with the boundary not lying within :code:`tdc_params['constants']['nfitmin']` of either end of the window). For each allowed partition boundary point, fit a line to each of the two subsets, and select the boundary point that minimizes the RMS of the fit residuals.
    4. If the absolute value of the difference in slope between the two best-fit lines exceeds :code:`tdc_params['constants']['max_slope']`, a touchdown has occurred.
    5. If a touchdown is detected, repeat the fitting routine in step 4 to find the touchdown point, and exit the loop.
    6. If no touchdown is detected over the whole :code:`tdc_params['range']`, exit the loop.

The :meth:`microscope.microscope.Microscope.td_cap` will break its :class:`qcodes.Loop` if either :code:`scanner.Scanner.break_loop` or :code:`scanner.Scanner.td_has_occurred` is :code:`True`. The former is set to :code:`True` if: any of the safety limits are exceeded, the touchdown is interrupted by the user, or a touchdown is detected. The latter is only set to :code:`True` if a touchdown is detected.

    .. note::
        Whenever :code:`scanner.Scanner.break_loop` is set to :code:`True`, the scanner will be retracted to the voltage prescribed by the microscope's temperature mode (:code:`'LT'` or :code:`'RT'`).

    .. note::
        It is very important to find a low-noise regime for the capacitance measurment in order to avoid false touchdowns or not detecting a real touchdown. It seems the most effective knob to turn in order fix noise problems is :code:`CAP_lockin.frequency`. In the Bluefors 3K system, scatter of < 1 fF is typical and acceptable.

.. _approach:

Approaching the Sample
----------------------

    .. seealso:: :ref:`/examples/ApproachGetSurfaceExample.ipynb`, :meth:`microscope.microscope.Microscope.approach` and :ref:`capacitivetouchdown`.

The initial approach of the sample is done by iteratively performing capacitive touchdowns and :meth:`instruments.atto.AttocubeController.step` towards the sample in the z direction until a touchdown is detected. The basic flow of :meth:`microscope.microscope.Microscope.approach` goes as follows:

    - Run :meth:`microscope.microscope.Microscope.td_cap` to see if the SQUID is already close to the sample.
    - If no touchdown is detected, while the :meth:`microscope.microscope.Microscope.td_cap` loop is not broken:

        - Perform the requested number of z Attocube steps towards the sample
        - Run :meth:`microscope.microscope.Microscope.td_cap`
    - If the loop was broken because a touchdown was detected, run :meth:`microscope.microscope.Microscope.td_cap` to confirm that a touchdown occurred.

.. _getsurface:

Acquiring a Surface
-------------------

    .. seealso:: :ref:`/examples/ApproachGetSurfaceExample.ipynb`, :meth:`utils.make_scan_grids`, :meth:`utils.make_xy_grids`, and :ref:`capacitivetouchdown`.

In order to scan, we must know where the sample surface is. To acquire a surface, we perform capacitive touchdowns on a grid of x, y positions and fit a plane to the measured touchdown heights. The resulting fit coefficients are stored in the dictionary :code:`scanner.Scanner.metadata['plane']`, which has keys :code:`'x'`, :code:`'y'`, and :code:`'z'`. The sample plane for given x and y grids is then given by:

    .. code-block:: python

        coeffs = scanner.Scanner.metadata['plane']
        sample_plane = x_grid * coeffs['x'] + y_grid * coeffs['y'] + coeffs['z']

This means that :code:`coeffs['x']` and :code:`coeffs['y']` are the x and y gradients of the sample plane in DAQ voltage units, and :code:`coeffs['z']` is the touchdown height at the origin :code:`[x_position, y_position] == [0, 0]`. To scan, say, :code:`0.5 V` above the sample surface, the z-axis scan grid is simply :code:`sample_plane - 0.5`.

    .. note:: The sample topography (i.e. touchdown voltage vs. x,y voltage) and plane are saved in a .mat file, and can be loaded into the program using :meth:`scanner.Scanner.load_surface`.

    .. note:: When you perform a touchdown at the origin :code:`[x_position, y_position] == [0, 0]`, :code:`scanner.Scanner.metadata['plane']` is automatically updated with the new touchdown voltage.

    .. note:: This plane is trusted until the Attocubes are moved by :meth:`atto.AttocubeController.step`, at which point :class:`atto.AttocubeController.surface_is_current` is set to :code:`False`, and you will not be able to scan until you've acquired a new plane or manually set :code:`atto.surface_is_current = True`.

For samples that are not flat and therefore not well-approximated by a plane, there is the option to instead scan parallel to a surface formed by interpolating the touchdown points, by setting :code:`"surface_type": "surface"` in the :ref:`measurementconfig` file. The :class:`scanner.Scanner.surface_interp` object is an instance of :class:`scipy.interpolate.Rbf`, which forms a radial basis function representation of multi-dimensional data (similar to spline interpolation, but more general). To see what the expected touchdown voltage at point :code:`x, y` is, one can simply run :code:`scanner.Scanner.surface_interp(x,y)`.

    .. warning:: Calculation of of the :code:`Rbf` representation of the scan array (array of voltages to be written to the DAQ AOs during a scan) is very memory intensive. If the DAQ sampling rate is too high or the scan is too large or slow, you will get a :code:`MemoryError`.

    .. warning:: It is easy to introduce measurement artifacts when scanning an interpolated surface, particularly for measurements that are very sensitive to SQUID-sample separation (e.g. local susceptibility). You should only use this functionality if you can be reasonably sure you are not introducing artifacts.


.. _scanning:

Scanning
--------

    .. seealso:: :class:`plots.ScanPlot`

    .. note:: When measuring susceptibility while scanning, it is very important to choose the susceptibility lockin frequency and scan parameters such that each pixel corresponds to an integer number of lockin periods, so as to avoid beating/aliasing effects.

See :ref:`/examples/ScanSurfaceExample.ipynb` for a demonstration of scanning a plane with a :class:`microscope.susceptometer.SusceptometerMicroscope`.
