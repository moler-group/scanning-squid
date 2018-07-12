#: Various Python utilities
import os
import sys
import time
import json
from typing import Dict, List, Sequence, Any, Union, Tuple

#: Plotting and math modules
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
import matplotlib.colors as colors
import numpy as np
from scipy.linalg import lstsq
from IPython.display import clear_output

#: Qcodes for running measurements and saving data
import qcodes as qc
from qcodes.station import Station
from qcodes.instrument_drivers.stanford_research.SR830 import SR830

#: NI DAQ library
import nidaqmx
from nidaqmx.constants import AcquisitionType

#: scanning-squid modules
import squids
import atto
import utils
from scanner import Scanner
from daq import DAQAnalogInputs
from plots import ScanPlot, TDCPlot
from .micropscope import Microscope

#: Pint for manipulating physical units
from pint import UnitRegistry
ureg = UnitRegistry()
#: Tell UnitRegistry instance what a Phi0 is, and that Ohm = ohm
with open('squid_units.txt', 'w') as f:
    f.write('Phi0 = 2.067833831e-15 * Wb\n')
    f.write('Ohm = ohm\n')
ureg.load_definitions('./squid_units.txt')

import logging
log = logging.getLogger(__name__)

class DispersiveMicroscope(Microscope):
    """Scanning dispersive SQUID microscope class.
    """
    def __init__(self, config_file: str, temp: str, ureg: Any=ureg, log_level: Any=logging.INFO,
                 log_name: str=None, **kwargs) -> None:
        super().__init__(config_file, temp, ureg, log_level, log_name, **kwargs)