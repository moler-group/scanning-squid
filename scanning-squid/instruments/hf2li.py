import qcodes as qc
from qcodes.instrument.base import Instrument
import qcodes.utils.validators as vals
from typing import Dict, List, Optional, Sequence, Any, Union
import numpy as np

import zhinst.ziPython, zhinst.utils

import logging
log = logging.getLogger(__name__)

class HF2LI(Instrument):
	def __init__(self, name: str, device_serial: str, **kwargs) -> None:
		super().__init__(name, **kwargs)
		instr = zhinst.utils.create_api_session(device_serial, 1, required_devtype='HF2LI')
		self.daq, self.device_id, self.props = instr
		log.info('Successfully connected to {}.'.format(name))