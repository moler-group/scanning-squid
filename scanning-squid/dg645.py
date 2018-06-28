from qcodes import VisaInstrument
from qcodes.utils.validators import Strings, Enum
import logging
log = logging.getLogger(__name__)

class DG645(VisaInstrument):
	def __init__(self, name, address, **kwargs):
		super().__init__(name, address, terminator='\n', **kwargs)
		self.name = name
		self.channel_mapping = {
			'T0': 0, 'T1': 1, 'A': 2, 'B': 3, 'C': 4,
			'D': 5, 'E': 6, 'F': 7, 'G': 8, 'H': 9
		}
		self.display_mapping = {
			'trig_rate': 0,
			'trig_thresh': 1,
			'trig_single_shot': 2,
			'trig_line': 3,
			'advanced_trig_enable': 4,
			'trig_holdoff': 5,
			'prescale_config': 6,
			'burst_mode': 7,
			'burst: delay': 8,
			'burst_count': 9,
			'burst_period': 10,
			'channel_delay': 11,
			'channel_output_levels': 12,
			'channel_output_polarity': 13,
			'burst_T0_config': 14
		}
	def calibrate(self) -> str:
		response = self.ask('*CAL?')
		if int(response) == 0:
			log.info('Auto calibration successful.')
		elif int(response) == 17:
			log.info('Auto calibration failed.')
		return response

	def self_test(self) -> str:
		response = self.ask('*TST?')
		if int(response) == 0:
			log.info('Self test successful.')
		elif int(response) == 16:
			log.info('Self test failed.')
		return response

	def reset(self) -> None:
		log.info('Resetting {}.'.format(self.name))
		self.write('*RST')

	def save_settings(self, location: int) -> None:
		log.info('Saving instrument settings to location {}.'.format(location))
		self.write('*SAV {}'.format(location))

	def trigger(self) -> None:
		self.write('*TRG')

	def wait(self) -> None:
		self.write('*WAI')