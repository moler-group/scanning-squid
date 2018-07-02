from qcodes import VisaInstrument
import qcodes.utils.validators as vals
import logging
log = logging.getLogger(__name__)

class AFG3000(VisaInstrument):
    """Qcodes driver for Tektronix AFG3000 series arbitrary function generator.

    Not all instrument functionality is included here.
    Logan Bishop-Van Horn (2018)
    """
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator='\r\n', timeout=20, **kwargs)
        self.name = name

        #: Output parameters
        for out in [1, 2]:
            self.add_parameter('impedance_output{}'.format(out),
                       label='Output {} impedance'.format(out),
                       unit='',
                       get_cmd='OUTPut{}:IMPedance?'.format(out),
                       get_parser=str,
                       set_cmd='OUTPut{}:IMPedance {{}}'.format(out),
                       vals=vals.Strings()
                )
            self.add_parameter('polarity_output{}'.format(out),
                       label='Output {} polarity'.format(out),
                       unit='',
                       get_cmd='OUTPut{}:POLarity?'.format(out),
                       get_parser=str,
                       set_cmd='OUTPut{}:POLarity {{}}'.format(out),
                       vals=vals.Enum('NORMal', 'NORM', 'INVerted', 'INV')
                ) 
            self.add_parameter('state_output{}'.format(out),
                       label='Output {} state'.format(out),
                       unit='',
                       get_cmd='OUTPut{}:STATe?'.format(out),
                       get_parser=lambda x: bool(int(x)),
                       set_cmd='OUTPut{}:STATe {{}}'.format(out),
                       vals=vals.Enum('OFF', 0, 'ON', 1)
                )  
        self.add_parameter('trigger_mode',
                   label='Trigger mode',
                   unit='',
                   get_cmd='OUTPut:TRIGger:MODE?',
                   get_parser=str,
                   set_cmd='OUTPut:TRIGger:MODE {}',
                   vals=vals.Enum('TRIGger', 'TRIG', 'SYNC')
            )

        #: Source parameters
        for src in [1, 2]:
            #: Amplitude modulation
            self.add_parameter('am_depth{}'.format(src),
                       label='Source {} AM depth'.format(src),
                       unit='',
                       get_cmd='SOURce{}:AM:DEPTh?'.format(src),
                       get_parser=str,
                       set_cmd='SOURce{}:AM:DEPTh {{}}'.format(src),
                       vals=vals.Strings()
                )
            self.add_parameter('am_internal_freq{}'.format(src),
                       label='Source {} AM interal frequency'.format(src),
                       unit='',
                       get_cmd='SOURce{}:AM:INTernal:FREQuency?'.format(src),
                       get_parser=str,
                       set_cmd='SOURce{}:AM:INTernal:FREQuency {{}}'.format(src),
                       vals=vals.Strings()
                )              
            self.add_parameter('am_internal_function{}'.format(src),
                       label='Source {} AM interal function'.format(src),
                       unit='',
                       get_cmd='SOURce{}:AM:INTernal:FUNCtion?'.format(src),
                       get_parser=str,
                       set_cmd='SOURce{}:AM:INTernal:FUNCtion {{}}'.format(src),
                       vals=vals.Enum(
                        'SINusoid', 'SIN',
                        'SQUare',  'SQU',
                        'TRIangle', 'TRI',
                        'RAMP',
                        'NRAMp', 'NRAM',
                        'PRNoise', 'PRN',
                        'USER', 'USER1', 'USER2', 'USER3', 'USER4',
                        'EMEMory', 'EMEM',
                        'EFILe', 'EFIL')
                ) 
            self.add_parameter('am_internal_efile{}'.format(src),
                       label='Source {} AM interal EFile'.format(src),
                       unit='',
                       get_cmd='SOURce{}:AM:INTernal:FUNCtion:EFILe?'.format(src),
                       get_parser=str,
                       set_cmd='SOURce{}:AM:INTernal:FUNCtion:EFILe {{}}'.format(src),
                       vals=vals.Strings()
                )
            self.add_parameter('am_internal_source{}'.format(src),
                       label='Source {} AM source'.format(src),
                      f unit='',
                       get_cmd='SOURce{}:AM:SOURce?'.format(src),
                       get_parser=str,
                       set_cmd='SOURce{}:AM:SOURce? {{}}'.format(src),
                       vals=vals.Enum('INTernal', 'INT', 'EXTernal', 'EXT')
                )
            self.add_parameter('am_state{}'.format(src),
                       label='Source {} AM interal state'.format(src),
                       unit='',
                       get_cmd='SOURce{}:AM:STATe?'.format(src),
                       get_parser=lambda x: bool(int(x)),
                       set_cmd='SOURce{}:AM:STATe {{}}'.format(src),
                       vals=vals.Enum('OFF', 0, 'ON', 1)
                )
            #: Burst mode
            self.add_parameter('burst_mode{}'.format(src),
                       label='Source {} burst mode'.format(src),
                       unit='',
                       get_cmd='SOURce{}:BURSt:MODE?'.format(src),
                       get_parser=str,
                       set_cmd='SOURce{}:BURSt:MODE {{}}'.format(src),
                       vals=vals.Enum('TRIGgered', 'TRIG', 'GATed', 'GAT')
                )
            self.add_parameter('burst_ncycles{}'.format(src),
                       label='Source {} burst N cycles'.format(src),
                       unit='',
                       get_cmd='SOURce{}:BURSt:NCYCles?'.format(src),
                       get_parser=float,
                       set_cmd='SOURce{}:BURSt:NCYCles {{}}'.format(src),
                       vals=vals.MultiType(
                            vals.Ints(min_value=1, max_value=1000000),
                            vals.Enum('INFinity', 'INF', 'MAXimum', 'MAX', 'MINimum', 'MIN'))
                )
            self.add_parameter('burst_state{}'.format(src),
                       label='Source {} burst state'.format(src),
                       unit='',
                       get_cmd='SOURce{}:BURSt:STATe?'.format(src),
                       get_parser=lambda x: bool(int(x)),
                       set_cmd='SOURce{}:BURSt:STATe {{}}'.format(src),
                       vals=vals.Enum('OFF', 0, 'ON', 1)
                )
            self.add_parameter('burst_tdelay{}'.format(src),
                       label='Source {} burst time delay'.format(src),
                       unit='',
                       get_cmd='SOURce{}:BURSt:TDELay?'.format(src),
                       get_parser=str,
                       set_cmd='SOURce{}:BURSt:TDELay {{}}'.format(src),
                       vals=vals.Strings()
                )
            if src == 1:
                combine_enum = ('NOISe', 'NOIS', 'EXTernal', 'EXT', 'BOTH', '')
            else:
                combine_enum = ('NOISe', 'NOIS', '')
            self.add_parameter('combine{}'.format(src),
                       label='Source {} combine signals'.format(src),
                       unit='',
                       get_cmd='SOURce{}:COMBine:FEED ?'.format(src),
                       get_parser=str,
                       set_cmd='SOURce{}:COMBine:FEED {{}}'.format(src),
                       vals=vals.Enum(combine_enum)
                ) 

            #: Frequency modulation
            self.add_parameter('fm_deviation{}'.format(src),
                       label='Source {} FM deviation'.format(src),
                       unit='',
                       get_cmd='SOURce{}:FM:DEViation?'.format(src),
                       get_parser=str,
                       set_cmd='SOURce{}:FM:DEViation {{}}'.format(src),
                       vals=vals.Strings()
                )
            self.add_parameter('fm_internal_freq{}'.format(src),
                       label='Source {} FM interal frequency'.format(src),
                       unit='',
                       get_cmd='SOURce{}:FM:INTernal:FREQuency?'.format(src),
                       get_parser=str,
                       set_cmd='SOURce{}:FM:INTernal:FREQuency {{}}'.format(src),
                       vals=vals.Strings()
                )              
            self.add_parameter('fm_internal_function{}'.format(src),
                       label='Source {} FM interal function'.format(src),
                       unit='',
                       get_cmd='SOURce{}:FM:INTernal:FUNCtion?'.format(src),
                       get_parser=str,
                       set_cmd='SOURce{}:FM:INTernal:FUNCtion {{}}'.format(src),
                       vals=vals.Enum(
                        'SINusoid', 'SIN',
                        'SQUare',  'SQU',
                        'TRIangle', 'TRI',
                        'RAMP',
                        'NRAMp', 'NRAM',
                        'PRNoise', 'PRN',
                        'USER', 'USER1', 'USER2', 'USER3', 'USER4',
                        'EMEMory', 'EMEM',
                        'EFILe', 'EFIL')
                ) 
            self.add_parameter('fm_internal_efile{}'.format(src),
                       label='Source {} FM interal EFile'.format(src),
                       unit='',
                       get_cmd='SOURce{}:FM:INTernal:FUNCtion:EFILe?'.format(src),
                       get_parser=str,
                       set_cmd='SOURce{}:FM:INTernal:FUNCtion:EFILe {{}}'.format(src),
                       vals=vals.Strings(),
                       snapshot_get=False
                )
            self.add_parameter('fm_internal_source{}'.format(src),
                       label='Source {} FM source'.format(src),
                      f unit='',
                       get_cmd='SOURce{}:FM:SOURce?'.format(src),
                       get_parser=str,
                       set_cmd='SOURce{}:FM:SOURce? {{}}'.format(src),
                       vals=vals.Enum('INTernal', 'INT', 'EXTernal', 'EXT')
                )
            self.add_parameter('fm_state{}'.format(src),
                       label='Source {} FM interal state'.format(src),
                       unit='',
                       get_cmd='SOURce{}:FM:STATe?'.format(src),
                       get_parser=lambda x: bool(int(x)),
                       set_cmd='SOURce{}:FM:STATe {{}}'.format(src),
                       vals=vals.Enum('OFF', 0, 'ON', 1)
                ) 

            #: Frequency controls                 
            self.add_parameter('center_freq{}'.format(src),
                       label='Source {} center frequency'.format(src),
                       unit='',
                       get_cmd='SOURce{}:FREQuency:CENTer?'.format(src),
                       get_parser=str,
                       set_cmd='SOURce{}:FREQuency:CENTer {{}}'.format(src),
                       vals=vals.Strings()
                )
            self.add_parameter('concurrent_freq{}'.format(src),
                       label='Source {} concurrent frequency'.format(src),
                       unit='',
                       get_cmd='SOURce{}:FM::CONCurrent?'.format(src),
                       get_parser=lambda x: bool(int(x)),
                       set_cmd='SOURce{}:FM::CONCurrent {{}}'.format(src),
                       vals=vals.Enum('OFF', 0, 'ON', 1)
                ) 
            self.add_parameter('freq_cw{}'.format(src),
                       label='Source {} continuous frequency'.format(src),
                       unit='',
                       get_cmd='SOURce{}:FREQuency:CW?'.format(src),
                       get_parser=str,
                       set_cmd='SOURce{}:FREQuency:CW {{}}'.format(src),
                       vals=vals.Strings()
                )
            self.add_parameter('freq_fixed{}'.format(src),
                       label='Source {} fixed frequency'.format(src),
                       unit='',
                       get_cmd='SOURce{}:FREQuency:FIXed?'.format(src),
                       get_parser=str,
                       set_cmd='SOURce{}:FREQuency:FIXed {{}}'.format(src),
                       vals=vals.Strings()
                )
            self.add_parameter('freq_mode{}'.format(src),
                       label='Source {} frequency mode'.format(src),
                       unit='',
                       get_cmd='SOURce{}:FREQuency:MODE?'.format(src),
                       get_parser=str,
                       set_cmd='SOURce{}:FREQuency:MODE {{}}'.format(src),
                       vals=vals.Enum('CW', 'FIXed', 'FIX', 'SWEep', 'SWE')
                )
            self.add_parameter('freq_span{}'.format(src),
                       label='Source {} frequency span'.format(src),
                       unit='',
                       get_cmd='SOURce{}:FREQuency:SPAN?'.format(src),
                       get_parser=str,
                       set_cmd='SOURce{}:FREQuency:SPAN {{}}'.format(src),
                       vals=vals.Strings()
                )
            self.add_parameter('freq_start{}'.format(src),
                       label='Source {} frequency start'.format(src),
                       unit='',
                       get_cmd='SOURce{}:FREQuency:STARt?'.format(src),
                       get_parser=str,
                       set_cmd='SOURce{}:FREQuency:STARt {{}}'.format(src),
                       vals=vals.Strings()
                )
            self.add_parameter('freq_stop{}'.format(src),
                       label='Source {} frequency stop'.format(src),
                       unit='',
                       get_cmd='SOURce{}:FREQuency:STOP?'.format(src),
                       get_parser=str,
                       set_cmd='SOURce{}:FREQuency:STOP {{}}'.format(src),
                       vals=vals.Strings()
                )

    def calibrate(self):
        self.write('CALibration:ALL')

    def self_test(self):
        self.write('DIAGnostic:ALL')
        self.wait()

    def abort(self):
        self.write('ABORt')
        self.wait()

    def reset(self):
        log.info('Resetting {}.'.format(self.name))
        self.write('*RST')

    def save(self, location):
        if location not in [0, 1, 2, 3, 4]:
            raise ValueError('location must be in {}.'.format([0, 1, 2, 3, 4]))
        self.write('*SAVE {}'.format(location))

    def recall(self, location):
        if location not in [0, 1, 2, 3, 4]:
            raise ValueError('location must be in {}.'.format([0, 1, 2, 3, 4]))
        self.write('*RCL {}'.format(location))