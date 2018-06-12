from qcodes.instrument.base import Instrument
from typing import Dict, Any

class SQUID(Instrument):
    """SQUID sensor base class. Simply records sensor metadata.
    No gettable or settable parameters.
    """
    def __init__(self, squid_config: Dict[str, Any], **kwargs) -> None:
        """
        Args:
            squid_config: SQUID configuration dict.
                Simply added to instrument metadata.
            **kwargs: Keyword arguments passed to Instrument constructor.
        """
        super().__init__(squid_config['name'], **kwargs)
        self.metadata.update(squid_config)
    
    def clear_instances(self):
        for inst in self.instances():
            self.remove_instance(inst)

class Susceptometer(SQUID):
    """Records SQUID susceptometer metadata.
    """
    def __init__(self, squid_config: Dict[str, Any], **kwargs) -> None:
        """
        Args:
            squid_config: SQUID configuration dict.
                Simply added to instrument metadata.
            **kwargs: Keyword arguments passed to Instrument constructor.
        """
        super().__init__(squid_config, **kwargs)

class Sampler(SQUID):
    """Records sampler metadata.
    """
    def __init__(self, squid_config: Dict[str, Any], **kwargs) -> None:
        """
        Args:
            squid_config: SQUID configuration dict.
                Simply added to instrument metadata.
            **kwargs: Keyword arguments passed to Instrument constructor.
        """
        super().__init__(squid_config, **kwargs)

class Dispersive(SQUID):
    """Records dispersive SQUID metadata.
    """
    def __init__(self, squid_config: Dict[str, Any], **kwargs) -> None:
        """
        Args:
            squid_config: SQUID configuration dict.
                Simply added to instrument metadata.
            **kwargs: Keyword arguments passed to Instrument constructor.
        """
        super().__init__(squid_config, **kwargs)
