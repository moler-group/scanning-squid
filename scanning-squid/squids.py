"""
This file is part of the scanning-squid package.

Copyright (c) 2018 Logan Bishop-Van Horn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

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
