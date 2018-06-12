import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import numpy as np
from utils import make_scan_vectors, make_scan_grids
from typing import Dict, List, Optional, Sequence, Any, Union

class ScanPlot(object):
    """Plot displaying acquired images in all measurement channels, updated live during a scan.
    """
    def __init__(self,  scan_params: Dict[str, Any], prefactors: Any,
                 ureg: Any, **kwargs) -> None:
        """
        Args:
            scan_params: Scan parameters as defined in measurement configuration file.
            prefactors: Dict of pint quantities defining conversion factor from
                DAQ voltage to real units for each measurement channel.
            ureg: pint UnitRegistry, manages units.
        """
        self.scan_params = scan_params
        self.ureg = ureg
        self.Q_ = ureg.Quantity
        self.prefactors = prefactors
        self.channels = scan_params['channels']
        self.fast_ax = scan_params['fast_ax']
        self.slow_ax = 'y' if self.fast_ax == 'x' else 'x'
        self.line_colors = ['#d80202' ,'#545454' ,'#777777' ,'#a8a8a8', '#d1d1d1']
        cols = len(self.channels.keys())
        rows = 3
        self.fig, self.ax = plt.subplots(rows, cols, figsize=(10,4.5), tight_layout=True,
                                         gridspec_kw={"height_ratios":[0.075, 1, 0.5]})
        self.fig.patch.set_alpha(1)
        self.plots = {'colorbars': {}, 'images': {}, 'lines': {}}
        i = 0
        for ch in self.channels.keys():
            self.plots['colorbars'].update({ch: {'cax': self.ax[0][i]}})
            self.plots['images'].update({ch: {'ax' :self.ax[1][i]}})
            self.plots['lines'].update({ch: self.ax[2][i]})
            i += 1
        for ch, ax in self.plots['images'].items():
            ax['ax'].set_aspect('equal')
            ax['ax'].set_xlabel('x position [V]')
            ax['ax'].set_ylabel('y position [V]')
        for ch, ax in self.plots['lines'].items():
            ax.set_aspect('auto')
            ax.set_xlabel('{} position [V]'.format(self.fast_ax))
        self.init_empty()
            
    def init_empty(self):
        """Initialize the plot with all images empty. They will be filled during the scan.
        """
        self.scan_vectors = make_scan_vectors(self.scan_params, self.ureg)
        self.X, self.Y = np.meshgrid(self.scan_vectors['x'], self.scan_vectors['y'])
        empty = np.full_like(self.X, np.nan, dtype=np.double)
        for ch in self.channels.keys():
            im = self.plots['images'][ch]['ax'].pcolormesh(self.X, self.Y, empty)
            self.plots['images'][ch].update({'quad': im})
            cbar = plt.colorbar(im, cax=self.plots['colorbars'][ch]['cax'], orientation='horizontal')
            self.plots['colorbars'][ch]['cax'].set_label(r'{}'.format(self.channels[ch]['unit_latex']))
            self.plots['colorbars'][ch].update({'cbar': cbar})
        for ax, ch in zip(self.ax[0], self.channels.keys()):
            ax.set_title(self.channels[ch]['label'])
            
    def update(self, data_set: Any, loop_counter: Any, num_lines: Optional[int]=5,
               offline: Optional[bool]=False) -> None:
        """Update the plot with updated DataSet. Called after each line of the scan.
        Args:
            DataSet: active data set, with a new line of data added with each loop iteration.
            loop_counter: utils.Counter instance, lets us know where we are in the scan.
            num_lines: Number of previous linecuts to plot, including the line just scanned.
                Currently can only handle num_lines <= 5.
            offline: False if this is being called during a scan.
        TODO: Add support for arbitrary num_lines.
        """
        self.location = data_set.location
        self.fig.suptitle(self.location, x=0.5, y=1) 
        data = self._to_real_units(data_set)
        meta = data_set.metadata['loop']['metadata']
        slow_ax = 'x' if meta['fast_ax'] == 'y' else 'y'
        line = loop_counter.count if not offline else meta['scan_size'][slow_ax] - 1
        for ch in self.channels.keys():
            idx = meta['channels'][ch]['ai']
            data_ch = data[:,idx,:]
            if self.fast_ax.lower() == 'y':
                data_ch = data_ch.T
            self._clear_artists(self.plots['images'][ch]['ax'])
            self._clear_artists(self.plots['lines'][ch])
            norm = colors.Normalize().autoscale(np.ma.masked_invalid(data_ch))
            self.plots['images'][ch]['quad'] = self.plots['images'][ch]['ax'].pcolormesh(self.X, self.Y, np.ma.masked_invalid(data_ch), norm=norm)
            self.plots['colorbars'][ch]['cbar'] = self.fig.colorbar(self.plots['images'][ch]['quad'],
                                                                    cax=self.plots['colorbars'][ch]['cax'],
                                                                    orientation='horizontal')
            self.plots['colorbars'][ch]['cbar'].locator = ticker.MaxNLocator(nbins=3)
            self.plots['colorbars'][ch]['cbar'].update_ticks()
            self.plots['colorbars'][ch]['cbar'].set_label(r'{}'.format(self.channels[ch]['unit_latex']))
            self.plots['colorbars'][ch]['cbar'].update_normal(self.plots['images'][ch]['quad'])
            self.plots['images'][ch]['ax'].relim()
            #self.plots['colorbars'][ch]['cax'].relim()
            self.plots['lines'][ch].relim()
            self.plots['colorbars'][ch]['cax'].minorticks_on()
            #: Update linecuts
            xdata = self.scan_vectors[self.fast_ax]
            if line < num_lines:
                for l in range(line+1):
                    ydata = data_ch[:,l] if self.fast_ax == 'y' else data_ch[l,:]
                    self.plots['lines'][ch].plot(xdata, ydata, lw=2, color=self.line_colors[line-l])
            else:
                for l in range(num_lines):
                    ydata = data_ch[:,line-num_lines+l+1] if self.fast_ax == 'y' else data_ch[line-num_lines+l+1,:]
                    self.plots['lines'][ch].plot(xdata, ydata, lw=2, color=self.line_colors[num_lines-l-1])
        self.fig.canvas.draw()
        self.fig.show()
        
    def save(self, fname=None):
        """Save plot to png file.
        Args:
            fname: File to which to save the plot.
                If fname is None, saves to data location as scan.png
        """
        if fname is None:
            fname = self.location + '/scan.png'
        plt.savefig(fname)
        
    def _to_real_units(self, data_set: Any) -> Any:
        """Converts DataSet arrays from DAQ voltage to real units using recorded metadata.
        Args:
            data_set: qcodes DataSet created by Microscope.scan_plane
        Returns:
            numpy ndarray like the DataSet array, but in real units as prescribed by
                factors in DataSet metadata.
        """
        data = np.full_like(data_set.daq_ai_voltage, np.nan, dtype=np.double)
        meta = data_set.metadata['loop']['metadata']
        for ch in self.channels.keys():
            idx = meta['channels'][ch]['ai']
            array = data_set.daq_ai_voltage[:,idx,:] * self.ureg('V')
            unit = self.scan_params['channels'][ch]['unit']
            data[:,idx,:] = (array * self.Q_(self.prefactors[ch])).to(unit)
        return data
    
    def _clear_artists(self, ax):
        """Clears lines and collections of lines from given matplotlib axis.
        Args:
            ax: axis to clear.
        """
        for artist in ax.lines + ax.collections:
            artist.remove()
            
class ScanPlotFromDataSet(ScanPlot):
    """Generate ScanPlot instance from a completed DataSet rather than during a Loop.
    """
    def __init__(self, scan_data: Any, ureg: Optional[Any]=None) -> None:
        """
        Args:
            scan_data: DataSet to plot, as created by microscope.scan_plane
            ureg: pint UnitRegistry, manages units.
        """
        if ureg is None:
            from pint import UnitRegistry
            ureg = UnitRegistry()
            #: Tell the UnitRegistry what a Phi0 is, and that ohm = Ohm
            with open('squid_units.txt', 'w') as f:
                f.write('Phi0 = 2.067833831e-15 * Wb\n')
                f.write('Ohm = ohm\n')
            ureg.load_definitions('./squid_units.txt')
        meta = scan_data.metadata['loop']['metadata']
        super().__init__(meta, meta['prefactors'], ureg)
        self.update(scan_data, None, offline=True)
