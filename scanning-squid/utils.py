# This file is part of the scanning-squid package.
#
# Copyright (c) 2018 Logan Bishop-Van Horn
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import numpy as np
from typing import Dict, List, Optional, Sequence, Any, Union, Tuple, Callable
import qcodes as qc
from qcodes.instrument.parameter import ArrayParameter
from scipy import io
from collections import OrderedDict
import json

#: Tell the UnitRegistry what a Phi0 is, and that ohm and Ohm are the same thing.
with open('squid_units.txt', 'w') as f:
    f.write('Phi0 = 2.067833831e-15 * Wb\n')
    f.write('Ohm = ohm\n')

class Counter(object):
    """Simple counter used to keep track of progress in a Loop.
    """
    def __init__(self):
        self.count = 0
        
    def advance(self):
        self.count += 1

def load_json_ordered(filename: str) -> OrderedDict:
    """Loads json file as an ordered dict.
    Args:
        filename: Path to json file to be loaded.
    Returns:
        OrderedDict: odict
            OrderedDict containing data from json file.
    """
    with open(filename) as f:
        odict = json.load(f, object_pairs_hook=OrderedDict)
    return odict
        
def next_file_name(fpath: str, extension: str) -> str:
    """Appends an integer to fpath to create a unique file name:
        fpath + {next unused integer} + '.' + extension

    Args:
        fpath: Path to file you want to create (no extension).
        extension: Extension of file you want to create.

    Returns:
        str: next_file_name
            Unique file name starting with fpath and ending with extension.
    """
    i = 0
    while os.path.exists('{}{}.{}'.format(fpath, i, extension)):
        i += 1
    return '{}{}.{}'.format(fpath, i, extension)

def make_scan_vectors(scan_params: Dict[str, Any], ureg: Any) -> Dict[str, Sequence[float]]:
    """Creates x and y vectors for given scan parameters.

    Args:
        scan_params: Scan parameter dict
        ureg: pint UnitRegistry, manages units.

    Returns:
        Dict: scan_vectors
            {axis_name: axis_vector} for x, y axes.
    """
    Q_ = ureg.Quantity
    center = []
    size = []
    rng = []
    for ax in ['x', 'y']:
        center.append(Q_(scan_params['center'][ax]).to('V').magnitude)
        size.append(scan_params['scan_size'][ax])
        rng.append(Q_(scan_params['range'][ax]).to('V').magnitude)
    x = np.linspace(center[0] - 0.5 * rng[0], center[0] + 0.5 * rng[0], size[0])
    y = np.linspace(center[1] - 0.5 * rng[1], center[1] + 0.5 * rng[1], size[1])
    return {'x': x, 'y': y}

def make_scan_grids(scan_vectors: Dict[str, Sequence[float]], slow_ax: str,
                    fast_ax: str, fast_ax_pts: int, plane: Dict[str, float],
                    height: float) -> Dict[str, Any]:
    """Makes meshgrids of scanner positions to write to DAQ analog outputs.

    Args:
        scan_vectors: Dict of {axis_name: axis_vector} for x, y axes (from make_scan_vectors).
        slow_ax: Name of the scan slow axis ('x' or 'y').
        fast_ax: Name of the scan fast axis ('x' or 'y').
        fast_ax_pts: Number of points to write to DAQ analog outputs to scan fast axis.
        plane: Dict of x, y, z values defining the plane to scan (provided by scanner.get_plane).
        height: Height above the sample surface (in DAQ voltage) at which to scan.
            More negative means further from sample; 0 means 'in contact'.

    Returns:
        Dict: scan_grids
            {axis_name: axis_scan_grid} for x, y, z, axes.
    """
    slow_ax_vec = scan_vectors[slow_ax]
    fast_ax_vec = np.linspace(scan_vectors[fast_ax][0],
                              scan_vectors[fast_ax][-1],
                              fast_ax_pts)
    if fast_ax == 'y':
        X, Y = np.meshgrid(slow_ax_vec, fast_ax_vec, indexing='ij')
    else:
        X, Y = np.meshgrid(fast_ax_vec, slow_ax_vec, indexing='xy')
    Z = X * plane['x'] + Y * plane['y'] + plane['z'] + height
    return {'x': X, 'y': Y, 'z': Z}

def make_scan_surface(surface_type: str, scan_vectors: Dict[str, Sequence[float]], slow_ax: str,
                    fast_ax: str, fast_ax_pts: int, plane: Dict[str, float], height: float,
                    interpolator: Optional[Callable]=None):
    """Makes meshgrids of scanner positions to write to DAQ analog outputs.

    Args:
        surface_type: Either 'plane' or 'surface'.
        scan_vectors: Dict of {axis_name: axis_vector} for x, y axes (from make_scan_vectors).
        slow_ax: Name of the scan slow axis ('x' or 'y').
        fast_ax: Name of the scan fast axis ('x' or 'y').
        fast_ax_pts: Number of points to write to DAQ analog outputs to scan fast axis.
        plane: Dict of x, y, z values defining the plane to scan (provided by scanner.get_plane).
        height: Height above the sample surface (in DAQ voltage) at which to scan.
            More negative means further from sample; 0 means 'in contact'.
        interpolator: Instance of scipy.interpolate.Rbf used to interpolate touchdown points.
            Only required if surface_type == 'surface'. Default: None.

    Returns:
        Dict: scan_grids
            {axis_name: axis_scan_grid} for x, y, z, axes.
    """
    if surface_type.lower() not in ['plane', 'surface']:
        raise ValueError('surface_type must be "plane" or "surface".')
    plane_grids = make_scan_grids(scan_vectors, slow_ax, fast_ax, fast_ax_pts, plane, height)
    if surface_type.lower() == 'plane':
        return plane_grids
    else:
        if interpolator is None:
            msg = 'surface_type == "surface", so you must specify an instance of scipy.interpolate.Rbf'
            msg += '(namely microscope.scanner.surface_interp).'
            raise ValueError(msg)
        Z = interpolator(plane_grids['x'], plane_grids['y'])
        surface_grids = {'x': plane_grids['x'], 'y': plane_grids['y'], 'z': Z + height}
        return surface_grids

def make_xy_grids(scan_vectors: Dict[str, Sequence[float]], slow_ax: str,
                  fast_ax: str) -> Dict[str, Any]:
    """Makes meshgrids from x, y scan_vectors (used for plotting, etc.).

    Args:
        scan_vectors: Dict of {axis_name: axis_vector} for x, y axes (from make_scan_vectors).
        slow_ax: Name of scan slow axis ('x' or 'y').
        fast_ax: Name of scan fast axis ('x' or 'y').

    Returns:
        Dict: xy_grids
            {axis_name: axis_grid} for x, y axes.
    """
    slow_ax_vec = scan_vectors[slow_ax]
    fast_ax_vec = scan_vectors[fast_ax]
    if fast_ax == 'y':
        X, Y = np.meshgrid(slow_ax_vec, fast_ax_vec, indexing='ij')
    else:
        X, Y = np.meshgrid(fast_ax_vec, slow_ax_vec, indexing='xy')
    return {'x': X, 'y': Y}

def validate_scan_params(scanner_config: Dict[str, Any], scan_params: Dict[str, Any],
                         scan_grids: Dict[str, Any], pix_per_line: int, pts_per_line: int,
                         temp: str, ureg: Any, logger: Any) -> None:
    """Checks whether requested scan parameters are consistent with microscope limits.

    Args:
        scanner_config: Scanner configuration dict as defined in microscope configuration file.
        scan_params: Scan parameter dict as defined in measurements configuration file.
        scan_grids: Dict of x, y, z scan grids (from make_scan_grids).
        pix_per_line: Number of pixels per line of the scan.
        pts_per_line: Number of points per line sampled by the DAQ (to be averaged down to pix_per_line)
        temp: Temperature mode of the microscope ('LT' or 'RT').
        ureg: pint UnitRegistry, manages physical units.
        logger: Used to log the fact that the scan was validated.
    """
    Q_ = ureg.Quantity
    voltage_limits = scanner_config['voltage_limits'][temp]
    unit = scanner_config['voltage_limits']['unit']
    for ax in ['x', 'y', 'z']:
        limits = [(lim * ureg(unit)).to('V').magnitude for lim in voltage_limits[ax]]
        if np.max(scan_grids[ax]) < min(limits) or np.max(scan_grids[ax] > max(limits)):
            err = 'Requested {} axis position is outside of allowed range: {} V.'
            raise ValueError(err.format(ax, limits))        
    if pts_per_line % pix_per_line != 0:
        err = 'Current per-channel DAQ rate, line time, and number of channels '
        err += 'are incompatible with requested number of fast axis pixels. '
        err += 'Please adjust your scan parameters and/or DAQ rate.'
        raise ValueError(err)
    logger.info('Scan parameters are valid. Starting scan.')

def to_real_units(data_set: Any, ureg: Any=None) -> Any:
    """Converts DataSet arrays from DAQ voltage to real units using recorded metadata.
        Preserves shape of DataSet arrays.

    Args:
        data_set: qcodes DataSet created by Microscope.scan_plane
        ureg: Pint UnitRegistry. Default None.
        
    Returns:
        np.ndarray: data
            ndarray like the DataSet array, but in real units as prescribed by
            factors in DataSet metadata.
    """
    if ureg is None:
        from pint import UnitRegistry
        ureg = UnitRegistry()
        ureg.load_definitions('./squid_units.txt')
    meta = data_set.metadata['loop']['metadata']
    data = np.full_like(data_set.daq_ai_voltage, np.nan, dtype=np.double)
    for i, ch in enumerate(meta['channels'].keys()):
        array = data_set.daq_ai_voltage[:,i,:] * ureg('V')
        unit = meta['channels'][ch]['unit']
        data[:,i,:] = (array * ureg.Quantity(meta['prefactors'][ch])).to(unit)
    return data

def scan_to_arrays(scan_data: Any, ureg: Optional[Any]=None, real_units: Optional[bool]=True,
                   xy_unit: Optional[str]=None) -> Dict[str, Any]:
    """Extracts scan data from DataSet and converts to requested units.

    Args:
        scan_data: qcodes DataSet created by Microscope.scan_plane
        ureg: pint UnitRegistry, manages physical units.
        real_units: If True, converts z-axis data from DAQ voltage into
            units specified in measurement configuration file.
        xy_unit: String describing quantity with dimensions of length.
            If xy_unit is not None, scanner x, y DAQ ao voltage will be converted to xy_unit
            according to scanner constants defined in microscope configuration file.

    Returns:
        Dict: arrays
            Dict of x, y vectors and grids, and measured data in requested units.
    """
    if ureg is None:
        from pint import UnitRegistry
        ureg = UnitRegistry()
        #: Tell the UnitRegistry what a Phi0 is, and that ohm and Ohm are the same thing.
        with open('squid_units.txt', 'w') as f:
            f.write('Phi0 = 2.067833831e-15 * Wb\n')
            f.write('Ohm = ohm\n')
        ureg.load_definitions('./squid_units.txt')
    Q_ = ureg.Quantity
    meta = scan_data.metadata['loop']['metadata']
    scan_vectors = make_scan_vectors(meta, ureg)
    slow_ax = 'x' if meta['fast_ax'] == 'y' else 'y'
    grids = make_xy_grids(scan_vectors, slow_ax, meta['fast_ax'])
    arrays = {'X': grids['x'] * ureg('V'), 'Y': grids['y']* ureg('V')}
    arrays.update({'x': scan_vectors['x'] * ureg('V'), 'y': scan_vectors['y'] * ureg('V')})
    for ch, info in meta['channels'].items():
        array = scan_data.daq_ai_voltage[:,info['idx'],:] * ureg('V')
        if real_units:
            pre = meta['prefactors'][ch]
            arrays.update({ch: (Q_(pre) * array).to(info['unit'])})
        else:
            arrays.update({ch: array})
    if real_units and xy_unit is not None:
        bendc = scan_data.metadata['station']['instruments']['benders']['metadata']['constants']
        for ax in ['x', 'y']:
            grid = (grids[ax] * ureg('V') * Q_(bendc[ax])).to(xy_unit)
            vector = (scan_vectors[ax] * ureg('V') * Q_(bendc[ax])).to(xy_unit)
            arrays.update({ax.upper(): grid, ax: vector})
    return arrays

def td_to_arrays(td_data: Any, ureg: Optional[Any]=None, real_units: Optional[bool]=True) -> Dict[str, Any]:
    """Extracts scan data from DataSet and converts to requested units.

    Args:
        td_data: qcodes DataSet created by Microscope.td_cap
        ureg: pint UnitRegistry, manages physical units.
        real_units: If True, converts data from DAQ voltage into
            units specified in measurement configuration file.
    Returns:
        Dict: arrays
            Dict of measured data in requested units.
    """
    if ureg is None:
        from pint import UnitRegistry
        ureg = UnitRegistry()
        #: Tell the UnitRegistry what a Phi0 is, and that ohm and Ohm are the same thing.
        with open('squid_units.txt', 'w') as f:
            f.write('Phi0 = 2.067833831e-15 * Wb\n')
            f.write('Ohm = ohm\n')
        ureg.load_definitions('./squid_units.txt')
    Q_ = ureg.Quantity
    meta = td_data.metadata['loop']['metadata']
    h = [Q_(val).to('V').magnitude for val in meta['range']]
    dV = Q_(meta['dV']).to('V').magnitude
    heights = np.linspace(h[0], h[1], int((h[1]-h[0])/dV))
    arrays = {'height': heights * ureg('V')}
    for ch, info in meta['channels'].items():
        array = td_data.daq_ai_voltage[:,info['idx'],0] * ureg('V')
        if real_units:
            pre = meta['prefactors'][ch]
            arrays.update({ch: (Q_(pre) * array).to(info['unit'])})
        else:
            arrays.update({ch: array})
    return arrays

def scan_to_mat_file(scan_data: Any, real_units: Optional[bool]=True, xy_unit: Optional[bool]=None,
    fname: Optional[str]=None, interpolator: Optional[Callable]=None) -> None:
    """Export DataSet created by microscope.scan_surface to .mat file for analysis.

    Args:
        scan_data: qcodes DataSet created by Microscope.scan_plane
        real_units: If True, converts z-axis data from DAQ voltage into
            units specified in measurement configuration file.
        xy_unit: String describing quantity with dimensions of length.
            If xy_unit is not None, scanner x, y DAQ ao voltage will be converted to xy_unit
            according to scanner constants defined in microscope configuration file.
        fname: File name (without extension) for resulting .mat file.
            If None, uses the file name defined in measurement configuration file.
        interpolator: Instance of scipy.interpolate.Rbf, used to interpolate touchdown points.
            Default: None.
    """
    from pint import UnitRegistry
    ureg = UnitRegistry()
    ureg.load_definitions('./squid_units.txt')
    Q_ = ureg.Quantity
    meta = scan_data.metadata['loop']['metadata']
    arrays = scan_to_arrays(scan_data, ureg=ureg, real_units=real_units, xy_unit=xy_unit)
    mdict = {}
    for name, arr in arrays.items():
        if real_units:
            if xy_unit:
                unit = meta['channels'][name]['unit'] if name.lower() not in ['x', 'y'] else xy_unit
            else: 
                unit = meta['channels'][name]['unit'] if name.lower() not in ['x', 'y'] else 'V'
        else:
            unit = 'V'
        if meta['fast_ax'] == 'y':
            arr = arr.T
        mdict.update({name: {'array': arr.to(unit).magnitude, 'unit': unit}})
    if interpolator is not None:
        surf =  interpolator(arrays['X'], arrays['Y'])
        surf = surf if meta['fast_ax'] == 'x' else surf.T
        mdict.update({'surface': {'array': surf, 'unit': 'V'}})
    mdict.update({'prefactors': meta['prefactors'], 'location': scan_data.location})
    if fname is None:
        fname = meta['fname']
    fpath = scan_data.location + '/'
    io.savemat(next_file_name(fpath + fname, 'mat'), mdict)

def td_to_mat_file(td_data: Any, real_units: Optional[bool]=True, fname: Optional[str]=None) -> None:
    """Export DataSet created by microscope.td_cap to .mat file for analysis.

    Args:
        td_data: qcodes DataSet created by Microscope.td_cap
        real_units: If True, converts data from DAQ voltage into
            units specified in measurement configuration file.
        fname: File name (without extension) for resulting .mat file.
            If None, uses the file name defined in measurement configuration file.
    """
    from pint import UnitRegistry
    ureg = UnitRegistry()
    ureg.load_definitions('./squid_units.txt')
    Q_ = ureg.Quantity
    meta = td_data.metadata['loop']['metadata']
    arrays = td_to_arrays(td_data, ureg=ureg, real_units=real_units)
    mdict = {}
    for name, arr in arrays.items():
        if name is not 'height':
            unit = meta['channels'][name]['unit'] if real_units else 'V'
            mdict.update({name: {'array': arr.to(unit).magnitude, 'unit': unit}})
    mdict.update({'height': {'array': arrays['height'].to('V').magnitude, 'unit': 'V'}})
    mdict.update({
        'prefactors': meta['prefactors'],
        'location': td_data.location
        })
    try:
        mdict.update({'td_height': td_data.metadata['loop']['metadata']['td_height']})
    except KeyError:
        pass
    if fname is None:
        fname = meta['fname']
    fpath = td_data.location + '/'
    io.savemat(next_file_name(fpath + fname, 'mat'), mdict)

def moving_avg(x: Union[List, np.ndarray], y: Union[List, np.ndarray],
    window_width: int) -> Tuple[np.ndarray]:
    """Given 1D arrays x and y, calculates the moving average of y.

    Args:
        x: x data (1D array).
        y: y data to be averaged (1D array).
        window_width: Width of window over which to average.
        
    Returns:
        Tuple[np.ndarray]: x, ymvg_avg
            x data with ends trimmed according to width_width, moving average of y data
    """
    cs_vec = np.cumsum(np.insert(y, 0, 0))
    ymvg_avg = (cs_vec[window_width:] - cs_vec[:-window_width]) / window_width
    xs = int(np.ceil(window_width / 2)) - 1
    xf = -xs if window_width % 2 else -(xs + 1)
    return x[xs:xf], ymvg_avg

def fit_line(x: Union[list, np.ndarray], y: Union[list, np.ndarray]) -> Tuple[np.ndarray, float]:
    """Fits a line to x, y(x) and returns (polynomial_coeffs, rms_residual).

    Args:
        x: List or np.ndarry, independent variable.
        y: List or np.ndarry, dependent variable.

    Returns:
        Tuple[np.ndarray, float]: p, rms
            Array of best-fit polynomial coefficients, rms of residuals.
    """
    p, residuals, _, _, _ = np.polyfit(x, y, 1, full=True)
    rms = np.sqrt(np.mean(np.square(residuals)))
    return p, rms

def clear_artists(ax):
    for artist in ax.lines + ax.collections:
        artist.remove()
