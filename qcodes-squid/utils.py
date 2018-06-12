import os
import numpy as np
from typing import Dict, List, Optional, Sequence, Any, Union
import qcodes as qc
from scipy import io

class Counter(object):
    """Simple counter used to keep track of progress in a Loop.
    """
    def __init__(self):
        self.count = 0
        
    def advance(self):
        self.count += 1
        
def next_file_name(fpath: str, extension: str) -> str:
    """Appends an integer to fpath to create a unique file name:
        fpath + {next unused integer} + '.' + extension
    Args:
        fpath: Path to file you want to create (no extension).
        extension: Extension of file you want to create.
    Returns:
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
        Dict of {axis_name: axis_vector} for x, y axes.
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
        Dict of {axis_name: axis_scan_grid} for x, y, z, axes.
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

def make_xy_grids(scan_vectors: Dict[str, Sequence[float]], slow_ax: str,
                  fast_ax: str) -> Dict[str, Any]:
    """Makes meshgrids from x, y scan_vectors (used for plotting, etc.).
    Args:
        scan_vectors: Dict of {axis_name: axis_vector} for x, y axes (from make_scan_vectors).
        slow_ax: Name of scan slow axis ('x' or 'y').
        fast_ax: Name of scan fast axis ('x' or 'y').
    Returns:
        Dict of {axis_name: axis_grid} for x, y axes.
    """
    slow_ax_vec = scan_vectors[slow_ax]
    fast_ax_vec = scan_vectors[fast_ax]
    if fast_ax == 'y':
        X, Y = np.meshgrid(slow_ax_vec, fast_ax_vec, indexing='ij')
    else:
        X, Y = np.meshgrid(fast_ax_vec, slow_ax_vec, indexing='xy')
    return {'x': X, 'y': Y}

def validate_scan_params(scanner_config: Dict[str, Any], scan_params: Dict[str, Any],
                         scan_grids: Dict[str, Any], temp: str, ureg: Any,
                         logger: Any) -> None:
    """Checks whether requested scan parameters are consistent with microscope limits.
    Args:
        scanner_config: Scanner configuration dict as defined in microscope configuration file.
        scan_params: Scan parameter dict as defined in measurements configuration file.
        scan_grids: Dict of x, y, z scan grids (from make_scan_grids).
        temp: Temperature mode of the microscope ('LT' or 'RT').
        ureg: pint UnitRegistry, manages physical units.
        logger: Used to log the fact that the scan was validated.
    """
    Q_ = ureg.Quantity
    for axis in ['x', 'y', 'z']:
        limit = (Q_(scanner_config['voltage_limits'][temp][axis])
                 .to('V').magnitude)
        if np.max(np.abs(scan_grids[axis])) > limit:
            err = 'Requested {} axis position is outside of allowed range: +/- {} V.'
            raise ValueError(err.format(axis, limit))        
    x_pixels = scan_params['scan_size']['x']
    y_pixels = scan_params['scan_size']['y']
    logger.info('Scan parameters are valid. Starting scan.')
    
def to_arrays(scan_data: Any, ureg: Optional[Any]=None, real_units: Optional[bool]=True,
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
        array = scan_data.daq_ai_voltage[:,info['ai'],:] * ureg('V')
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

def scan_to_mat_file(scan_data: Any, real_units: Optional[bool]=True,
                     xy_unit: Optional[bool]=None, fname: Optional[str]=None) -> None:
    """Export DataSet created by microscope.scan_plane to .mat file for analysis.
    Args:
        scan_data: qcodes DataSet created by Microscope.scan_plane
        real_units: If True, converts z-axis data from DAQ voltage into
            units specified in measurement configuration file.
        xy_unit: String describing quantity with dimensions of length.
            If xy_unit is not None, scanner x, y DAQ ao voltage will be converted to xy_unit
            according to scanner constants defined in microscope configuration file.
        fname: File name (without extension) for resulting .mat file.
            If None, uses the file name defined in measurement configuration file.
    """
    from pint import UnitRegistry
    ureg = UnitRegistry()
    #: Tell the UnitRegistry what a Phi0 is, and that ohm and Ohm are the same thing.
    with open('squid_units.txt', 'w') as f:
        f.write('Phi0 = 2.067833831e-15 * Wb\n')
        f.write('Ohm = ohm\n')
    ureg.load_definitions('./squid_units.txt')
    Q_ = ureg.Quantity
    meta = scan_data.metadata['loop']['metadata']
    arrays = to_arrays(scan_data, ureg=ureg, real_units=real_units, xy_unit=xy_unit)
    mdict = {}
    for name, arr in arrays.items():
        if real_units:
            if xy_unit:
                unit = meta['channels'][name]['unit'] if name.lower() not in ['x', 'y'] else xy_unit
            else: 
                unit = meta['channels'][name]['unit'] if name.lower() not in ['x', 'y'] else 'V'
        else:
            unit = 'V'
        mdict.update({name: {'array': arr.to(unit).magnitude, 'unit': unit}})
    mdict.update({'prefactors': meta['prefactors'], 'location': scan_data.location})
    if fname is None:
        fname = meta['fname']
    fpath = scan_data.location + '/'
    io.savemat(next_file_name(fpath + fname, 'mat'), mdict)