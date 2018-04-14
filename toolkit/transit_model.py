from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import batman
from copy import deepcopy
import astropy.units as u

__all__ = ['transit_model_b', 'transit_model_b_depth_t0', 'params_b']

# Planet b:
params_b = batman.TransitParams()
params_b.per = 4.88780258
params_b.t0 = 2454605.89146
params_b.inc = 89.3470
params_b.rp = 0.0034**0.5

# a/rs = b/cosi
b = 0.141

ecosw = 0.261  # Winn et al. 2010
esinw = 0.085  # Winn et al. 2010
eccentricity = np.sqrt(ecosw**2 + esinw**2)
omega = np.degrees(np.arctan2(esinw, ecosw))

ecc_factor = (np.sqrt(1 - eccentricity**2) /
              (1 + eccentricity * np.sin(np.radians(omega))))

params_b.a = b / np.cos(np.radians(params_b.inc)) / ecc_factor
params_b.ecc = eccentricity
params_b.w = omega
params_b.u = [0.6373, 0.0554]  # Morris+ 2017a
params_b.limb_dark = 'quadratic'

params_b.depth_error = 0.00002
params_b.duration = 0.098 * u.day


def transit_model_b(times, params=params_b):
    """
    Get a transit model for TRAPPIST-1b at ``times``.

    Parameters
    ----------
    times : `~numpy.ndarray`
        Times in Julian date
    params : `~batman.TransitParams`
        Transiting planet parameters

    Returns
    -------
    flux : `numpy.ndarray`
        Fluxes at each time
    """
    m = batman.TransitModel(params, times)
    model = m.light_curve(params)
    return model


def transit_model_b_depth_t0(times, depth, t0, f0=1.0):
    """
    Get a transit model for TRAPPIST-1b at ``times``.

    Parameters
    ----------
    times : `~numpy.ndarray`
        Times in Julian date
    depth : float
        Depth of transit
    t0 : float
        Mid-transit time [JD]
    Returns
    -------
    flux : `numpy.ndarray`
        Fluxes at each time
    """
    params = deepcopy(params_b)
    params.t0 = t0
    params.rp = np.sqrt(depth)
    m = batman.TransitModel(params, times)
    model = f0*m.light_curve(params)
    return model
