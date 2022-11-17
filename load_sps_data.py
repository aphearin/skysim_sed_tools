"""Module loads SPS data used by DSPS to calculate galaxy SEDs and photometry"""
import os
from jax import numpy as jnp
from glob import glob
import numpy as np
import typing
from collections import OrderedDict


class FilterTrans(typing.NamedTuple):
    wave: jnp.ndarray
    transmission: jnp.ndarray


class SPSData(typing.NamedTuple):
    lgZsun_bin_mids: jnp.ndarray
    log_age_gyr: jnp.ndarray
    ssp_wave: jnp.ndarray
    ssp_spectra: jnp.ndarray


def load_filter_data(bandpat, drn):
    filter_fnames = glob(os.path.join(drn, bandpat + "*transmission.npy"))
    filter_data = OrderedDict()
    for fn in filter_fnames:
        filter_ndarray = np.load(fn)
        bn = os.path.basename(fn)
        pat = bn[: bn.find("_transmission")]
        filter_data[pat] = FilterTrans(
            filter_ndarray["wave"], filter_ndarray["transmission"]
        )
    return filter_data


def load_ssp_spectra(drn):
    """Load the SPS data used to calculate galaxy SED

    Parameters
    ----------
    drn : string
        directory storing SSP spectra

    Returns
    -------
    lgZsun_bin_mids : ndarray of shape (n_met, )
        grid of log10(Z/Zsun) associated with the SSP spectra

    log_age_gyr : ndarray of shape (n_age, )
        grid of stellar age log10(t_age/Gyr) associated with the SSP spectra

    ssp_wave : ndarray of shape (n_ssp_wave, )
        Wavelength in Angstroms at which the SSP SEDs are tabulated

    ssp_spectra : ndarray of shape (n_met, n_age, n_ssp_wave)
        grid of SSP spectra luminosities in Lsun/Hz

    """
    lgZsun_bin_mids = np.load(os.path.join(drn, "lgZsun_bin_mids.npy"))
    log_age_gyr = np.load(os.path.join(drn, "log_age_gyr.npy"))
    ssp_wave = np.load(os.path.join(drn, "ssp_wave.npy"))
    ssp_spectra = np.load(os.path.join(drn, "ssp_spectra.npy"))
    sps_data = SPSData(lgZsun_bin_mids, log_age_gyr, ssp_wave, ssp_spectra)
    return sps_data
