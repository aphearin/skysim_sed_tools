"""This module provides functions that calculate the restframe SED of a synthetic galaxy
in SkySim5000 from its underlying model parameters.

"""
import numpy as np
from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap
from dsps.seds_from_tables import _calc_sed_kern
from dsps.mzr import mzr_model
from dsps.mzr import DEFAULT_MZR_PARAMS as DEFAULT_MZR_PARAM_DICT
from dsps.attenuation_kernels import sbl18_k_lambda, _flux_ratio, RV_C00
from dsps.utils import _jax_get_dt_array
from diffmah.individual_halo_assembly import (
    DEFAULT_MAH_PARAMS as DEFAULT_MAH_PARAM_DICT,
)
from diffmah.individual_halo_assembly import _calc_halo_history
from diffstar.stars import _sfr_history_from_mah


_calc_sed_vmap = jjit(vmap(_calc_sed_kern, in_axes=(*[None] * 5, *[0] * 2, None)))
_interp_vmap = jjit(vmap(jnp.interp, in_axes=[0, None, 0]))

DEFAULT_MZR_PARAMS = np.array(list(DEFAULT_MZR_PARAM_DICT.values()))[:-1]
DEFAULT_MDF_SCATTER = float(list(DEFAULT_MZR_PARAM_DICT.values())[-1])

MIN_SFR = 1e-7
T_MIN = 0.05  # Gyr
N_T = 100

UV_BUMP_W0 = 2175
UV_BUMP_DW = 350


DIFFMAH_KEYS = [
    "diffmah_logmp_fit",
    "diffmah_mah_logtc",
    "diffmah_early_index",
    "diffmah_late_index",
]
_ms_keylist = ("lgmcrit", "lgy_at_mcrit", "indx_lo", "indx_hi", "tau_dep")
DIFFSTAR_U_MS_KEYS = ["diffstar_u_" + key for key in _ms_keylist]
DIFFSTAR_U_Q_KEYS = ["diffstar_u_" + key for key in ("qt", "qs", "q_drop", "q_rejuv")]


def get_fit_params(
    data, mah_keys=DIFFMAH_KEYS, ms_keys=DIFFSTAR_U_MS_KEYS, q_keys=DIFFSTAR_U_Q_KEYS
):
    """Read the mock galaxy data table and return the diffmah and diffstar fit params

    Parameters
    ----------
    data : astropy Table of length (n_h, )
        Synthetic galaxy catalog

    Returns
    -------
    mah_params : ndarray of shape (n_h, 4)

    ms_u_params : ndarray of shape (n_h, 5)

    q_u_params : ndarray of shape (n_h, 4)

    """
    mah_params = np.array([data[key] for key in mah_keys]).T
    u_ms_params = np.array([data[key] for key in ms_keys]).T
    u_q_params = np.array([data[key] for key in q_keys]).T
    return mah_params, u_ms_params, u_q_params


@jjit
def _get_att_curve_kern(wave_angstrom, dust_params):
    """Calculate dust attenuation curve from model params

    Parameters
    ----------
    wave_angstrom : ndarray of shape (n, )

    dust_params : ndarray of shape (3, )
        dust_Eb, dust_delta, dust_Av = dust_params

    Returns
    -------
    att_curve : ndarray of shape (n, )
        Fractional reduction of the flux at the input wavelength

    """
    dust_Eb, dust_delta, dust_Av = dust_params

    wave_micron = wave_angstrom / 10_000
    dust_x0_microns = UV_BUMP_W0 / 10_000
    bump_width_microns = UV_BUMP_DW / 10_000

    axEbv = sbl18_k_lambda(
        wave_micron, dust_x0_microns, bump_width_microns, dust_Eb, dust_delta
    )
    att_curve = _flux_ratio(axEbv, RV_C00, dust_Av)

    return att_curve


@jjit
def _calc_rest_sed_single_diffstar_gal(
    t_obs,
    lgZsun_bin_mids,
    log_age_gyr,
    ssp_spectra,
    mah_params,
    u_ms_params,
    u_q_params,
    mzr_params=DEFAULT_MZR_PARAMS,
    lgmet_scatter=DEFAULT_MDF_SCATTER,
    lgt0=1.14,
):
    """Calculate restframe SED of an individual Diffstar galaxy

    Parameters
    ----------
    t_obs : float
        Age of the universe in Gyr at the redshift of the galaxy

    lgZsun_bin_mids : ndarray of shape (n_met, )
        grid of log10(Z/Zsun) associated with the SSP spectra

    log_age_gyr : ndarray of shape (n_age, )
        grid of stellar age log10(t_age/Gyr) associated with the SSP spectra

    ssp_spectra : ndarray of shape (n_met, n_age, n_ssp_wave)
        grid of SSP spectra luminosities in Lsun/Hz

    mah_params : ndarray of shape (4, )
        Parameters of the diffmah model for dark matter halo mass assembly
        logm0, logtc, early_index, late_index = mah_params

    u_ms_params : ndarray of shape (5, )
        Parameters of the diffstar model for main-sequence galaxy SFH
        u_lgmcrit, u_lgy_at_mcrit, u_indx_lo, u_indx_hi, u_tau_dep = u_ms_params

    u_q_params : ndarray of shape (4, )
        Parameters of the diffstar model for galaxy quenching
        u_qt, u_qs, u_drop, u_rejuv = u_q_params

    mzr_params : ndarray of shape (10, )
        Parameters of the DSPS model for the mass-metallicity scaling relation
        Default values are given by DEFAULT_MZR_PARAMS defined at top of module

    lgmet_scatter : float
        Scatter in dex of the metallicity distrubtion function
        Default value is given by DEFAULT_MDF_SCATTER defined at top of module

    lgt0 : float
        Base-10 log of the age of the universe at z=0 in Gyr
        Default value is 1.14 for Planck15-like cosmology

    Returns
    -------
    rest_sed : ndarray of shape (n_ssp_wave, )
        restframe SED of the galaxy in Lsun/Hz

    galaxy_data : 5-element tuple storing additional information about the galaxy

        t_table : ndarray of shape (N_T, )
            cosmic time in Gyr

        dmhdt_table : ndarray of shape (N_T, )
            Halo mass accretion rate history in Msun/yr

        log_mah_table : ndarray of shape (N_T, )
            Halo mass assembly history log10(Mhalo/Msun)

        sfh_table : ndarray of shape (N_T, )
            Star formation history in Msun/yr

        logsmh_table : ndarray of shape (N_T, )
            History of stellar mass log10(Mstar/Msun)

        lgmet : float
            Stores galaxy metallicity log10(Z/Zsun)

    """
    # Define time table used for SFH integrations
    t_table = jnp.linspace(T_MIN, t_obs, 100)
    lgt_gyr = jnp.log10(t_table)
    dt_gyr = _jax_get_dt_array(t_table)

    # Calculate halo MAH and galaxy SFH
    dmhdt_table, log_mah_table, sfh_table = _calc_galhalo_history(
        lgt_gyr, dt_gyr, lgt0, mah_params, u_ms_params, u_q_params
    )

    # Integrate galaxy SFH
    sfh_table = jnp.where(sfh_table < MIN_SFR, MIN_SFR, sfh_table)
    smh_table = jnp.cumsum(sfh_table * dt_gyr) * 1e9
    logsmh_table = jnp.log10(smh_table)
    logsm_t_obs = logsmh_table[-1]

    # Calculate galaxy metallicity
    lgmet = mzr_model(logsm_t_obs, t_obs, *mzr_params)

    galaxy_data = t_table, dmhdt_table, log_mah_table, sfh_table, lgmet

    # Calculate SED via age- and metallicity-weighted sum of SSP spectra
    rest_sed = _calc_sed_kern(
        t_obs,
        lgZsun_bin_mids,
        log_age_gyr,
        ssp_spectra,
        t_table,
        logsmh_table,
        lgmet,
        lgmet_scatter,
    )

    return rest_sed, galaxy_data


@jjit
def _calc_galhalo_history(lgt_gyr, dt_gyr, lgt0, mah_params, u_ms_params, u_q_params):
    """ """
    # Unpack diffmah parameters
    logm0, logtc, early_index, late_index = mah_params
    k = DEFAULT_MAH_PARAM_DICT["mah_k"]
    all_mah_params = (logm0, logtc, k, early_index, late_index)

    # Calculate mass assembly history of diffmah halo
    dmhdt, log_mah = _calc_halo_history(lgt_gyr, lgt0, *all_mah_params)

    # Calculate SFH of diffstar galaxy
    sfh = _sfr_history_from_mah(
        lgt_gyr, dt_gyr, dmhdt, log_mah, u_ms_params, u_q_params
    )

    return dmhdt, log_mah, sfh
