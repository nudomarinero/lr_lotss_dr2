
import numpy as np

from astropy.table import Table, Column
import astropy.units as u


"""
Global 5-sigma depths estimated from the peak of the depth distribution:

depth_summary['depthlo'][1:-1][np.argmax(depth_summary['counts_ptsrc_g'][1:-1])]+0.05
depth_summary['depthlo'][1:-1][np.argmax(depth_summary['counts_ptsrc_r'][1:-1])]+0.05
depth_summary['depthlo'][1:-1][np.argmax(depth_summary['counts_ptsrc_z'][1:-1])]+0.05

"""

f0 = (3631*u.Jy).to(u.uJy).value
nanomaggy_to_ujy = 10**((23.9-22.5)/2.5)
nanovega_to_ujy_w1 = 10**((23.9-2.699-22.5)/2.5)
nanovega_to_ujy_w2 = 10**((23.9-3.339-22.5)/2.5)

def flux_to_lupt(flux, fluxerr, b):
    """ Calculate asinh magnitudes 'luptitudes' for given fluxes.

    Parameters
    ----------
    flux : array
        Flux density (in units of microjansky)
    fluxerr : array
        Uncertainty on flux (in units of microjansky)
    b : float
        Dimensionless softening parameter for asinh magnitudes. Set externally
        and

    Returns
    -------

    lupt : array
        asinh magnitudes (luptitudes) with AB zeropoint
    lupterr : array
        Uncertainty on lupt

    """
    lupt = (-2.5/np.log(10)) * (np.arcsinh((flux/f0)/(2*b)) + np.log(b))
    lupterr = (2.5/np.log(10)) * np.abs(fluxerr/flux) / np.sqrt(1 + (2*b / (flux/f0))**2)
    return lupt, lupterr

def load_legacy(legacy_path, format='fits'):
    """ Load a Legacy Surveys optical catalog and process ready for use.

    Parameters
    ----------
    legacy_path : str
        Path to input Legacy Catalog brick or sweep
    format : str (default = 'fits')
        astropy compatible catalog format.


    Returns
    -------
    sweep : astropy.Table class
        Processed Legacy catalog suitable for likelihood ratio or photo-z use.
    """

    sweep = Table.read(legacy_path, format=format)

    bands_all = ['G', 'R', 'Z', 'W1', 'W2', 'W3', 'W4']
    bands_lupt = ['G', 'R', 'Z', 'W1', 'W2']

    for band in bands_all:
        sweep[f'FLUX_{band}'] *= nanomaggy_to_ujy / sweep[f'MW_TRANSMISSION_{band}']
        sweep[f'FLUXERR_{band}'] = (sweep[f'FLUX_IVAR_{band}']**-0.5 *
                                    nanomaggy_to_ujy /
                                    sweep[f'MW_TRANSMISSION_{band}'])

        if band in bands_lupt:
            b = np.median((1/np.sqrt(sweep[f'PSFDEPTH_{band}'])) * nanomaggy_to_ujy) / f0
            lupt, lupterr = flux_to_lupt(sweep[f'FLUX_{band}'], sweep[f'FLUXERR_{band}'], b)

            sweep.add_column(Column(data=lupt,
                                    name=f'MAG_{band}', meta={'b':b}))
            sweep.add_column(Column(data=lupterr,
                                    name=f'MAGERR_{band}', meta={'b':b}))

    sweep['ANYMASK_OPT'] = ((sweep['ANYMASK_G'] + sweep['ANYMASK_R'] + sweep['ANYMASK_Z']) > 0)

    return sweep


def load_unwise(unwise_path, legacy_cat, format='fits'):
    """ Load a unWISE multi-band catalog and process ready for use.

    Parameters
    ----------
    unwise_path : str
        Path to input unwise band-merged catalog
    legacy_cat : astropy.Table class
        Legacy Survey sweep within which the unWISE catalog is positioned.
        Needed in order to propagate the asinh magnitude softening paramater
        for consistency between Legacy and unWISE mags/errors.
    format : str (default = 'fits')
        astropy compatible catalog format input catalog.


    Returns
    -------
    output : astropy.Table class
        Processed unWISE catalog suitable for likelihood ratio.
    """

    unwise = Table.read(unwise_path, format=format)

    output = Table()

    main_cols = ['ra', 'dec', 'unwise_objid', 'primary']
    for col in main_cols:
        output[col.upper()] = unwise[col]

    output['FLUX_W1'] = unwise['flux'][:,0] * nanovega_to_ujy_w1
    output['FLUXERR_W1'] = unwise['dflux'][:,0] * nanovega_to_ujy_w1

    output['FLUX_W2'] = unwise['flux'][:,1] * nanovega_to_ujy_w2
    output['FLUXERR_W2'] = unwise['dflux'][:,1] * nanovega_to_ujy_w2

    for band in ['W1', 'W2']:
        # Asinh Mag softening paramaters set to match Legacy Survey
        b = legacy_cat[f'MAG_{band}'].meta['b']
        lupt, lupterr = flux_to_lupt(output[f'FLUX_{band}'], output[f'FLUXERR_{band}'], b)

        lupt[output[f'FLUXERR_{band}'] == 0.] = -99.
        lupterr[output[f'FLUXERR_{band}'] == 0.] = -99.

        output.add_column(Column(data=lupt,
                                 name=f'MAG_{band}', meta={'b':b}))
        output.add_column(Column(data=lupterr,
                                 name=f'MAGERR_{band}', meta={'b':b}))

    # Keep additional columns for potential later use in deciding match quality
    extra_cols = ['qf', 'rchi2', 'fracflux', 'fwhm', 'sky',
              'coadd_id', 'unwise_detid', 'nm', 'flags_unwise', 'flags_info']
    for col in extra_cols:
        output[col.upper()] = unwise[col]

    return output


if __name__ == '__main__':

    # Insert any two random catalogs here for testing:
    test_sweep = load_legacy('../../data/raw/sweep-030p030-040p035.fits')
    test_unwise = load_unwise('../../data/raw/1677p227.cat.fits', test_sweep)
