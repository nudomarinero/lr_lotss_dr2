import os
import urllib.request
from glob import glob
import numpy as np
from astropy.table import Table, vstack, hstack
from astropy.coordinates import SkyCoord 
from astropy import units as u
from dotenv import load_dotenv, find_dotenv
import os

from catalog_preprocess import load_legacy, load_unwise

# Load environment config variables
# https://saurabh-kumar.com/python-dotenv/
load_dotenv(find_dotenv())
LEGACY_DATA_PATH = os.getenv("LEGACY_DATA_PATH")
UNWISE_DATA_PATH = os.getenv("UNWISE_DATA_PATH")

try:
    BASEPATH = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(BASEPATH, "..", "..", "data")
except NameError:
    if os.path.exists("data"):
        BASEPATH = "."
        data_path = os.path.join(BASEPATH, "data")
    else:
        BASEPATH = os.getcwd()
        data_path = os.path.join(BASEPATH, "..", "..", "data")
cache_path = os.path.join(data_path, "catalogue_cache")


### UnWISE
print("Load unWISE")
master_unwise = os.path.join(UNWISE_DATA_PATH, "combined", "master_unwise.fits")
t_unwise_o = Table.read(master_unwise)

## Filter unWISE
t_unwise = t_unwise_o[
    (t_unwise_o['PRIMARY'] == 1) 
]
del t_unwise_o

print("N unWISE", len(t_unwise))

## Rename conflicting columns
old_col = ('RA', 'DEC', 'FLUX_W1', 'FLUXERR_W1', 
    'FLUX_W2', 'FLUXERR_W2', 'MAG_W1', 'MAGERR_W1', 
    'MAG_W2', 'MAGERR_W2')
new_col = [c+"_U" for c in old_col]
t_unwise.rename_columns(old_col, new_col)

## Get error in ra and dec ### CHECK
t_unwise["RAERR_U"] = t_unwise["FWHM"][:,0]
t_unwise["DECERR_U"] = t_unwise["FWHM"][:,0]

### Legacy
print("Load Legacy")
master_legacy = os.path.join(LEGACY_DATA_PATH, "combined_sweep", "master_sweep_processed.fits")
t_sweep_o = Table.read(master_legacy)

## Legacy uid
t_sweep_o['UID_L'] = np.char.add(
    np.char.add(
        t_sweep_o['BRICKNAME'].astype("unicode"), 
        "_"), 
    np.char.zfill(t_sweep_o['OBJID'].astype("unicode"), 7))

## Filter Legacy sweep
t_sweep = t_sweep_o[
    (t_sweep_o['TYPE'] != 'DUP')
]
del t_sweep_o

print("N Legacy", len(t_sweep))

## Rename conflicting columns
old_col = ('RA', 'DEC', 'FLUX_W1', 'FLUXERR_W1', 
    'FLUX_W2', 'FLUXERR_W2', 'MAG_W1', 'MAGERR_W1', 
    'MAG_W2', 'MAGERR_W2')
new_col = [c+"_L" for c in old_col]
t_sweep.rename_columns(old_col, new_col)

## Rename coord columns
t_sweep.rename_columns(
    ['RA_IVAR', 'DEC_IVAR'], 
    ['RAERR_L', 'DECERR_L'])

### Cross-match
print("Start cross-match")
c_unwise = SkyCoord(ra=t_unwise["RA_U"]*u.deg, dec=t_unwise["DEC_U"]*u.deg)
c_sweep = SkyCoord(ra=t_sweep["RA_L"]*u.deg, dec=t_sweep["DEC_L"]*u.deg)
idx, d2d, d3d = c_unwise.match_to_catalog_sky(c_sweep)

print(len(t_unwise[d2d.arcsec <= 2]))
print(t_sweep.colnames)
print(t_unwise.colnames)

### Combine the three catalogues
t_sweep_matched = t_sweep[idx[d2d.arcsec <= 2]]
t_unwise_matched = t_unwise[d2d.arcsec <= 2]
print(" matched Legacy:", len(t_sweep_matched))
print(" matched unWISE:", len(t_unwise_matched))

t_matched = hstack([t_sweep_matched, t_unwise_matched])
print(" matched:", len(t_matched))
del t_unwise_matched

objid_unmatched = ~np.isin(
    t_sweep['UID_L'], 
    t_sweep_matched['UID_L'])
t_sweep_unmatched = t_sweep[objid_unmatched]
print(" legacy_unmatched:", len(t_sweep_unmatched))
del t_sweep_matched

t_unwise_unmatched = t_unwise[d2d.arcsec > 2]
print(" unwise_unmatched:", len(t_unwise_unmatched))

t_legacy_all = vstack([t_matched, t_sweep_unmatched])
print(" matched+sweep_unmatched:", len(t_legacy_all))
del t_matched
del t_sweep_unmatched

t_all = vstack([t_legacy_all, t_unwise_unmatched])
print(" All:", len(t_all))
del t_legacy_all
del t_unwise_unmatched

print(t_all.colnames)
### Select params for LR

## Coordinates

t_all['RA'] = t_all['RA_L']
t_all['RA'][t_all['RA_L'].mask] = t_all['RA_U'][t_all['RA_L'].mask]
t_all['DEC'] = t_all['DEC_L']
t_all['DEC'][t_all['DEC_L'].mask] = t_all['DEC_U'][t_all['DEC_L'].mask]

## Fluxes
t_all['MAG_W1'] = t_all['MAG_W1_L']
t_all['MAG_W1'][t_all['MAG_W1_L'].mask] = t_all['MAG_W1_U'][t_all['MAG_W1_L'].mask]
t_all['MAG_W2'] = t_all['MAG_W2_L']
t_all['MAG_W2'][t_all['MAG_W2_L'].mask] = t_all['MAG_W2_U'][t_all['MAG_W2_L'].mask]


t_all['UID_L'].fill_value = 'N/A'
t_all['UNWISE_OBJID'].fill_value = 'N/A'
t_all['MAG_R'].fill_value = np.nan
t_all['MAG_W1'].fill_value = np.nan
t_all['MAG_W2'].fill_value = np.nan

t_all['UNWISE_OBJID'][t_all['UNWISE_OBJID'] == '0.0'] = 'N/A'
t_all['MAG_W1'][t_all['MAG_W1'] == -99.] = np.nan
t_all['MAG_W2'][t_all['MAG_W2'] == -99.] = np.nan

### Save data
out_columns = ['RA', 'DEC', 'UID_L', 'UNWISE_OBJID', 'MAG_R', 'MAG_W1', 'MAG_W2']
t_all[out_columns].write(os.path.join(LEGACY_DATA_PATH, "combined.fits"), 
    overwrite=True)
