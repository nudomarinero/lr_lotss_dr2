import os
from glob import glob
import numpy as np
from astropy.table import Table, vstack
from dotenv import load_dotenv, find_dotenv

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


bands_all = ['G', 'R', 'Z', 'W1', 'W2', 'W3', 'W4']
bands_lupt = ['G', 'R', 'Z', 'W1', 'W2']
sweep_columns = (
    ['BRICKNAME', 'OBJID', 'TYPE'] +
    ['RA', 'DEC', 'RA_IVAR', 'DEC_IVAR'] +
    ['FLUX_{}'.format(b) for b in bands_all] + 
    ['FLUXERR_{}'.format(b) for b in bands_all] +
    ['MAG_{}'.format(b) for b in bands_lupt] + 
    ['MAGERR_{}'.format(b) for b in bands_lupt] +
    ['ANYMASK_OPT', 'ANYMASK_G', 'ANYMASK_R', 'ANYMASK_Z']
    )

## Process the master sweep
print("Load Legacy catalogue")
master_sweep = load_legacy(
    os.path.join(LEGACY_DATA_PATH, "combined_sweep", "master_sweep.fits"))

print("Output processed catalogue")
output_sweep_name = os.path.join(
    LEGACY_DATA_PATH, 
    "combined_sweep", 
    "master_sweep_processed.fits")
master_sweep[sweep_columns].write(output_sweep_name, overwrite=True)

## Process the individual unWISE data
list_unwise = sorted(glob(os.path.join(UNWISE_DATA_PATH, "band_merged", "*.cat.fits")))

for name_unwise in list_unwise:
    print("Processing", name_unwise)
    output_name = os.path.join(name_unwise[:-9]+".cat_processed.fits")
    if not os.path.exists(output_name):
        cat_unwise = load_unwise(name_unwise, master_sweep)
        cat_unwise.write(output_name, overwrite=True)