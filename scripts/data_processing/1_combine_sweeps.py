import os
from glob import glob
import numpy as np
from astropy.table import Table, vstack
from dotenv import load_dotenv, find_dotenv

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
    ['FLUX_IVAR_{}'.format(b) for b in bands_all] +
    ['MW_TRANSMISSION_{}'.format(b) for b in bands_all] +
    ['PSFDEPTH_{}'.format(b) for b in bands_lupt] +
    ['ANYMASK_G', 'ANYMASK_R', 'ANYMASK_Z']
    )

output_path = os.path.join(LEGACY_DATA_PATH, "combined_sweep")
sweep_path = os.path.join(LEGACY_DATA_PATH, "sweeps")

list_sweeps = sorted(glob(os.path.join(sweep_path, "*.fits")))

print("Load", list_sweeps[0])
t0 = Table.read(list_sweeps[0])[sweep_columns]
for sweep in list_sweeps[1:]:
    print("Load", sweep)
    t = Table.read(sweep)[sweep_columns]
    print("Combine")
    t0 = vstack([t0, t])
del t

os.makedirs(output_path, exist_ok=True)
output_name = os.path.join(output_path, "master_sweep.fits")

print("Start output")
t0.write(output_name, overwrite=True)
