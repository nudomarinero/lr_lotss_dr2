"""
Download the unWISE data corresponding to a given sweep.
The data is saved in a directory.
"""
from glob import glob
import os
import urllib.request
from astropy.table import Table
from dotenv import load_dotenv, find_dotenv
import numpy as np


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
template_unwise_url = "https://faun.rc.fas.harvard.edu/unwise/release/band-merged/{}"

margin_ra = 0.01
margin_dec = 0.01


def sweep_edges(sweep):
    """Get the real edges of the sources in a sweep catalogue.

    Parameters
    ----------
    sweep : string
        Name of the sweep catalogue

    Returns
    -------
    min_ra : float
        Minimum Right ascession
    max_ra : float
        Maximum Right ascession
    min_dec : float
        Minimum declination
    max_dec : float
        Maximum declination
    """
    t_sweep = Table.read(sweep)
    min_ra = np.min(t_sweep["RA"])
    max_ra = np.max(t_sweep["RA"])
    min_dec = np.min(t_sweep["DEC"])
    max_dec = np.max(t_sweep["DEC"])
    del t_sweep
    return min_ra, max_ra, min_dec, max_dec

def parse_coord_string(coord_string):
    """Parse the coordinates from an unWISE name string.
    """
    ra = float(coord_string[:4])/10
    s_sign = coord_string[4]
    if s_sign == "p":
        sign = 1
    else:
        sign = -1
    dec = sign*float(coord_string[5:8])/10
    return ra, dec

def load_unwise_catalogues():
    """Load the unWISE list of catalogues and parse their coordinates.

    Returns
    -------
    output : astropy.Table class
        unWISE catalog list with 'name', 'RA' and 'DEC' columns.
    """
    unwise_file = os.path.join(data_path, "raw", "unwise_list.txt")
    names = []
    ras = []
    decs = []
    with open(unwise_file, "r") as unwise_in:
        for l in unwise_in.readlines():
            name = l[:-1]
            ra, dec = parse_coord_string(name)
            names.append(name)
            ras.append(ra)
            decs.append(dec)
    return Table([names, ras, decs], names=('name', 'RA', 'DEC'))

def download_unwise(unwise_name, cache_path):
    """Download the unWISE catalogue to a cache area.
    """
    unwise_catalogue = os.path.join(cache_path, unwise_name)
    if not os.path.exists(unwise_catalogue):
        unwise_url = template_unwise_url.format(unwise_name)
        urllib.request.urlretrieve(unwise_url, unwise_catalogue)

def download_unwise_from_sweep(sweep, cache_path=None):
    """
    """
    min_ra, max_ra, min_dec, max_dec = sweep_edges(sweep)
    # print(min_ra, max_ra, min_dec, max_dec)
    ## Load the list
    unwise = load_unwise_catalogues()
    ## Check which unwise catalogues are in the area
    selection = (
        (unwise["RA"] >= (min_ra - margin_ra)) &
        (unwise["RA"] <= (max_ra + margin_ra)) &
        (unwise["DEC"] >= (min_dec - margin_dec)) &
        (unwise["DEC"] <= (max_dec + margin_dec))
    )
    #print(unwise[selection])
    for r in unwise[selection]:
        print(r["name"])
        download_unwise(r["name"], cache_path)

def retrieve_unwise_data():
    """Retrieve all the unWISE data corresponding to the downloaded sweeps
    in the data area.
    """
    sweep_list = glob(os.path.join(LEGACY_DATA_PATH, "sweeps", "*.fits"))
    for sweep in sweep_list:
        download_unwise_from_sweep(sweep, 
            cache_path=os.path.join(UNWISE_DATA_PATH, "band_merged"))

if __name__ == "__main__":
    #test_sweep = os.path.join(data_path, "samples", "sweep-000p025-010p030-reduced.fits")
    #test_sweep = os.path.join(data_path, "samples", "sweep-000p025-010p030.fits")
    #download_unwise_from_sweep(test_sweep, 
    #    cache_path=os.path.join(data_path, "catalogue_cache"))
    print(LEGACY_DATA_PATH)
    print(UNWISE_DATA_PATH)
    retrieve_unwise_data()
