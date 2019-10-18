import os
import urllib.request
from astropy.table import Table
from dotenv import load_dotenv, find_dotenv
import numpy as np


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

# Load environment config variables
# https://saurabh-kumar.com/python-dotenv/
load_dotenv(find_dotenv())
LEGACY_DATA_PATH = os.getenv("LEGACY_DATA_PATH")

template_url_n = "https://portal.nersc.gov/project/cosmo/data/legacysurvey/dr8/north/tractor/{}/tractor-{}.fits"
template_url_s = "https://portal.nersc.gov/project/cosmo/data/legacysurvey/dr8/south/tractor/{}/tractor-{}.fits"

## Aux
def url_is_alive(url):
    """
    Checks that a given URL is reachable.
    :param url: A URL
    :rtype: bool
    """
    request = urllib.request.Request(url)
    request.get_method = lambda: 'HEAD'

    try:
        urllib.request.urlopen(request)
        return True
    except urllib.request.HTTPError:
        return False


##
brick_table = Table.read(os.path.join(data_path, 
    "raw", "survey-bricks.fits.gz"))

brick_table_north = Table.read(os.path.join(data_path, 
    "raw", "survey-bricks-dr8-north.fits.gz"))
brick_table_south = Table.read(os.path.join(data_path, 
    "raw", "survey-bricks-dr8-south.fits.gz"))

cond = ((brick_table["DEC"] <= 40) & 
        (brick_table["DEC"] >= 17.5) &
        ((brick_table["RA"] >= 324) | 
         (brick_table["RA"] <= 51))
)

download_names = brick_table[cond]["BRICKNAME"]

north_bricks = download_names[np.isin(download_names, brick_table_north["brickname"])]
south_bricks = download_names[np.isin(download_names, brick_table_south["brickname"])]

print(len(north_bricks))
print(len(south_bricks))
intersection_bricks = np.intersect1d(north_bricks, south_bricks)
print(len(intersection_bricks))
# 465
# 18152
# 454




# with open(os.path.join(BASEPATH, "download_legacy.sh"), 
#           "w") as out:
#     out.write("#!/bin/bash\n")
#     out.write("cd {}\n".format(LEGACY_DATA_PATH))
#     for name in download_names:
#         ra = name[:3]
#         #url_north = template_url_n.format(ra, name)
#         url_south = template_url_s.format(ra, name)
#         #print(name, 
#         #    url_is_alive(url_north), 
#         #    url_is_alive(url_south))
#         out.write("wget {}\n".format(url_south))
        

