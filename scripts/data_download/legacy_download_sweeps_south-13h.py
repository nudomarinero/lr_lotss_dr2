import os
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

template_url = "https://portal.nersc.gov/project/cosmo/data/legacysurvey/dr8/south/sweep/8.0/{}"


# Get the boundaries of the Sweeps
file_name = os.path.join(data_path, "raw", "legacysurvey_dr8_south_sweep_8.0.txt")
with open(file_name, "r") as inp: 
    sweep_lines = inp.readlines()

sweeps = np.array([s[:-1] for s in sweep_lines])

min_ra = np.array([int(s[6:9]) for s in sweep_lines])
max_ra = np.array([int(s[14:17]) for s in sweep_lines])
min_dec = np.array([int(s[10:13]) if s[9] == "p" else -int(s[10:13]) 
    for s in sweep_lines])
max_dec = np.array([int(s[18:21]) if s[17] == "p" else -int(s[18:21]) 
    for s in sweep_lines])

cond = (
    (min_dec <= 70.) & 
    (max_dec >= 24.5) &
    (min_ra <= 283) &
    (max_ra >= 108) 
)

#assert len(sweeps[cond]) == 36
print(sweeps[cond])

with open(os.path.join(BASEPATH, "download_legacy_sweeps_south-13h.sh"), 
          "w") as out:
    out.write("#!/bin/bash\n")
    #out.write("cd {}\n".format(LEGACY_DATA_PATH))
    for name in sweeps[cond]:
        url = template_url.format(name)
        out.write("wget {}\n".format(url))
        

