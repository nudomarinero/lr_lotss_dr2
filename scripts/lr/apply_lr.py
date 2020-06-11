# %% 
from glob import glob
import multiprocessing
import pickle
import os
import sys
import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Table, join
from astropy import units as u
from dotenv import load_dotenv, find_dotenv

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

sys.path.append(os.path.join(BASEPATH, '..', '..', 'src'))
from mltier1 import MultiMLEstimator, parallel_process, get_sigma_all

# LOAD CONFIG from ENV file
# Save as .env
#LEGACY_DATA_PATH=/disk02/jsm/Legacy_data-south-13h/Legacy
#UNWISE_DATA_PATH=/disk02/jsm/Legacy_data-south-13h/unWISE
#REGION=s13a
load_dotenv(find_dotenv())
COMBINED_DATA_PATH = os.getenv("COMBINED_DATA_PATH")
PARAMS_PATH = os.getenv("PARAMS_PATH")
THRESHOLD = os.getenv("THRESHOLD")
RADIO_CATALOGUE = os.getenv("RADIO_CATALOGUE")
OUTPUT_RADIO_CATALOGUE = os.getenv("OUTPUT_RADIO_CATALOGUE")

# Default config parameters
base_optical_catalogue = COMBINED_DATA_PATH
params = pickle.load(open(PARAMS_PATH, "rb"))
colour_limits = np.array([0.7, 1.2, 1.5, 2. , 2.4, 2.8, 3.1, 3.6, 4.1])
threshold = float(THRESHOLD)
max_major = 15
radius = 15

# input_catalogue = os.path.join(
#     os.path.join(data_path, "samples", "LoTSS_DR2_rolling.gaus_0h.fits"))
# output_catalogue = os.path.join(
#     os.path.join(data_path, "samples", "LoTSS_DR2_rolling.gaus_0h.lr.fits"))
input_catalogue = RADIO_CATALOGUE
output_catalogue = OUTPUT_RADIO_CATALOGUE

# %% 
bin_list, centers, Q_0_colour, n_m, q_m = params

## Load the catalogues
print("Load optical catalogue")
combined = Table.read(base_optical_catalogue)
print("Load input catalogue")
lofar_all = Table.read(input_catalogue)

## Filter the input catalogue
lofar = lofar_all[
    ~np.isnan(lofar_all['Maj']) & 
    (lofar_all['Maj'] < max_major)
    ]

## Get the coordinates
coords_combined = SkyCoord(combined['RA'], 
                        combined['DEC'], 
                        unit=(u.deg, u.deg), 
                        frame='icrs')
coords_lofar = SkyCoord(lofar['RA'], 
                    lofar['DEC'], 
                    unit=(u.deg, u.deg), 
                    frame='icrs')

## Get the colours for the combined catalogue
print("Get auxiliary columns")
combined["colour"] = combined["MAG_R"] - combined["MAG_W1"]
combined_aux_index = np.arange(len(combined))
combined_legacy = (
    ~np.isnan(combined["MAG_R"]) & 
    ~np.isnan(combined["MAG_W1"]) & 
    ~np.isnan(combined["MAG_W2"])
)
combined_wise =(
    np.isnan(combined["MAG_R"]) & 
    ~np.isnan(combined["MAG_W1"])
)
combined_wise2 =(
    np.isnan(combined["MAG_R"]) & 
    np.isnan(combined["MAG_W1"])
)

# Start with the W2-only, W1-only, and "less than lower colour" bins
colour_bin_def = [{"name":"only W2", "condition": combined_wise2},
                {"name":"only WISE", "condition": combined_wise},
                {"name":"-inf to {}".format(colour_limits[0]), 
                "condition": (combined["colour"] < colour_limits[0])}]

# Get the colour bins
for i in range(len(colour_limits)-1):
    name = "{} to {}".format(colour_limits[i], colour_limits[i+1])
    condition = ((combined["colour"] >= colour_limits[i]) & 
                (combined["colour"] < colour_limits[i+1]))
    colour_bin_def.append({"name":name, "condition":condition})

# Add the "more than higher colour" bin
colour_bin_def.append({"name":"{} to inf".format(colour_limits[-1]), 
                    "condition": (combined["colour"] >= colour_limits[-1])})

# Apply the categories
combined["category"] = np.nan
for i in range(len(colour_bin_def)):
    combined["category"][colour_bin_def[i]["condition"]] = i

## Define number of CPUs
n_cpus_total = multiprocessing.cpu_count()
n_cpus = max(1, n_cpus_total-1)
print(f"Use {n_cpus} CPUs")

## Start matching
print("X-match")
idx_lofar, idx_i, d2d, d3d = search_around_sky(
    coords_lofar, coords_combined, radius*u.arcsec
    )
idx_lofar_unique = np.unique(idx_lofar)
def apply_ml(i, likelihood_ratio_function):
    idx_0 = idx_i[idx_lofar == i]
    d2d_0 = d2d[idx_lofar == i]
    
    category = combined["category"][idx_0].astype(int)
    mag = combined["MAG_R"][idx_0]
    mag[category == 0] = combined["MAG_W2"][idx_0][category == 0]
    mag[category == 1] = combined["MAG_W1"][idx_0][category == 1]
    
    lofar_ra = lofar[i]["RA"]
    lofar_dec = lofar[i]["DEC"]
    lofar_pa = lofar[i]["PA"]
    lofar_maj_err = lofar[i]["E_Maj"]
    lofar_min_err = lofar[i]["E_Min"]
    c_ra = combined["RA"][idx_0]
    c_dec = combined["DEC"][idx_0]
    c_ra_err = np.ones_like(c_ra)*0.6/3600.
    c_dec_err = np.ones_like(c_ra)*0.6/3600.
    
    sigma_0_0, det_sigma = get_sigma_all(lofar_maj_err, lofar_min_err, lofar_pa, 
                    lofar_ra, lofar_dec, 
                    c_ra, c_dec, c_ra_err, c_dec_err)

    lr_0 = likelihood_ratio_function(mag, d2d_0.arcsec, sigma_0_0, det_sigma, category)
    
    chosen_index = np.argmax(lr_0)
    result = [combined_aux_index[idx_0[chosen_index]], # Index
            (d2d_0.arcsec)[chosen_index],                        # distance
            lr_0[chosen_index]]                                  # LR
    return result
likelihood_ratio = MultiMLEstimator(Q_0_colour, n_m, q_m, centers)
def ml(i):
    return apply_ml(i, likelihood_ratio)
print("Run LR")
res = parallel_process(idx_lofar_unique, ml, n_jobs=n_cpus)
lofar["lr"] = np.nan                   # Likelihood ratio
lofar["lr_dist"] = np.nan              # Distance to the selected source
lofar["lr_index"] = np.nan             # Index of the optical source in combined
(lofar["lr_index"][idx_lofar_unique], 
    lofar["lr_dist"][idx_lofar_unique], 
    lofar["lr"][idx_lofar_unique]) = list(map(list, zip(*res)))

## 
lofar["lrt"] = lofar["lr"]
lofar["lrt"][np.isnan(lofar["lr"])] = 0
lofar["lr_index_sel"] = lofar["lr_index"]
lofar["lr_index_sel"][lofar["lrt"] < threshold] = np.nan

## Save combined matches
combined["lr_index_sel"] = combined_aux_index.astype(float)
print("Combine catalogues")
pwl = join(lofar, combined, join_type='left', keys='lr_index_sel')
print("Clean catalogues")
for col in pwl.colnames:
    fv = pwl[col].fill_value
    if (isinstance(fv, np.float64) and (fv != 1e+20)):
        print(col, fv)
        pwl[col].fill_value = 1e+20
print("Save output")
pwl["RA_2"].name = "ra"
pwl["DEC_2"].name = "dec"
pwl["RA_1"].name = "RA"
pwl["DEC_2"].name = "DEC"
pwl.filled().write(output_catalogue, format="fits")

    
