# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ML match for LOFAR and the Legacy-WISE catalogue: Source catalogue
#
# This version computes the final parameters for the 0h region in an iterative fashion. The computation of the $Q_0$ is also included here.

# %% [markdown]
# ## Configuration
#
# ### Load libraries and setup

# %%
import pickle
import os
import sys
from glob import glob
from shutil import copyfile
import numpy as np
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
import yaml
from dotenv import load_dotenv, find_dotenv
from IPython.display import clear_output

# %%
try:
    BASEPATH = os.path.dirname(os.path.realpath(__file__))
    ROOTPATH = os.path.join(BASEPATH, "..", "..")
except NameError as e:
    if os.path.exists("data"):
        BASEPATH = os.path.realpath(".")
        ROOTPATH = BASEPATH
    else:
        BASEPATH = os.getcwd()
        ROOTPATH = os.path.join(BASEPATH, "..", "..")

data_path = os.path.join(ROOTPATH, "data")
src_path = os.path.join(ROOTPATH, "src")
config_path = os.path.join(ROOTPATH, "config")

# %%
sys.path.append(src_path)
from mltier1 import (get_center, get_n_m, estimate_q_m, Field, SingleMLEstimator, MultiMLEstimator,
                     parallel_process, get_sigma_all, get_q_m, get_threshold, q0_min_level, q0_min_numbers,
                     get_n_m_kde, estimate_q_m_kde, get_q_m_kde, describe, Q_0)

# %%
# %load_ext autoreload

# %%
# %autoreload

# %%
import matplotlib.pyplot as plt

# %%
# %matplotlib inline

# %% [markdown]
# ### Configuration

# %%
with open(os.path.join(config_path, "params.yml"), "r") as ymlfile:
    cfg_all = yaml.load(ymlfile)

# %%
load_dotenv(find_dotenv())
REGION = os.getenv("REGION")
config = cfg_all[REGION]

# %%
region_name = config["region_name"]
radio_catalogue = os.path.join(data_path, "samples", config["radio_catalogue"])
combined_catalogue = os.path.join(data_path, "samples", config["combined_catalogue"])
dec_down = config["dec_down"]
dec_up = config["dec_up"]
ra_down = config["ra_down"]
ra_up = config["ra_up"]
max_major = config["max_major"]
colour_limits_post = np.array(config["colour_limits_post"])

# %% [markdown]
# ### General configuration

# %%
save_intermediate = True
plot_intermediate = True

# %%
idp = os.path.join(data_path, "idata", region_name)

# %%
os.makedirs(idp, exist_ok=True)

# %% [markdown]
# ### Area limits

# %%
# # Test samples LoTSS_DR2_DUMMYCAT_FORPEPE_0h.srl.fits
# dec_down = 22.25
# dec_up = 32.
# ra_down = 0.
# ra_up = 20.5

# %%
margin_ra = 0.1
margin_dec = 0.1

# %%
field = Field(ra_down, ra_up, dec_down, dec_up)

# %%
field_full = field
#field_full = Field(160.0, 232.0, 42.0, 62.0)

# %%
field_optical = Field(
    ra_down - margin_ra, 
    ra_up + margin_ra, 
    dec_down - margin_dec, 
    dec_up + margin_dec)

# %% [markdown]
# ## Load data

# %%
combined_all = Table.read(combined_catalogue)

# %% [markdown]
# We will start to use the updated catalogues that include the output of the LOFAR Galaxy Zoo work.

# %%
#lofar_all = Table.read("data/LOFAR_HBA_T1_DR1_catalog_v0.9.srl.fits")
#lofar_all = Table.read(os.path.join(data_path, "samples", "P005p28.fits"))
#lofar_all = Table.read(os.path.join(data_path, "samples", "LoTSS_DR2_RA0INNER_v0.9.srl.fits"))
lofar_all = Table.read(radio_catalogue)

# %%
np.array(combined_all.colnames)

# %%
np.array(lofar_all.colnames)

# %%
describe(lofar_all['Maj'])

# %% [markdown]
# ### Filter catalogues
#
# We will take the sources in the main region but also discard sources with a Major axis size bigger than 15 arsecs.

# %%
lofar_aux = lofar_all[~np.isnan(lofar_all['Maj'])]

# %%
lofar = field.filter_catalogue(lofar_aux[(lofar_aux['Maj'] < max_major)], 
                               colnames=("RA", "DEC"))

# %%
lofar_full = field_full.filter_catalogue(lofar_aux[(lofar_aux['Maj'] < max_major)], 
                                         colnames=("RA", "DEC"))

# %%
combined = field_optical.filter_catalogue(combined_all, 
                                colnames=("RA", "DEC"))

# %%
combined_nomargin = field.filter_catalogue(combined_all, 
                                colnames=("RA", "DEC"))

# %% [markdown]
# ### Additional data
#
# Compute some additional data that was not available in the catalogues like the colour or an auxiliary array with an index.

# %%
combined["colour"] = combined["MAG_R"] - combined["MAG_W1"]

# %%
combined_aux_index = np.arange(len(combined))

# %% [markdown]
# ### Sky coordinates

# %%
coords_combined = SkyCoord(combined['RA'], 
                           combined['DEC'], 
                           unit=(u.deg, u.deg), 
                           frame='icrs')

# %%
coords_lofar = SkyCoord(lofar['RA'], 
                       lofar['DEC'], 
                       unit=(u.deg, u.deg), 
                       frame='icrs')

# %% [markdown]
# ### Class of sources in the combined catalogue
#
# The sources are grouped depending on the available photometric data.

# %%
combined_legacy = (
    ~np.isnan(combined["MAG_R"]) & 
    ~np.isnan(combined["MAG_W1"]) & 
    ~np.isnan(combined["MAG_W2"])
)

# %%
combined_wise =(
    np.isnan(combined["MAG_R"]) & 
    ~np.isnan(combined["MAG_W1"])
)

# %%
combined_wise2 =(
    np.isnan(combined["MAG_R"]) & 
    np.isnan(combined["MAG_W1"])
)

# %%
print("Total     - ", len(combined))
print("R and W1  - ", np.sum(combined_legacy))
print("Only WISE - ", np.sum(combined_wise))
print("Only W2   - ", np.sum(combined_wise2))

# %% [markdown]
# ### Colour categories
#
# The colour categories will be used after the first ML match

# %%
plt.hist(combined["colour"], bins=100);

# %%
#from astroML.plotting import hist as amlhist
from astropy.visualization import hist as amlhist

# %%
list(range(10,100,10))

# %%
np.round(np.percentile(combined["colour"][~np.isnan(combined["colour"])], list(range(10,100,10))), 1)
# array([-0.5,  0.1,  0.6,  1. ,  1.3,  1.6,  2. ,  2.5,  3.1])

# %%
#colour_limits = [0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0]
colour_limits = np.round(np.percentile(combined["colour"][~np.isnan(combined["colour"])], list(range(10,100,10))), 1)

# %% [markdown]
# Manually defined colour bins a posteriori

# %%
colour_limits = colour_limits_post

# %%
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

# %% [markdown]
# The dictionary ''colour_bin_def'' contains the indices of the different colour categories.

# %%
colour_bin_def

# %% [markdown]
# A colour category variable (numerical index from 0 to 11) is assigned to each row

# %%
combined["category"] = np.nan
for i in range(len(colour_bin_def)):
    combined["category"][colour_bin_def[i]["condition"]] = i

# %% [markdown]
# We check that there are no rows withot a category assigned

# %%
np.sum(np.isnan(combined["category"]))

# %% [markdown]
# We get the number of sources of the combined catalogue in each colour category. It will be used at a later stage to compute the $Q_0$ values

# %%
numbers_combined_bins = np.array([np.sum(a["condition"]) for a in colour_bin_def])

# %%
numbers_combined_bins

# %%
np.sum(numbers_combined_bins)

# %% [markdown]
# ## Description

# %% [markdown]
# ### Sky coverage

# %%
plt.rcParams["figure.figsize"] = (15,5)
plt.plot(lofar_all["RA"],
     lofar_all["DEC"],
     ls="", marker=",");

# %%
plt.rcParams["figure.figsize"] = (15,5)
plt.plot(lofar_full["RA"],
     lofar_full["DEC"],
     ls="", marker=",");

# %%
plt.rcParams["figure.figsize"] = (8,5)
plt.plot(lofar["RA"],
     lofar["DEC"],
     ls="", marker=",");

# %%
len(lofar)

# %% [markdown]
# ### Summary of galaxy types in the combined catalogue

# %%
np.sum(combined_legacy) # Matches # 12790855

# %%
plt.rcParams["figure.figsize"] = (15,5)
plt.subplot(1,3,1)
plt.hist(combined["MAG_R"][combined_legacy], bins=50)
plt.xlabel("R")
plt.subplot(1,3,2)
plt.hist(combined["MAG_W1"][combined_legacy], bins=50)
plt.xlabel("W1")
plt.subplot(1,3,3)
plt.hist((combined["MAG_R"] - combined["MAG_W1"])[combined_legacy], bins=50)
plt.xlabel("(R - W1)");

# %%
np.sum(combined_wise) # Only WISE

# %%
plt.rcParams["figure.figsize"] = (15,5)
plt.subplot(1,3,1)
plt.hist(combined["MAG_W1"][combined_wise], bins=50, density=True)
plt.hist(combined["MAG_W1"][combined_legacy], bins=50, alpha=0.4, density=True)
plt.xlabel("W1")
plt.subplot(1,3,2)
plt.hist((22 - combined["MAG_W1"])[combined_wise], bins=50, density=True)
plt.hist((combined["MAG_R"] - combined["MAG_W1"])[combined_legacy], bins=50, alpha=0.4, density=True)
plt.xlabel("(R - W1) (R lim. = 22)")
plt.subplot(1,3,3)
plt.hist((23 - combined["MAG_W1"])[combined_wise], bins=50, density=True)
plt.hist((combined["MAG_R"] - combined["MAG_W1"])[combined_legacy], bins=50, alpha=0.4, density=True)
plt.xlabel("(R - W1) (R lim. = 23)");

# %%
np.sum(combined_wise2) # Only W2

# %%
plt.rcParams["figure.figsize"] = (15,5)
plt.subplot(1,3,1)
plt.hist(combined["MAG_W2"][combined_wise2], bins=50, density=True)
plt.hist(combined["MAG_W2"][combined_legacy], bins=50, alpha=0.4, density=True)
plt.xlabel("W2")
plt.subplot(1,3,2)
plt.hist((22 - combined["MAG_W2"])[combined_wise2], bins=50, density=True)
plt.hist((combined["MAG_R"] - combined["MAG_W2"])[combined_legacy], bins=50, alpha=0.4, density=True)
plt.xlabel("(R - W2) (R lim. = 22)")
plt.subplot(1,3,3)
plt.hist((23 - combined["MAG_W2"])[combined_wise2], bins=50, density=True)
plt.hist((combined["MAG_R"] - combined["MAG_W2"])[combined_legacy], bins=50, alpha=0.4, density=True)
plt.xlabel("(R - W2) (R lim. = 23)");

# %% [markdown]
# ## Maximum Likelihood 1st iteration

# %% [markdown]
# ### First estimation of $Q_0$ for r-band

# %%
n_iter = 10

# %%
rads = list(range(1,26))

# %%
Q0_r = None # 0.6983157523356884

# %%
if Q0_r is None:
    q_0_comp_r = Q_0(coords_lofar, coords_combined[combined_legacy], field)

# %%
if Q0_r is None:
    q_0_rad_r = []
    q_0_rad_r_std = []
    for radius in rads:
        q_0_rad_aux = []
        for i in range(n_iter):
            try:
                out = q_0_comp_r(radius=radius)
            except ZeroDivisionError:
                continue
            else:
                q_0_rad_aux.append(out)
        q_0_rad_r.append(np.mean(q_0_rad_aux))
        q_0_rad_r_std.append(np.std(q_0_rad_aux))
        print(
            "{:2d} {:7.5f} +/- {:7.5f} [{:7.5f} {:7.5f}]".format(
                radius,
                np.mean(q_0_rad_aux),
                np.std(q_0_rad_aux),
                np.min(q_0_rad_aux),
                np.max(q_0_rad_aux),
            )
        )
    if save_intermediate:
        np.savez_compressed(
            os.path.join(idp, "Q0_r.npz"),
            q_0_rad_r = q_0_rad_r,
            q_0_rad_r_std = q_0_rad_r_std
        )

# %%
if Q0_r is None:
    plt.rcParams["figure.figsize"] = (5, 5)
    plt.plot(rads, q_0_rad_r)
    plt.plot(rads, np.array(q_0_rad_r) + 3 * np.array(q_0_rad_r_std), ls=":", color="b")
    plt.plot(rads, np.array(q_0_rad_r) - 3 * np.array(q_0_rad_r_std), ls=":", color="b")
    plt.xlabel("Radius (arcsecs)")
    plt.ylabel("$Q_0 r-band$")
    plt.ylim([0, 1])

# %%
if Q0_r is None:
    Q0_r = q_0_rad_r[4]

# %%
print(Q0_r)

# %% [markdown]
# ### Compute q(m) and n(m)
#
# #### R-band preparation

# %%
bandwidth_r = 0.5

# %%
catalogue_r = combined[combined_legacy]

# %%
bin_list_r = np.linspace(11.5, 29.5, 361) # Bins of 0.05

# %%
center_r = get_center(bin_list_r)

# %%
n_m_r = get_n_m_kde(catalogue_r["MAG_R"], center_r, field.area, bandwidth=bandwidth_r)

# %%
n_m_r_cs = np.cumsum(n_m_r)

# %%
plt.rcParams["figure.figsize"] = (5,5)
plt.plot(center_r, n_m_r_cs);

# %%
q_m_r = estimate_q_m_kde(catalogue_r["MAG_R"], 
                      center_r, 
                      n_m_r, 
                      coords_lofar, 
                      coords_combined[combined_legacy], 
                      radius=5, 
                      bandwidth=bandwidth_r)

# %%
q_m_r_cs = np.cumsum(q_m_r)

# %%
plt.rcParams["figure.figsize"] = (5,5)
plt.plot(center_r, q_m_r_cs);

# %%
plt.rcParams["figure.figsize"] = (5,5)
plt.plot(center_r, q_m_r/n_m_r);

# %% [markdown]
# #### W1-band preparation

# %%
bandwidth_w1 = 0.5

# %%
catalogue_w1 = combined[combined_wise]

# %%
bin_list_w1 = np.linspace(11.5, 25.0, 361) # Bins of 0.05

# %%
center_w1 = get_center(bin_list_w1)

# %%
n_m_w1 = get_n_m_kde(catalogue_w1["MAG_W1"], center_w1, field.area, bandwidth=bandwidth_w1)

# %%
plt.rcParams["figure.figsize"] = (5,5)
plt.plot(center_w1, np.cumsum(n_m_w1));

# %%
q_m_w1 = estimate_q_m_kde(catalogue_w1["MAG_W1"], 
                      center_w1, 
                      n_m_w1, coords_lofar, 
                      coords_combined[combined_wise], 
                      radius=5, 
                      bandwidth=bandwidth_w1)

# %%
plt.plot(center_w1, np.cumsum(q_m_w1));

# %%
plt.plot(center_w1, q_m_w1/n_m_w1);

# %% [markdown]
# #### W2-band preparation

# %%
bandwidth_w2 = 0.5

# %%
catalogue_w2 = combined[combined_wise2]

# %%
bin_list_w2 = np.linspace(14., 26., 241) # Bins of 0.1

# %%
center_w2 = get_center(bin_list_w2)

# %%
n_m_w2 = get_n_m_kde(catalogue_w2["MAG_W2"], center_w2, field.area, bandwidth=bandwidth_w2)

# %%
plt.rcParams["figure.figsize"] = (5,5)
plt.plot(center_w2, np.cumsum(n_m_w2));

# %%
q_m_w2 = estimate_q_m_kde(catalogue_w2["MAG_W2"], 
                      center_w2, 
                      n_m_w2, coords_lofar, 
                      coords_combined[combined_wise2], 
                      radius=5, 
                      bandwidth=bandwidth_w2)

# %%
plt.plot(center_w2, np.cumsum(q_m_w2));

# %%
plt.plot(center_w2, q_m_w2/n_m_w2);

# %% [markdown]
# ### r-band match

# %% [markdown]
# #### $Q_0$ and likelihood estimator

# %%
# # Initial test
# Q0_r = 0.65
# Q0_w1 = 0.237
# Q0_w2 = 0.035

# %%
likelihood_ratio_r = SingleMLEstimator(Q0_r, n_m_r, q_m_r, center_r)

# %% [markdown]
# We will get the number of CPUs to use in parallel in the computations

# %%
import multiprocessing

# %%
n_cpus_total = multiprocessing.cpu_count()

# %%
n_cpus = max(1, n_cpus_total-1)

# %%
print(n_cpus)

# %% [markdown]
# Get the possible matches up to a radius of 15 arcseconds in this first step 

# %%
radius = 15

# %% [markdown]
# All the LOFAR sources are combined with the legacy sources (sources with r-band data).

# %%
idx_lofar, idx_i, d2d, d3d = search_around_sky(
    coords_lofar, coords_combined[combined_legacy], radius*u.arcsec)

# %%
idx_lofar_unique = np.unique(idx_lofar)

# %%
lofar["lr_r"] = np.nan                   # Likelihood ratio
lofar["lr_dist_r"] = np.nan              # Distance to the selected source
lofar["lr_index_r"] = np.nan             # Index of the PanSTARRS source in combined

# %%
total_sources = len(idx_lofar_unique)
combined_aux_index = np.arange(len(combined))


# %%
def ml(i):
    idx_0 = idx_i[idx_lofar == i]
    d2d_0 = d2d[idx_lofar == i]
    mag = catalogue_r["MAG_R"][idx_0]
    
    lofar_ra = lofar[i]["RA"]
    lofar_dec = lofar[i]["DEC"]
    lofar_pa = lofar[i]["PA"]
    lofar_maj_err = lofar[i]["E_Maj"]
    lofar_min_err = lofar[i]["E_Min"]
    c_ra = catalogue_r["RA"][idx_0]
    c_dec = catalogue_r["DEC"][idx_0]
    c_ra_err = np.ones_like(c_ra)*0.6/3600.
    c_dec_err = np.ones_like(c_ra)*0.6/3600.
    
    sigma_0_0, det_sigma = get_sigma_all(lofar_maj_err, lofar_min_err, lofar_pa, 
                      lofar_ra, lofar_dec, 
                      c_ra, c_dec, c_ra_err, c_dec_err)
    
    lr_0 = likelihood_ratio_r(mag, d2d_0.arcsec, sigma_0_0, det_sigma)
    chosen_index = np.argmax(lr_0)
    result = [combined_aux_index[combined_legacy][idx_0[chosen_index]], # Index
              (d2d_0.arcsec)[chosen_index],                        # distance
              lr_0[chosen_index]]                                  # LR
    return result


# %%
#from joblib import Parallel, delayed
#from tqdm import tqdm, tqdm_notebook

# %%
#res = Parallel(n_jobs=n_cpus)(delayed(ml)(i) for i in tqdm_notebook(idx_lofar_unique))
res = parallel_process(idx_lofar_unique, ml, n_jobs=n_cpus)

# %%
(lofar["lr_index_r"][idx_lofar_unique], 
 lofar["lr_dist_r"][idx_lofar_unique], 
 lofar["lr_r"][idx_lofar_unique]) = list(map(list, zip(*res)))

# %% [markdown]
# #### Threshold and selection for r-band

# %%
lofar["lr_r"][np.isnan(lofar["lr_r"])] = 0

# %%
threshold_r = np.percentile(lofar["lr_r"], 100*(1 - Q0_r))

# %%
threshold_r #0.525 before

# %%
plt.rcParams["figure.figsize"] = (15,6)
ax1 = plt.subplot(1,2,1)
plt.hist(lofar[lofar["lr_r"] != 0]["lr_r"], bins=200)
plt.vlines([threshold_r], 0, 200)
ax1.set_yscale("log", nonposy='clip')
#plt.ylim([0, 200])
ax2 = plt.subplot(1,2,2)
plt.hist(np.log10(lofar[lofar["lr_r"] != 0]["lr_r"]+1), bins=200)
plt.vlines(np.log10(threshold_r+1), 0, 200)
ticks, _ = plt.xticks()
plt.xticks(ticks, ["{:.1f}".format(10**t-1) for t in ticks])
ax2.set_yscale("log", nonposy='clip')
#plt.ylim([0, 200]);

# %%
lofar["lr_index_sel_r"] = lofar["lr_index_r"]
lofar["lr_index_sel_r"][lofar["lr_r"] < threshold_r] = np.nan

# %% [markdown]
# Save LR for r-band in external file

# %%
columns = ["lr_r", "lr_dist_r", "lr_index_r", "lr_index_sel_r"]
np.savez_compressed(os.path.join(idp, "lr_r.npz"), lr_r=lofar[columns])

# %% [markdown]
# ### W1-band match

# %% [markdown]
# We will work with the sample that has not been already cross-matched

# %%
subsample_w1 = (lofar["lr_r"] < threshold_r)

# %% [markdown]
# #### Compute the W1 $Q_0$

# %%
coords_lofar_alt = coords_lofar[subsample_w1]

# %%
q_0_comp_w1 = Q_0(coords_lofar_alt, coords_combined[combined_wise], field)

# %%
q_0_rad_w1 = []
q_0_rad_w1_std = []
for radius in rads:
    q_0_rad_aux = []
    for i in range(n_iter):
        out = q_0_comp_w1(radius=radius)
        q_0_rad_aux.append(out)
    q_0_rad_w1.append(np.mean(q_0_rad_aux))
    q_0_rad_w1_std.append(np.std(q_0_rad_aux))
    print(
        "{:2d} {:7.5f} +/- {:7.5f} [{:7.5f} {:7.5f}]".format(
            radius,
            np.mean(q_0_rad_aux),
            np.std(q_0_rad_aux),
            np.min(q_0_rad_aux),
            np.max(q_0_rad_aux),
        )
    )

# %%
q_0_rad_w1 = np.array(q_0_rad_w1)
q_0_rad_w1_std = np.array(q_0_rad_w1_std)

# %%
plt.rcParams["figure.figsize"] = (5, 5)
plt.plot(rads, q_0_rad_w1)
plt.plot(rads, q_0_rad_w1 + 3 * q_0_rad_w1_std, ls=":", color="b")
plt.plot(rads, q_0_rad_w1 - 3 * q_0_rad_w1_std, ls=":", color="b")
plt.xlabel("Radius (arcsecs)")
plt.ylabel("$Q_0 W1-band$")
plt.ylim([0, 0.5])

# %%
Q0_w1 = q_0_rad_w1[4] #0.41136

# %% [markdown]
# #### Create the likelihood estimator and run

# %%
likelihood_ratio_w1 = SingleMLEstimator(Q0_w1, n_m_w1, q_m_w1, center_w1)

# %%
idx_lofar_w1, idx_i_w1, d2d_w1, d3d_w1 = search_around_sky(
    coords_lofar[subsample_w1], coords_combined[combined_wise], radius*u.arcsec)

# %%
idx_lofar_unique_w1 = np.unique(idx_lofar_w1)

# %%
lofar["lr_w1"] = np.nan                   # Likelihood ratio
lofar["lr_dist_w1"] = np.nan              # Distance to the selected source
lofar["lr_index_w1"] = np.nan             # Index of the PanSTARRS source in combined


# %%
def ml_w1(i):
    idx_0 = idx_i_w1[idx_lofar_w1 == i]
    d2d_0 = d2d_w1[idx_lofar_w1 == i]
    mag = catalogue_w1["MAG_W1"][idx_0]
    
    lofar_ra = lofar[subsample_w1][i]["RA"]
    lofar_dec = lofar[subsample_w1][i]["DEC"]
    lofar_pa = lofar[subsample_w1][i]["PA"]
    lofar_maj_err = lofar[subsample_w1][i]["E_Maj"]
    lofar_min_err = lofar[subsample_w1][i]["E_Min"]
    c_ra = catalogue_w1["RA"][idx_0]
    c_dec = catalogue_w1["DEC"][idx_0]
    c_ra_err = np.ones_like(c_ra)*0.6/3600.
    c_dec_err = np.ones_like(c_ra)*0.6/3600.
    
    sigma_0_0, det_sigma = get_sigma_all(lofar_maj_err, lofar_min_err, lofar_pa, 
                      lofar_ra, lofar_dec, 
                      c_ra, c_dec, c_ra_err, c_dec_err)
    
    lr_0 = likelihood_ratio_w1(mag, d2d_0.arcsec, sigma_0_0, det_sigma)
    chosen_index = np.argmax(lr_0)
    result = [combined_aux_index[combined_wise][idx_0[chosen_index]], # Index
              (d2d_0.arcsec)[chosen_index],                        # distance
              lr_0[chosen_index]]                                  # LR
    return result


# %%
res_w1 = parallel_process(idx_lofar_unique_w1, ml_w1, n_jobs=n_cpus)
#res = Parallel(n_jobs=n_cpus)(delayed(ml_w1)(i) for i in tqdm_notebook(idx_lofar_unique))

# %%
indices_w1 = np.arange(len(lofar))[subsample_w1][idx_lofar_unique_w1]

# %%
(lofar["lr_index_w1"][indices_w1], 
 lofar["lr_dist_w1"][indices_w1], 
 lofar["lr_w1"][indices_w1]) = list(map(list, zip(*res_w1)))

# %% [markdown]
# #### Threshold and selection for W1 band

# %%
lofar["lr_w1"][np.isnan(lofar["lr_w1"])] = 0

# %% [markdown]
# The threshold can be adjusted to match the new $Q_0$ value obtained with the alternative method.

# %%
threshold_w1 = np.percentile(lofar[subsample_w1]["lr_w1"], 100*(1 - Q0_w1))

# %%
threshold_w1 # 0.026 before

# %%
plt.rcParams["figure.figsize"] = (15,6)
ax1 = plt.subplot(1,2,1)
plt.hist(lofar[lofar["lr_w1"] != 0]["lr_w1"], bins=200)
plt.vlines([threshold_w1], 0, 100)
ax1.set_yscale("log", nonposy='clip')
#plt.ylim([0,100])
ax2 = plt.subplot(1,2,2)
plt.hist(np.log10(lofar[lofar["lr_w1"] != 0]["lr_w1"]+1), bins=200)
plt.vlines(np.log10(np.array([threshold_w1])+1), 0, 100)
ticks, _ = plt.xticks()
plt.xticks(ticks, ["{:.1f}".format(10**t-1) for t in ticks])
ax2.set_yscale("log", nonposy='clip')
#plt.ylim([0,100]);

# %%
np.sum(lofar["lr_w1"] >= threshold_w1)

# %%
lofar["lr_index_sel_w1"] = lofar["lr_index_w1"]
lofar["lr_index_sel_w1"][lofar["lr_w1"] < threshold_w1] = np.nan

# %% [markdown]
# Save LR of the W1-band in an external file

# %%
columns = ["lr_w1", "lr_dist_w1", "lr_index_w1", "lr_index_sel_w1"]
np.savez_compressed(os.path.join(idp, "lr_w1.npz"), lr_w1=lofar[columns])

# %% [markdown]
# ### W2-band match

# %%
subsample_w2 = (lofar["lr_r"] < threshold_r) & (lofar["lr_w1"] < threshold_w1)

# %% [markdown]
# #### Compute the W2 $Q_0$

# %%
coords_lofar_alt2 = coords_lofar[subsample_w2]

# %%
q_0_comp_w2 = Q_0(coords_lofar_alt2, coords_combined[combined_wise2], field)

# %%
q_0_rad_w2 = []
q_0_rad_w2_std = []
for radius in rads:
    q_0_rad_aux = []
    for i in range(n_iter):
        out = q_0_comp_w2(radius=radius)
        q_0_rad_aux.append(out)
    q_0_rad_w2.append(np.mean(q_0_rad_aux))
    q_0_rad_w2_std.append(np.std(q_0_rad_aux))
    print(
        "{:2d} {:7.5f} +/- {:7.5f} [{:7.5f} {:7.5f}]".format(
            radius,
            np.mean(q_0_rad_aux),
            np.std(q_0_rad_aux),
            np.min(q_0_rad_aux),
            np.max(q_0_rad_aux),
        )
    )

# %%
q_0_rad_w2 = np.array(q_0_rad_w2)
q_0_rad_w2_std = np.array(q_0_rad_w2_std)

# %%
plt.rcParams["figure.figsize"] = (5, 5)
plt.plot(rads, q_0_rad_w2)
plt.plot(rads, q_0_rad_w2 + 3 * q_0_rad_w2_std, ls=":", color="b")
plt.plot(rads, q_0_rad_w2 - 3 * q_0_rad_w2_std, ls=":", color="b")
plt.xlabel("Radius (arcsecs)")
plt.ylabel("$Q_0 W2-band$")
plt.ylim([0, 0.1])

# %%
Q0_w2 = q_0_rad_w2[4] # 0.03364

# %% [markdown]
# #### Create the likelihood estimator and run

# %%
likelihood_ratio_w2 = SingleMLEstimator(Q0_w2, n_m_w2, q_m_w2, center_w2)

# %%
idx_lofar_w2, idx_i_w2, d2d_w2, d3d_w2 = search_around_sky(
    coords_lofar[subsample_w2], coords_combined[combined_wise2], radius*u.arcsec)

# %%
idx_lofar_unique_w2 = np.unique(idx_lofar_w2)

# %%
lofar["lr_w2"] = np.nan                   # Likelihood ratio
lofar["lr_dist_w2"] = np.nan              # Distance to the selected source
lofar["lr_index_w2"] = np.nan             # Index of the PanSTARRS source in combined


# %%
def ml_w2(i):
    idx_0 = idx_i_w2[idx_lofar_w2 == i]
    d2d_0 = d2d_w2[idx_lofar_w2 == i]
    mag = catalogue_w2["MAG_W2"][idx_0]
    
    lofar_ra = lofar[subsample_w2][i]["RA"]
    lofar_dec = lofar[subsample_w2][i]["DEC"]
    lofar_pa = lofar[subsample_w2][i]["PA"]
    lofar_maj_err = lofar[subsample_w2][i]["E_Maj"]
    lofar_min_err = lofar[subsample_w2][i]["E_Min"]
    c_ra = catalogue_w2["RA"][idx_0]
    c_dec = catalogue_w2["DEC"][idx_0]
    c_ra_err = np.ones_like(c_ra)*0.6/3600.
    c_dec_err = np.ones_like(c_ra)*0.6/3600.
    
    sigma_0_0, det_sigma = get_sigma_all(lofar_maj_err, lofar_min_err, lofar_pa, 
                      lofar_ra, lofar_dec, 
                      c_ra, c_dec, c_ra_err, c_dec_err)
    
    lr_0 = likelihood_ratio_w2(mag, d2d_0.arcsec, sigma_0_0, det_sigma)
    chosen_index = np.argmax(lr_0)
    result = [combined_aux_index[combined_wise2][idx_0[chosen_index]], # Index
              (d2d_0.arcsec)[chosen_index],                        # distance
              lr_0[chosen_index]]                                  # LR
    return result


# %%
res_w2 = parallel_process(idx_lofar_unique_w2, ml_w2, n_jobs=n_cpus)
#res = Parallel(n_jobs=n_cpus)(delayed(ml_w2)(i) for i in tqdm_notebook(idx_lofar_unique))

# %%
indices_w2 = np.arange(len(lofar))[subsample_w2][idx_lofar_unique_w2]

# %%
(lofar["lr_index_w2"][indices_w2], 
 lofar["lr_dist_w2"][indices_w2], 
 lofar["lr_w2"][indices_w2]) = list(map(list, zip(*res_w2)))

# %% [markdown]
# #### Threshold and selection for W2 band

# %%
lofar["lr_w2"][np.isnan(lofar["lr_w2"])] = 0

# %%
threshold_w2 = np.percentile(lofar[subsample_w2]["lr_w2"], 100*(1 - Q0_w2))

# %%
threshold_w2 # 0.015 before

# %%
plt.rcParams["figure.figsize"] = (15,6)
ax1 = plt.subplot(1,2,1)
plt.hist(lofar[lofar["lr_w2"] != 0]["lr_w2"], bins=50)
plt.vlines([threshold_w2], 0, 10)
ax1.set_yscale("log", nonposy='clip')
#plt.ylim([0,100])
ax2 = plt.subplot(1,2,2)
plt.hist(np.log10(lofar[lofar["lr_w2"] != 0]["lr_w2"]+1), bins=50)
plt.vlines(np.log10(np.array([threshold_w2])+1), 0, 10)
ticks, _ = plt.xticks()
plt.xticks(ticks, ["{:.1f}".format(10**t-1) for t in ticks])
ax2.set_yscale("log", nonposy='clip')
#plt.ylim([0,100]);

# %%
lofar["lr_index_sel_w2"] = lofar["lr_index_w2"]
lofar["lr_index_sel_w2"][lofar["lr_w2"] < threshold_w2] = np.nan

# %% [markdown]
# ### Final selection of the match
#
# We combine the ML matching done in r-band, W1-band, and W2-band. All the galaxies were the LR is above the selection ratio for the respective band are finally selected.

# %%
lr_r_w1_w2 = (
    ~np.isnan(lofar["lr_index_sel_r"]) & 
    ~np.isnan(lofar["lr_index_sel_w1"]) & 
    ~np.isnan(lofar["lr_index_sel_w2"])
)
lr_r_w1 = (
    ~np.isnan(lofar["lr_index_sel_r"]) & 
    ~np.isnan(lofar["lr_index_sel_w1"]) & 
    np.isnan(lofar["lr_index_sel_w2"])
)
lr_r_w2 = (
    ~np.isnan(lofar["lr_index_sel_r"]) & 
    np.isnan(lofar["lr_index_sel_w1"]) & 
    ~np.isnan(lofar["lr_index_sel_w2"])
)
lr_w1_w2 = (
    np.isnan(lofar["lr_index_sel_r"]) & 
    ~np.isnan(lofar["lr_index_sel_w1"]) & 
    ~np.isnan(lofar["lr_index_sel_w2"])
)
lr_r = (
    ~np.isnan(lofar["lr_index_sel_r"]) & 
    np.isnan(lofar["lr_index_sel_w1"]) & 
    np.isnan(lofar["lr_index_sel_w2"])
)
lr_w1 = (
    np.isnan(lofar["lr_index_sel_r"]) & 
    ~np.isnan(lofar["lr_index_sel_w1"]) & 
    np.isnan(lofar["lr_index_sel_w2"])
)
lr_w2 = (
    np.isnan(lofar["lr_index_sel_r"]) & 
    np.isnan(lofar["lr_index_sel_w1"]) & 
    ~np.isnan(lofar["lr_index_sel_w2"])
)
lr_no_match = (
    np.isnan(lofar["lr_index_sel_r"]) & 
    np.isnan(lofar["lr_index_sel_w1"]) & 
    np.isnan(lofar["lr_index_sel_w2"])
)

# %%
print(np.sum(lr_r_w1_w2))
print(np.sum(lr_r_w1))
print(np.sum(lr_r_w2))
print(np.sum(lr_w1_w2))
print(np.sum(lr_r))
print(np.sum(lr_w1))
print(np.sum(lr_w2))
print(np.sum(lr_no_match))

# %%
# 0
# 0
# 0
# 0
# 68853
# 12390
# 653
# 18311

# %%
lofar["lr_index_1"] = np.nan
lofar["lr_dist_1"] = np.nan
lofar["lr_1"] = np.nan
lofar["lr_type_1"] = 0

# %%
len(lofar)

# %% [markdown]
# Enter the data into the table

# %%
r_selection = ~np.isnan(lofar["lr_index_sel_r"])
lofar["lr_index_1"][r_selection] = lofar["lr_index_r"][r_selection]
lofar["lr_dist_1"][r_selection] = lofar["lr_dist_r"][r_selection]
lofar["lr_1"][r_selection] = lofar["lr_r"][r_selection]
lofar["lr_type_1"][r_selection] = 1

w1_selection = ~np.isnan(lofar["lr_index_sel_w1"])
lofar["lr_index_1"][w1_selection] = lofar["lr_index_w1"][w1_selection]
lofar["lr_dist_1"][w1_selection] = lofar["lr_dist_w1"][w1_selection]
lofar["lr_1"][w1_selection] = lofar["lr_w1"][w1_selection]
lofar["lr_type_1"][w1_selection] = 2

w2_selection = ~np.isnan(lofar["lr_index_sel_w2"])
lofar["lr_index_1"][w2_selection] = lofar["lr_index_w2"][w2_selection]
lofar["lr_dist_1"][w2_selection] = lofar["lr_dist_w2"][w2_selection]
lofar["lr_1"][w2_selection] = lofar["lr_w2"][w2_selection]
lofar["lr_type_1"][w2_selection] = 3

# %% [markdown]
# Summary of the number of sources matched of each type

# %%
np.unique(lofar["lr_type_1"], return_counts=True)

# %%
t, c = np.unique(lofar["lr_type_1"], return_counts=True)

# %%
for i, t0 in enumerate(t):
    print("Match type {}: {}".format(t0, c[i]))

# %% [markdown]
# #### Duplicated sources
#
# This is the nymber of sources of the combined catalogue that are combined to multiple LOFAR sources. In the case of the catalogue of Gaussians the number can be very high.

# %%
values, counts = np.unique(lofar[lofar["lr_type_1"] != 0]["lr_index_1"], return_counts=True)

# %%
len(values[counts > 1]) # 101

# %%
n_dup, n_sour = np.unique(counts[counts > 1], return_counts=True)

# %%
plt.rcParams["figure.figsize"] = (6,6)
plt.semilogy(n_dup, n_sour, marker="x")
plt.xlabel("Number of multiple matches")
plt.ylabel("Number of sources in the category")

# %% [markdown]
# ### Save intermediate data

# %%
if save_intermediate:
    pickle.dump([bin_list_r, center_r, Q0_r, n_m_r, q_m_r], 
                open("{}/lofar_params_1r.pckl".format(idp), 'wb'))
    pickle.dump([bin_list_w1, center_w1, Q0_w1, n_m_w1, q_m_w1], 
                open("{}/lofar_params_1w1.pckl".format(idp), 'wb'))
    pickle.dump([bin_list_w2, center_w2, Q0_w2, n_m_w2, q_m_w2], 
                open("{}/lofar_params_1w2.pckl".format(idp), 'wb'))
    lofar.write("{}/lofar_m1.fits".format(idp), format="fits")

# %% [markdown]
# ## Second iteration using colour
#
# From now on we will take into account the effect of the colour. The sample was distributed in several categories according to the colour of the source and this is considered here.
#
# ### Rusable parameters for all the iterations
#
# These parameters are derived from the underlying population and will not change.
#
# First we compute the number of galaxies in each bin for the combined catalogue

# %%
bin_list = [bin_list_w2] + [bin_list_w1] + [bin_list_r for i in range(len(colour_bin_def))]
centers = [center_w2] + [center_w1] + [center_r for i in range(len(colour_bin_def))]

# %%
numbers_combined_bins = np.array([np.sum(a["condition"]) for a in colour_bin_def])

# %%
numbers_combined_bins

# %% [markdown]
# Get the colour category and magnitudes for the matched LOFAR sources

# %%
bandwidth_colour = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5]

# %%
n_m = []

# W2 only sources
n_m.append(get_n_m_kde(combined["MAG_W2"][combined["category"] == 0], 
                       centers[0], field.area, bandwidth=bandwidth_colour[0]))
# W1 only sources
n_m.append(get_n_m_kde(combined["MAG_W1"][combined["category"] == 1], 
                       centers[1], field.area, bandwidth=bandwidth_colour[1]))

# Rest of the sources
for i in range(2, len(colour_bin_def)):
    n_m.append(get_n_m_kde(combined["MAG_R"][combined["category"] == i], 
                           centers[i], field.area, bandwidth=bandwidth_colour[i]))

# %%
plt.rcParams["figure.figsize"] = (15,15)
for i, n_m_k in enumerate(n_m):
    plt.subplot(5,5,i+1)
    plt.plot(centers[i], np.cumsum(n_m_k))

# %% [markdown]
# ### Parameters of the matched sample
#
# The parameters derived from the matched LOFAR galaxies: $q_0$, q(m) and the number of sources per category.
#
# The columns "category", "W1mag" and "i" will contain the properties of the matched galaxies and will be updated in each iteration to save space.

# %%
lofar["category"] = np.nan
lofar["MAG_W2"] = np.nan
lofar["MAG_W1"] = np.nan
lofar["MAG_R"] = np.nan

# %%
c = ~np.isnan(lofar["lr_index_1"])
indices = lofar["lr_index_1"][c].astype(int)
lofar["category"][c] = combined[indices]["category"]
lofar["MAG_W2"][c] = combined[indices]["MAG_W2"]
lofar["MAG_W1"][c] = combined[indices]["MAG_W1"]
lofar["MAG_R"][c] = combined[indices]["MAG_R"]

# %% [markdown]
# The next parameter represent the number of matched LOFAR sources in each colour category.

# %%
numbers_lofar_combined_bins = np.array([np.sum(lofar["category"] == c) 
                                        for c in range(len(numbers_combined_bins))])

# %%
numbers_lofar_combined_bins

# %% [markdown]
# The $Q_0$ for each category are obtained by dividing the number of sources in the category by the total number of sources in the sample.

# %%
Q_0_colour = numbers_lofar_combined_bins/len(lofar) ### Q_0

# %%
q0_total = np.sum(Q_0_colour)

# %%
q0_total

# %% [markdown]
# The q(m) is not estimated with the method of Fleuren et al. but with the most updated distributions and numbers for the matches.

# %%
q_m = []
radius = 15. 

# W2 only sources
q_m.append(get_q_m_kde(lofar["MAG_W2"][lofar["category"] == 0], 
                   centers[0], 
                   radius=radius,
                   bandwidth=bandwidth_colour[0]))

# W1 only sources
q_m.append(get_q_m_kde(lofar["MAG_W1"][lofar["category"] == 1], 
                   centers[1], 
                   radius=radius,
                   bandwidth=bandwidth_colour[1]))

# Rest of the sources
for i in range(2, len(numbers_lofar_combined_bins)):
    q_m.append(get_q_m_kde(lofar["MAG_R"][lofar["category"] == i], 
                   centers[i], 
                   radius=radius,
                   bandwidth=bandwidth_colour[i]))

# %%
plt.rcParams["figure.figsize"] = (15,15)
for i, q_m_k in enumerate(q_m):
    plt.subplot(5,5,i+1)
    plt.plot(centers[i], np.cumsum(q_m_k))

# %%
plt.rcParams["figure.figsize"] = (12,10)

from matplotlib import cm
from matplotlib.collections import LineCollection

cm_subsection = np.linspace(0., 1., 16) 
colors = [ cm.viridis(x) for x in cm_subsection ]

low = np.nonzero(centers[1] >= 15)[0][0]
high = np.nonzero(centers[1] >= 22.2)[0][0]

fig, a = plt.subplots()

for i, q_m_k in enumerate(q_m):
    #plot(centers[i], q_m_old[i]/n_m_old[i])
    a = plt.subplot(4,4,i+1)
    if i not in [-1]:
        n_m_aux = n_m[i]/np.sum(n_m[i])
        lwidths = (n_m_aux/np.max(n_m_aux)*10).astype(float) + 1
        #print(lwidths)
        
        y_aux = q_m_k/n_m[i]
        factor = np.max(y_aux[low:high])
        y = y_aux
        #print(y)
        x = centers[i]
        
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        lc = LineCollection(segments, linewidths=lwidths, color=colors[i])
        
        a.add_collection(lc)
        
        #plot(centers[i], x/factor, color=colors[i-1])
        plt.xlim([12, 30])
        if i == 0:
            plt.xlim([10, 23])
        plt.ylim([0, 1.2*factor])

plt.subplots_adjust(
    left=0.125, 
    bottom=0.1, 
    right=0.9, 
    top=0.9,
    wspace=0.4, 
    hspace=0.2
)

# %% [markdown]
# * https://stackoverflow.com/questions/8931268/using-colormaps-to-set-color-of-line-in-matplotlib
# * https://stackoverflow.com/questions/19390895/matplotlib-plot-with-variable-line-width
# * https://stackoverflow.com/questions/432112/is-there-a-numpy-function-to-return-the-first-index-of-something-in-an-array

# %% [markdown]
# ### Save intermediate parameters

# %%
if save_intermediate:
    pickle.dump([bin_list, centers, Q_0_colour, n_m, q_m], 
                open("{}/lofar_params_2.pckl".format(idp), 'wb'))

# %% [markdown]
# ### Prepare for ML

# %%
selection = ~np.isnan(combined["category"]) # Avoid the two dreaded sources with no actual data
catalogue = combined[selection]

# %%
radius = 15


# %%
def apply_ml(i, likelihood_ratio_function):
    idx_0 = idx_i[idx_lofar == i]
    d2d_0 = d2d[idx_lofar == i]
    
    category = catalogue["category"][idx_0].astype(int)
    mag = catalogue["MAG_R"][idx_0]
    mag[category == 0] = catalogue["MAG_W2"][idx_0][category == 0]
    mag[category == 1] = catalogue["MAG_W1"][idx_0][category == 1]
    
    lofar_ra = lofar[i]["RA"]
    lofar_dec = lofar[i]["DEC"]
    lofar_pa = lofar[i]["PA"]
    lofar_maj_err = lofar[i]["E_Maj"]
    lofar_min_err = lofar[i]["E_Min"]
    c_ra = catalogue["RA"][idx_0]
    c_dec = catalogue["DEC"][idx_0]
    c_ra_err = np.ones_like(c_ra)*0.6/3600.
    c_dec_err = np.ones_like(c_ra)*0.6/3600.
    
    sigma_0_0, det_sigma = get_sigma_all(lofar_maj_err, lofar_min_err, lofar_pa, 
                      lofar_ra, lofar_dec, 
                      c_ra, c_dec, c_ra_err, c_dec_err)

    lr_0 = likelihood_ratio_function(mag, d2d_0.arcsec, sigma_0_0, det_sigma, category)
    
    chosen_index = np.argmax(lr_0)
    result = [combined_aux_index[selection][idx_0[chosen_index]], # Index
              (d2d_0.arcsec)[chosen_index],                        # distance
              lr_0[chosen_index]]                                  # LR
    return result


# %% [markdown]
# ### Run the cross-match
#
# This will not need to be repeated after

# %%
idx_lofar, idx_i, d2d, d3d = search_around_sky(
    coords_lofar, coords_combined[selection], radius*u.arcsec)

# %%
idx_lofar_unique = np.unique(idx_lofar)

# %% [markdown]
# ### Run the ML matching

# %%
likelihood_ratio = MultiMLEstimator(Q_0_colour, n_m, q_m, centers)


# %%
def ml(i):
    return apply_ml(i, likelihood_ratio)


# %%
res = parallel_process(idx_lofar_unique, ml, n_jobs=n_cpus)
#res = Parallel(n_jobs=n_cpus)(delayed(ml)(i) for i in tqdm_notebook(idx_lofar_unique))

# %%
lofar["lr_index_2"] = np.nan
lofar["lr_dist_2"] = np.nan
lofar["lr_2"] = np.nan

# %%
(lofar["lr_index_2"][idx_lofar_unique], 
 lofar["lr_dist_2"][idx_lofar_unique], 
 lofar["lr_2"][idx_lofar_unique]) = list(map(list, zip(*res)))

# %% [markdown]
# Get the new threshold for the ML matching. FIX THIS

# %%
lofar["lr_2"][np.isnan(lofar["lr_2"])] = 0

# %%
threshold = np.percentile(lofar["lr_2"], 100*(1 - q0_total))
#manual_q0 = 0.65
#threshold = np.percentile(lofar["lr_2"], 100*(1 - manual_q0))

# %%
threshold # Old: 0.69787

# %%
plt.rcParams["figure.figsize"] = (15,6)
plt.subplot(1,2,1)
plt.hist(lofar[lofar["lr_2"] != 0]["lr_2"], bins=200)
plt.vlines([threshold], 0, 1000)
plt.ylim([0,1000])
plt.subplot(1,2,2)
plt.hist(np.log10(lofar[lofar["lr_2"] != 0]["lr_2"]+1), bins=200)
plt.vlines(np.log10(threshold+1), 0, 1000)
ticks, _ = plt.xticks()
plt.xticks(ticks, ["{:.1f}".format(10**t-1) for t in ticks])
plt.ylim([0,1000]);

# %%
lofar["lr_index_sel_2"] = lofar["lr_index_2"]
lofar["lr_index_sel_2"][lofar["lr_2"] < threshold] = np.nan

# %%
n_changes = np.sum((lofar["lr_index_sel_2"] != lofar["lr_index_1"]) & 
                   ~np.isnan(lofar["lr_index_sel_2"]) &
                   ~np.isnan(lofar["lr_index_1"]))

# %%
n_changes # Old: 3135

# %% [markdown]
# Enter the results

# %%
# Clear aux columns
lofar["category"] = np.nan
lofar["MAG_W2"] = np.nan
lofar["MAG_W1"] = np.nan
lofar["MAG_R"] = np.nan

c = ~np.isnan(lofar["lr_index_sel_2"])
indices = lofar["lr_index_sel_2"][c].astype(int)
lofar["category"][c] = combined[indices]["category"]
lofar["MAG_W2"][c] = combined[indices]["MAG_W2"]
lofar["MAG_W1"][c] = combined[indices]["MAG_W1"]
lofar["MAG_R"][c] = combined[indices]["MAG_R"]

# %%
numbers_lofar_combined_bins = np.array([np.sum(lofar["category"] == c) 
                                        for c in range(len(numbers_combined_bins))])

# %%
numbers_lofar_combined_bins

# %% [markdown]
# ### Save intermediate data

# %%
if save_intermediate:
    lofar.write("{}/lofar_m2.fits".format(idp), format="fits")

# %% [markdown]
# ## Iterate until convergence

# %%
rerun_iter = False

# %%
if rerun_iter:
    lofar = Table.read("{}/lofar_m2.fits".format(idp))
    bin_list, centers, Q_0_colour, n_m, q_m = pickle.load(open("{}/lofar_params_2.pckl".format(idp), 'rb'))
    inter_data_list = glob("{}/lofar_m*.fits".format(idp))
    # Remove data
    for inter_data_file in inter_data_list:
        if inter_data_file[-7:-5] not in ["m1", "m2"]:
            #print(inter_data_file)
            os.remove(inter_data_file)
    # Remove images
    images_list = glob("{}/*.png".format(idp))
    for images in images_list:
        #print(images)
        os.remove(images)
    # Remove parameters
    inter_param_list = glob("{}/lofar_params_*.pckl".format(idp))
    for inter_param_file in inter_param_list:
        if inter_param_file[-7:-5] not in ["1i", "w1", "_2"]:
            #print(inter_param_file)
            os.remove(inter_param_file)

# %%
radius = 15. 

# %%
from matplotlib import pyplot as plt


# %%
def plot_q_n_m(q_m, n_m):
    fig, a = plt.subplots()

    for i, q_m_k in enumerate(q_m):
        #plot(centers[i], q_m_old[i]/n_m_old[i])
        a = plt.subplot(4,4,i+1)
        if i not in [-1]:
            n_m_aux = n_m[i]/np.sum(n_m[i])
            lwidths = (n_m_aux/np.max(n_m_aux)*10).astype(float) + 1
            #print(lwidths)

            y_aux = q_m_k/n_m[i]
            factor = np.max(y_aux[low:high])
            y = y_aux
            #print(y)
            x = centers[i]

            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, linewidths=lwidths, color=colors[i])

            a.add_collection(lc)

            #plot(centers[i], x/factor, color=colors[i-1])
            plt.xlim([12, 30])
            if i == 0:
                plt.xlim([10, 23])
            plt.ylim([0, 1.2*factor])

    plt.subplots_adjust(left=0.125, 
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9,
                    wspace=0.4, 
                    hspace=0.2)
    return fig


# %%
for j in range(10):
    iteration = j+3 
    print("Iteration {}".format(iteration))
    print("=============")
    ## Get new parameters
    # Number of matched sources per bin
    numbers_lofar_combined_bins = np.array([np.sum(lofar["category"] == c) 
                                            for c in range(len(numbers_combined_bins))])
    print("numbers_lofar_combined_bins")
    print(numbers_lofar_combined_bins)
    # q_0
    Q_0_colour_est = numbers_lofar_combined_bins/len(lofar) ### Q_0
    Q_0_colour = q0_min_level(Q_0_colour_est, min_level=0.001)
    print("Q_0_colour")
    print(Q_0_colour)
    q0_total = np.sum(Q_0_colour)
    print("Q_0_total: ", q0_total)
    # q_m
    q_m = []
    # W2 only sources
    q_m.append(get_q_m_kde(lofar["MAG_W2"][lofar["category"] == 0], 
                       centers[0], 
                       radius=radius,
                       bandwidth=bandwidth_colour[0]))
    # W1 only sources
    q_m.append(get_q_m_kde(lofar["MAG_W1"][lofar["category"] == 1], 
                       centers[1], 
                       radius=radius,
                       bandwidth=bandwidth_colour[1]))
    # Rest of the sources
    for i in range(2, len(numbers_lofar_combined_bins)):
        q_m.append(get_q_m_kde(lofar["MAG_R"][lofar["category"] == i], 
                       centers[i], 
                       radius=radius,
                       bandwidth=bandwidth_colour[i]))
    # Save new parameters
    if save_intermediate:
        pickle.dump([bin_list, centers, Q_0_colour, n_m, q_m], 
                    open("{}/lofar_params_{}.pckl".format(idp, iteration), 'wb'))
    if plot_intermediate:
        fig = plt.figure(figsize=(15,15))
        for i, q_m_k in enumerate(q_m):
            plt.subplot(5,5,i+1)
            plt.plot(centers[i], q_m_k)
        plt.savefig('{}/q0_{}.png'.format(idp, iteration))
        del fig
        fig = plt.figure(figsize=(15,15))
        for i, q_m_k in enumerate(q_m):
            plt.subplot(5,5,i+1)
            plt.plot(centers[i], q_m_k/n_m[i])
        plt.savefig('{}/q_over_n_{}.png'.format(idp, iteration))
        del fig
        fig = plot_q_n_m(q_m, n_m)
        plt.savefig('{}/q_over_n_nice_{}.png'.format(idp, iteration))
        del fig
    ## Define new likelihood_ratio
    likelihood_ratio = MultiMLEstimator(Q_0_colour, n_m, q_m, centers)
    def ml(i):
        return apply_ml(i, likelihood_ratio)
    ## Run the ML
    res = parallel_process(idx_lofar_unique, ml, n_jobs=n_cpus)
    #res = Parallel(n_jobs=n_cpus)(delayed(ml)(i) for i in tqdm_notebook(idx_lofar_unique))
    lofar["lr_index_{}".format(iteration)] = np.nan
    lofar["lr_dist_{}".format(iteration)] = np.nan
    lofar["lr_{}".format(iteration)] = np.nan
    (lofar["lr_index_{}".format(iteration)][idx_lofar_unique], 
     lofar["lr_dist_{}".format(iteration)][idx_lofar_unique], 
     lofar["lr_{}".format(iteration)][idx_lofar_unique]) = list(map(list, zip(*res)))
    lofar["lr_{}".format(iteration)][np.isnan(lofar["lr_{}".format(iteration)])] = 0
    ## Get and apply the threshold
    threshold = np.percentile(lofar["lr_{}".format(iteration)], 100*(1 - q0_total))
    #threshold = get_threshold(lofar[lofar["lr_{}".format(iteration)] != 0]["lr_{}".format(iteration)])
    print("Threshold: ", threshold)
    if plot_intermediate:
        fig = plt.figure(figsize=(15,6))
        plt.subplot(1,2,1)
        plt.hist(lofar[lofar["lr_{}".format(iteration)] != 0]["lr_{}".format(iteration)], bins=200)
        plt.vlines([threshold], 0, 1000)
        plt.ylim([0,1000])
        plt.subplot(1,2,2)
        plt.hist(np.log10(lofar[lofar["lr_{}".format(iteration)] != 0]["lr_{}".format(iteration)]+1), bins=200)
        plt.vlines(np.log10(threshold+1), 0, 1000)
        ticks, _ = plt.xticks()
        plt.xticks(ticks, ["{:.1f}".format(10**t-1) for t in ticks])
        plt.ylim([0,1000])
        plt.savefig('{}/lr_distribution_{}.png'.format(idp, iteration))
        del fig
    ## Apply the threshold
    lofar["lr_index_sel_{}".format(iteration)] = lofar["lr_index_{}".format(iteration)]
    lofar["lr_index_sel_{}".format(iteration)][lofar["lr_{}".format(iteration)] < threshold] = np.nan
    ## Enter changes into the catalogue
    # Clear aux columns
    lofar["category"] = np.nan
    lofar["MAG_W2"] = np.nan
    lofar["MAG_W1"] = np.nan
    lofar["MAG_R"] = np.nan
    # Update data
    c = ~np.isnan(lofar["lr_index_sel_{}".format(iteration)])
    indices = lofar["lr_index_sel_{}".format(iteration)][c].astype(int)
    lofar["category"][c] = combined[indices]["category"]
    lofar["MAG_W2"][c] = combined[indices]["MAG_W2"]
    lofar["MAG_W1"][c] = combined[indices]["MAG_W1"]
    lofar["MAG_R"][c] = combined[indices]["MAG_R"]
    # Save the data
    if save_intermediate:
        lofar.write("{}/lofar_m{}.fits".format(idp, iteration), format="fits")
    ## Compute number of changes
    n_changes = np.sum((
            lofar["lr_index_sel_{}".format(iteration)] != lofar["lr_index_sel_{}".format(iteration-1)]) & 
            ~np.isnan(lofar["lr_index_sel_{}".format(iteration)]) &
            ~np.isnan(lofar["lr_index_sel_{}".format(iteration-1)]))
    print("N changes: ", n_changes)
    t_changes = np.sum((
            lofar["lr_index_sel_{}".format(iteration)] != lofar["lr_index_sel_{}".format(iteration-1)]))
    print("T changes: ", t_changes)
    ## Check changes
    if n_changes == 0:
        break
    else:
        print("******** continue **********")

# %%
numbers_lofar_combined_bins = np.array([np.sum(lofar["category"] == c) 
                                        for c in range(len(numbers_combined_bins))])
numbers_lofar_combined_bins

# %%
if save_intermediate:
    pickle.dump([numbers_lofar_combined_bins, numbers_combined_bins], 
                open("{}/numbers_{}.pckl".format(idp, iteration), 'wb'))

# %% jupyter={"outputs_hidden": true}
good = False

# %%
if good:
    if os.path.exists("lofar_params.pckl"):
        os.remove("lofar_params.pckl")
    copyfile("{}/lofar_params_{}.pckl".format(idp, iteration), "lofar_params.pckl")

# %% jupyter={"outputs_hidden": true}
