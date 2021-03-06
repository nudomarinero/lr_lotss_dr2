# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ML match for LOFAR and the combined PanSTARRS WISE catalogue: Source catalogue

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
from IPython.display import clear_output

# %%
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

# %%
sys.path.append(os.path.join(BASEPATH, '..', '..', 'src'))
from mltier1 import (get_center, get_n_m, estimate_q_m, Field, SingleMLEstimator, MultiMLEstimator,
                     parallel_process, get_sigma_all, get_q_m, get_threshold, q0_min_level, q0_min_numbers,
                     get_n_m_kde, estimate_q_m_kde, get_q_m_kde, describe)

# %%
# %load_ext autoreload

# %%
# %autoreload

# %%
import matplotlib.pyplot as plt

# %%
# %matplotlib inline

# %% [markdown]
# ### General configuration

# %%
save_intermediate = True
plot_intermediate = True

# %%
idp = os.path.join(BASEPATH, "..", "..", "data", "idata", "main")

# %%
if not os.path.isdir(idp):
    os.makedirs(idp)

# %% [markdown]
# ### Area limits

# %%
# # Test samples P005p28.fits
# dec_down = 27.206
# dec_up = 29.8
# ra_down = 4.025
# ra_up = 7.08

# # Test samples LoTSS_DR2_RA0INNER_v0.9.srl
# dec_down = 27.7
# dec_up = 34.
# ra_down = 6.
# ra_up = 20.5

# Test samples LoTSS_DR2_DUMMYCAT_FORPEPE_0h.srl.fits
dec_down = 22.25
dec_up = 32.
ra_down = 0.
ra_up = 20.5

# %%
field = Field(ra_down, ra_up, dec_down, dec_up)

# %%
field_full = field
#field_full = Field(160.0, 232.0, 42.0, 62.0)

# %% [markdown]
# ## Load data

# %%
combined_all = Table.read(os.path.join(data_path, "samples", "test_combined.fits"))

# %% [markdown]
# We will start to use the updated catalogues that include the output of the LOFAR Galaxy Zoo work.

# %%
#lofar_all = Table.read("data/LOFAR_HBA_T1_DR1_catalog_v0.9.srl.fits")
#lofar_all = Table.read(os.path.join(data_path, "samples", "P005p28.fits"))
#lofar_all = Table.read(os.path.join(data_path, "samples", "LoTSS_DR2_RA0INNER_v0.9.srl.fits"))
lofar_all = Table.read(os.path.join(data_path, "samples", "LoTSS_DR2_DUMMYCAT_FORPEPE_0h.srl.fits"))

# %%
np.array(combined_all.colnames)

# %%
np.array(lofar_all.colnames)

# %%
describe(lofar_all['Maj'])

# %% [markdown]
# ### Filter catalogues
#
# We will take the sources in the main region but also discard sources with a Major axis size bigger than 30 arsecs. We will also discard all the sources that are not classified with the code 1 in "ID_flag". Henceforth, we only take sources marked as "ML" only.

# %%
max_major = 30

# %%
lofar_aux = lofar_all[~np.isnan(lofar_all['Maj'])]

# %%
lofar = field.filter_catalogue(lofar_aux[(lofar_aux['Maj'] < max_major)], 
                               colnames=("RA", "DEC"))

# %%
lofar_full = field_full.filter_catalogue(lofar_aux[(lofar_aux['Maj'] < max_major)], 
                                         colnames=("RA", "DEC"))

# %%
combined = field.filter_catalogue(combined_all, 
                               colnames=("RA", "DEC"))

# %% [markdown]
# ### Additional data

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
combined_legacy = (~np.isnan(combined["MAG_R"]) & 
                    ~np.isnan(combined["MAG_W1"]) & 
                    ~np.isnan(combined["MAG_W2"])
                   )

# %%
combined_wise =(np.isnan(combined["MAG_R"]) & 
                (~np.isnan(combined["MAG_W1"]))
               )

# %%
combined_wise2 =(np.isnan(combined["MAG_R"]) & 
                 np.isnan(combined["MAG_W1"]))

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

# %%
#colour_limits = [0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0]
colour_limits = np.round(np.percentile(combined["colour"][~np.isnan(combined["colour"])], list(range(10,100,10))), 1)

# %%
# Start with the W1-only, i-only and "less than lower colour" bins
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

# %%
colour_bin_def

# %%
combined["category"] = np.nan
for i in range(len(colour_bin_def)):
    combined["category"][colour_bin_def[i]["condition"]] = i

# %%
np.sum(np.isnan(combined["category"]))

# %% [markdown]
# We get the number of sources of the combined catalogue in each colour category. It will be used at a later stage to compute the $Q_0$ values

# %%
numbers_combined_bins = np.array([np.sum(a["condition"]) for a in colour_bin_def])

# %%
numbers_combined_bins

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
np.sum(combined_legacy) # Matches

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
# ## Maximum Likelihood 1st

# %% [markdown]
# ### i-band preparation

# %%
bandwidth_r = 0.5

# %%
catalogue_r = combined[combined_legacy]

# %%
bin_list_r = np.linspace(11.5, 29.5, 361) # Bins of 0.05

# %%
center_r = get_center(bin_list_r)

# %%
n_m_r1 = get_n_m(catalogue_r["MAG_R"], bin_list_r, field.area)

# %%
n_m_r = get_n_m_kde(catalogue_r["MAG_R"], center_r, field.area, bandwidth=bandwidth_r)

# %%
n_m_r_cs = np.cumsum(n_m_r)

# %%
plt.rcParams["figure.figsize"] = (5,5)
plt.plot(center_r, n_m_r1);
plt.plot(center_r, n_m_r_cs);

# %%
q_m_r1 = estimate_q_m(catalogue_r["MAG_R"], 
                      bin_list_r, 
                      n_m_r1, 
                      coords_lofar, 
                      coords_combined[combined_legacy], 
                      radius=5)

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
plt.plot(center_r, q_m_r1);
plt.plot(center_r, q_m_r_cs);

# %%
plt.rcParams["figure.figsize"] = (5,5)
plt.plot(center_r, q_m_r1/n_m_r1);
plt.plot(center_r, q_m_r/n_m_r);

# %% [markdown]
# ### W1-band preparation

# %%
bandwidth_w1 = 0.5

# %%
catalogue_w1 = combined[combined_wise]

# %%
bin_list_w1 = np.linspace(11.5, 29.5, 361) # Bins of 0.05

# %%
center_w1 = get_center(bin_list_w1)

# %%
n_m_w11 = get_n_m(catalogue_w1["MAG_W1"], bin_list_w1, field.area)

# %%
n_m_w1 = get_n_m_kde(catalogue_w1["MAG_W1"], center_w1, field.area, bandwidth=bandwidth_w1)

# %%
plt.rcParams["figure.figsize"] = (5,5)
plt.plot(center_w1, n_m_w11);
plt.plot(center_w1, np.cumsum(n_m_w1));

# %%
q_m_w11 = estimate_q_m(catalogue_w1["MAG_W1"], 
                      bin_list_w1, 
                      n_m_w11, coords_lofar, 
                      coords_combined[combined_wise], 
                      radius=5)

# %%
q_m_w1 = estimate_q_m_kde(catalogue_w1["MAG_W1"], 
                      center_w1, 
                      n_m_w1, coords_lofar, 
                      coords_combined[combined_wise], 
                      radius=5, 
                      bandwidth=bandwidth_w1)

# %%
plt.plot(center_w1, q_m_w11);
plt.plot(center_w1, np.cumsum(q_m_w1));

# %%
plt.plot(center_w1, q_m_w11/n_m_w11);
plt.plot(center_w1, q_m_w1/n_m_w1);

# %% [markdown]
# ### W2-band preparation

# %%
bandwidth_w2 = 0.5

# %%
catalogue_w2 = combined[combined_wise2]

# %%
bin_list_w2 = np.linspace(12., 22., 181) # Bins of 0.05

# %%
center_w2 = get_center(bin_list_w2)

# %%
n_m_w21 = get_n_m(catalogue_w2["MAG_W2"], bin_list_w2, field.area)

# %%
n_m_w2 = get_n_m_kde(catalogue_w2["MAG_W2"], center_w2, field.area, bandwidth=bandwidth_w2)

# %%
plt.rcParams["figure.figsize"] = (5,5)
plt.plot(center_w2, n_m_w21);
plt.plot(center_w2, np.cumsum(n_m_w2));

# %%
q_m_w21 = estimate_q_m(catalogue_w2["MAG_W2"], 
                      bin_list_w2, 
                      n_m_w21, coords_lofar, 
                      coords_combined[combined_wise2], 
                      radius=5)

# %%
q_m_w2 = estimate_q_m_kde(catalogue_w2["MAG_W2"], 
                      center_w2, 
                      n_m_w2, coords_lofar, 
                      coords_combined[combined_wise2], 
                      radius=5, 
                      bandwidth=bandwidth_w2)

# %%
plt.plot(center_w2, q_m_w21);
plt.plot(center_w2, np.cumsum(q_m_w2));

# %%
plt.plot(center_w2, q_m_w21/n_m_w21);
plt.plot(center_w2, q_m_w2/n_m_w2);

# %% [markdown]
# ### $Q_0$ and likelihood estimators

# %%
# Initial test
Q0_r = 0.647
Q0_w1 = 0.217
Q0_w2 = 0.027
# 
Q0_r = 0.65
Q0_w1 = 0.237
Q0_w2 = 0.035

# %%
likelihood_ratio_r = SingleMLEstimator(Q0_r, n_m_r, q_m_r, center_r)

# %%
likelihood_ratio_w1 = SingleMLEstimator(Q0_w1, n_m_w1, q_m_w1, center_w1)

# %%
likelihood_ratio_w2 = SingleMLEstimator(Q0_w2, n_m_w2, q_m_w2, center_w2)

# %% [markdown]
# We will get the number of CPUs to use in parallel in the computations

# %%
import multiprocessing

# %%
n_cpus_total = multiprocessing.cpu_count()

# %%
n_cpus = max(1, n_cpus_total-1)

# %% [markdown]
# ### r-band match

# %%
radius = 15

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
from joblib import Parallel, delayed
from tqdm import tqdm, tqdm_notebook

# %%
res = Parallel(n_jobs=n_cpus)(delayed(ml)(i) for i in tqdm_notebook(idx_lofar_unique))

# %%
# # Old version using concurrent futures
# res = parallel_process(idx_lofar_unique, ml, n_jobs=n_cpus)

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
threshold_r #4.8 before

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

# %%
idx_lofar, idx_i, d2d, d3d = search_around_sky(
    coords_lofar, coords_combined[combined_wise], radius*u.arcsec)

# %%
idx_lofar_unique = np.unique(idx_lofar)

# %%
lofar["lr_w1"] = np.nan                   # Likelihood ratio
lofar["lr_dist_w1"] = np.nan              # Distance to the selected source
lofar["lr_index_w1"] = np.nan             # Index of the PanSTARRS source in combined


# %%
def ml_w1(i):
    idx_0 = idx_i[idx_lofar == i]
    d2d_0 = d2d[idx_lofar == i]
    mag = catalogue_w1["MAG_W1"][idx_0]
    
    lofar_ra = lofar[i]["RA"]
    lofar_dec = lofar[i]["DEC"]
    lofar_pa = lofar[i]["PA"]
    lofar_maj_err = lofar[i]["E_Maj"]
    lofar_min_err = lofar[i]["E_Min"]
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
#res = parallel_process(idx_lofar_unique, ml_w1, n_jobs=n_cpus)
res = Parallel(n_jobs=n_cpus)(delayed(ml_w1)(i) for i in tqdm_notebook(idx_lofar_unique))

# %%
(lofar["lr_index_w1"][idx_lofar_unique], 
 lofar["lr_dist_w1"][idx_lofar_unique], 
 lofar["lr_w1"][idx_lofar_unique]) = list(map(list, zip(*res)))

# %% [markdown]
# #### Threshold and selection for W1 band

# %%
lofar["lr_w1"][np.isnan(lofar["lr_w1"])] = 0

# %%
threshold_w1 = np.percentile(lofar["lr_w1"], 100*(1 - Q0_w1))

# %%
threshold_w1 # 0.695 before

# %% [markdown]
# The threshold can be adjusted to match the new $Q_0$ value obtained with the alternative method.

# %%
threshold_w1b = np.percentile(lofar["lr_w1"], 100*(1 - 0.14577))

# %%
threshold_w1b

# %%
plt.rcParams["figure.figsize"] = (15,6)
ax1 = plt.subplot(1,2,1)
plt.hist(lofar[lofar["lr_w1"] != 0]["lr_w1"], bins=200)
plt.vlines([threshold_w1, threshold_w1b], 0, 100)
ax1.set_yscale("log", nonposy='clip')
#plt.ylim([0,100])
ax2 = plt.subplot(1,2,2)
plt.hist(np.log10(lofar[lofar["lr_w1"] != 0]["lr_w1"]+1), bins=200)
plt.vlines(np.log10(np.array([threshold_w1, threshold_w1b])+1), 0, 100)
ticks, _ = plt.xticks()
plt.xticks(ticks, ["{:.1f}".format(10**t-1) for t in ticks])
ax2.set_yscale("log", nonposy='clip')
#plt.ylim([0,100]);

# %%
np.sum(lofar["lr_w1"] >= threshold_w1)

# %%
np.sum(lofar["lr_w1"] >= threshold_w1b)

# %%
np.sum((lofar["lr_r"] < lofar["lr_w1"]) & (lofar["lr_w1"] >= threshold_w1b))

# %%
np.sum(lofar["lr_w1"] >= lofar["lr_r"])

# %%
lofar["lr_index_sel_w1"] = lofar["lr_index_w1"]
lofar["lr_index_sel_w1"][lofar["lr_w1"] < threshold_w1b] = np.nan

# %% [markdown]
# Save LR of the W1-band in an external file

# %%
columns = ["lr_w1", "lr_dist_w1", "lr_index_w1", "lr_index_sel_w1"]
np.savez_compressed(os.path.join(idp, "lr_w1.npz"), lr_w1=lofar[columns])

# %% [markdown]
# ### W2-band match

# %%
idx_lofar, idx_i, d2d, d3d = search_around_sky(
    coords_lofar, coords_combined[combined_wise2], radius*u.arcsec)

# %%
idx_lofar_unique = np.unique(idx_lofar)

# %%
lofar["lr_w2"] = np.nan                   # Likelihood ratio
lofar["lr_dist_w2"] = np.nan              # Distance to the selected source
lofar["lr_index_w2"] = np.nan             # Index of the PanSTARRS source in combined


# %%
def ml_w2(i):
    idx_0 = idx_i[idx_lofar == i]
    d2d_0 = d2d[idx_lofar == i]
    mag = catalogue_w2["MAG_W2"][idx_0]
    
    lofar_ra = lofar[i]["RA"]
    lofar_dec = lofar[i]["DEC"]
    lofar_pa = lofar[i]["PA"]
    lofar_maj_err = lofar[i]["E_Maj"]
    lofar_min_err = lofar[i]["E_Min"]
    c_ra = catalogue_w1["RA"][idx_0]
    c_dec = catalogue_w1["DEC"][idx_0]
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
#res = parallel_process(idx_lofar_unique, ml_w1, n_jobs=n_cpus)
res = Parallel(n_jobs=n_cpus)(delayed(ml_w2)(i) for i in tqdm_notebook(idx_lofar_unique))

# %%
(lofar["lr_index_w2"][idx_lofar_unique], 
 lofar["lr_dist_w2"][idx_lofar_unique], 
 lofar["lr_w2"][idx_lofar_unique]) = list(map(list, zip(*res)))

# %% [markdown]
# #### Threshold and selection for W2 band

# %%
lofar["lr_w2"][np.isnan(lofar["lr_w2"])] = 0

# %%
threshold_w2 = np.percentile(lofar["lr_w2"], 100*(1 - Q0_w2))

# %%
threshold_w2 # 0.695 before

# %%
threshold_w2b = np.percentile(lofar["lr_w2"], 100*(1 - 0.015820359056752522))

# %%
threshold_w2b

# %%
plt.rcParams["figure.figsize"] = (15,6)
ax1 = plt.subplot(1,2,1)
plt.hist(lofar[lofar["lr_w2"] != 0]["lr_w2"], bins=50)
plt.vlines([threshold_w2, threshold_w2b], 0, 10)
ax1.set_yscale("log", nonposy='clip')
#plt.ylim([0,100])
ax2 = plt.subplot(1,2,2)
plt.hist(np.log10(lofar[lofar["lr_w2"] != 0]["lr_w2"]+1), bins=50)
plt.vlines(np.log10(np.array([threshold_w2, threshold_w2b])+1), 0, 10)
ticks, _ = plt.xticks()
plt.xticks(ticks, ["{:.1f}".format(10**t-1) for t in ticks])
ax2.set_yscale("log", nonposy='clip')
#plt.ylim([0,100]);

# %%
lofar["lr_index_sel_w2"] = lofar["lr_index_w2"]
lofar["lr_index_sel_w2"][lofar["lr_w2"] < threshold_w2b] = np.nan

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
lofar["lr_index_1"] = np.nan
lofar["lr_dist_1"] = np.nan
lofar["lr_1"] = np.nan
lofar["lr_type_1"] = 0

# %%
len(lofar)

# %%
lr_column_list = ["lr_r", "lr_w1", "lr_w2"]
lr_threshold_list = np.array([threshold_r, threshold_w1b, threshold_w2b])
lr_thresholds = np.repeat(np.array(lr_threshold_list)[None, :], len(lofar), axis=0)
t = lofar[lr_column_list]

# %%
ta = t.as_array()

# %%
ta

# %%
ta.shape

# %%
len(ta[0])

# %%
type(ta[0][0])

# %%
ta.shape + (len(ta[0]),)

# %%
tav = ta.view(np.float64).reshape(ta.shape + (-1,))

# %%
tav >= lr_thresholds

# %%
thresholds_mask = tav >= lr_thresholds

# %%
(1,0) == (1,0)

# %%
lr_mask = np.any(thresholds_mask, axis=1)

# %%
np.sum(~np.any(thresholds_mask, axis=1))

# %%
np.argmax(tav, axis=1)

# %%
lr_1 = np.max(tav, axis=1)
lr_1[~lr_mask] = np.nan

# %%
dist_aux = lofar[["lr_dist_r", "lr_dist_w1", "lr_dist_w2"]].as_array()
dist_aux_view = dist_aux.view(np.float64).reshape(dist_aux.shape + (-1,))
lr_dist_1 = dist_aux_view[np.eye(3)[np.argmax(tav, axis=1)].astype(bool)]
lr_dist_1[~lr_mask] = 20 # radius + 5

# %%
index_aux = lofar[["lr_index_r", "lr_index_w1", "lr_index_w2"]].as_array()
index_aux_view = index_aux.view(np.float64).reshape(index_aux.shape + (-1,))
lr_index_1 = dist_aux_view[np.eye(3)[np.argmax(tav, axis=1)].astype(bool)]
lr_index_1[~lr_mask] = np.nan


# %%
def array_view(array):
    """Get a view of the Structured Array as a pure single type numpy array"""
    array_type = type(array[0][0])
    return array.view(array_type).reshape(array.shape + (-1,))

def select_lr(lr_array, lr_dist_array, lr_index_array, threshold_list=None):
    # Sanity check shapes
    assert lr_array.shape == lr_dist_array.shape
    assert lr_array.shape == lr_index_array.shape
    
    lr_array_view = array_view(lr_array)
    lr_dist_array_view = array_view(lr_dist_array)
    lr_index_array_view = array_view(lr_index_array)
    
    lr_argmax = np.argmax(lr_array_view, axis=1)
    max_indices = np.eye(3)[lr_argmax].astype(bool)
    
    lr = np.max(lr_array_view, axis=1)
    lr_dist = lr_dist_array_view[max_indices]
    lr_index = lr_index_array_view[max_indices]
    
    if threshold_list is not None:
        assert lr_array_view.shape[1] == len(threshold_list)
        lr_thresholds = np.array(threshold_list)
    
        thresholds_mask = (lr_array_view >= lr_thresholds)
        lr_mask = np.any(thresholds_mask, axis=1)
        
        lr[~lr_mask] = np.nan
        lr_dist[~lr_mask] = 20
        lr_index[~lr_mask] = np.nan
        lr_argmax[~lr_mask] = -1
        
        lr_type = np.sum(
            np.repeat(
                2**np.arange(3)[::-1][None, :], 
                len(lofar), 
                axis=0
            ) * thresholds_mask, 
            axis=1
        )
    else:
        lr_type = np.sum(
            np.repeat(
                2**np.arange(3)[::-1][None, :], 
                len(lofar), 
                axis=0
            ) * (lr_array_view > 0.), 
            axis=1
        )
        
        
    return lr, lr_dist, lr_index, lr_argmax, lr_type


# %%
lr, lr_dist, lr_index, lr_argmax, lr_type = select_lr(
    lofar[["lr_r", "lr_w1", "lr_w2"]].as_array(),
    lofar[["lr_dist_r", "lr_dist_w1", "lr_dist_w2"]].as_array(),
    lofar[["lr_index_r", "lr_index_w1", "lr_index_w2"]].as_array(),
    threshold_list=[threshold_r, threshold_w1b, threshold_w2b]
)

# %% [markdown]
# Enter the data into the table

# %%
lofar["lr_1"] = lr
lofar["lr_dist_1"] = lr_dist
lofar["lr_index_1"] = lr_index
lofar["lr_sel_1"] = lr_argmax
lofar["lr_type_1"] = lr_type

# %% [markdown]
# Summary of the number of sources matched of each type

# %%
np.unique(lr_type, return_counts=True)

# %%
t, c = np.unique(lr_type, return_counts=True)

# %%
for i, t0 in enumerate(t):
    print("Match type {} [{:03b}]: {}".format(t0, t0, c[i]))

# %%
np.sum(lr_type != 0)/len(lofar)

# %% [markdown]
# The number of sources for which the match in r-band and W1-band are above the threshold but gives a different match to the combined catalogue.

# %%
print(np.sum(lofar["lr_index_r"][lr_r_w1] != lofar["lr_index_w1"][lr_r_w1]))

# %% [markdown]
# #### Duplicated sources
#
# This is the nymber of sources of the combined catalogue that are combined to multiple LOFAR sources. In the case of the catalogue of Gaussians the number can be very high.

# %%
values, counts = np.unique(lofar[lofar["lr_type_1"] != 0]["lr_index_1"], return_counts=True)

# %%
len(values[counts > 1])

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
bin_list = [bin_list_w1 if i == 0 else bin_list_r for i in range(len(colour_bin_def))]
centers = [center_w1 if i == 0 else center_r for i in range(len(colour_bin_def))]

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
res = Parallel(n_jobs=n_cpus)(delayed(ml)(i) for i in tqdm_notebook(idx_lofar_unique))

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
n_changes # Old: 382

# %% [markdown]
# Enter the results

# %% jupyter={"outputs_hidden": true}
# Clear aux columns
lofar["category"] = np.nan
lofar["W1mag"] = np.nan
lofar["i"] = np.nan

c = ~np.isnan(lofar["lr_index_sel_2"])
indices = lofar["lr_index_sel_2"][c].astype(int)
lofar["category"][c] = combined[indices]["category"]
lofar["W1mag"][c] = combined[indices]["W1mag"]
lofar["i"][c] = combined[indices]["i"]

# %%
numbers_lofar_combined_bins = np.array([np.sum(lofar["category"] == c) 
                                        for c in range(len(numbers_combined_bins))])

# %%
numbers_lofar_combined_bins

# %%
np.sum(numbers_lofar_combined_bins)/len(lofar)

# %% [markdown]
# ### Save intermediate data

# %% jupyter={"outputs_hidden": true}
if save_intermediate:
    lofar.write("{}/lofar_m2.fits".format(idp), format="fits")

# %% [markdown]
# ## Iterate until convergence

# %% jupyter={"outputs_hidden": true}
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

# %% jupyter={"outputs_hidden": true}
radius = 15. 

# %% jupyter={"outputs_hidden": true}
from matplotlib import pyplot as plt


# %% jupyter={"outputs_hidden": true}
def plot_q_n_m(q_m, n_m):
    fig, a = plt.subplots()

    for i, q_m_k in enumerate(q_m):
        #plot(centers[i], q_m_old[i]/n_m_old[i])
        a = subplot(4,4,i+1)
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
            xlim([12, 30])
            if i == 0:
                xlim([10, 23])
            ylim([0, 1.2*factor])

    subplots_adjust(left=0.125, 
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
    # q_m_old
    q_m_old = []
    # W1 only sources
    q_m_old.append(get_q_m(lofar["W1mag"][lofar["category"] == 0], 
                   bin_list_w1, numbers_lofar_combined_bins[0], 
                   n_m_old[0], field.area, radius=radius))
    # Rest of the sources
    for i in range(1, len(numbers_lofar_combined_bins)):
        q_m_old.append(get_q_m(lofar["i"][lofar["category"] == i], 
                       bin_list_i, numbers_lofar_combined_bins[i],
                       n_m_old[i], field.area, radius=radius))
    # q_m
    q_m = []
    # W1 only sources
    q_m.append(get_q_m_kde(lofar["W1mag"][lofar["category"] == 0], 
                   centers[0], radius=radius, bandwidth=bandwidth_colour[0]))
    # Rest of the sources
    for i in range(1, len(numbers_lofar_combined_bins)):
        q_m.append(get_q_m_kde(lofar["i"][lofar["category"] == i], 
                       centers[i], radius=radius, bandwidth=bandwidth_colour[i]))
    # Save new parameters
    if save_intermediate:
        pickle.dump([bin_list, centers, Q_0_colour, n_m_old, q_m_old], 
                    open("{}/lofar_params_cumsum_{}.pckl".format(idp, iteration), 'wb'))
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
    likelihood_ratio = MultiMLEstimatorU(Q_0_colour, n_m, q_m, centers)
    def ml(i):
        return apply_ml(i, likelihood_ratio)
    ## Run the ML
    res = parallel_process(idx_lofar_unique, ml, n_jobs=n_cpus)
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
    lofar["W1mag"] = np.nan
    lofar["i"] = np.nan
    # Update data
    c = ~np.isnan(lofar["lr_index_sel_{}".format(iteration)])
    indices = lofar["lr_index_sel_{}".format(iteration)][c].astype(int)
    lofar["category"][c] = combined[indices]["category"]
    lofar["W1mag"][c] = combined[indices]["W1mag"]
    lofar["i"][c] = combined[indices]["i"]
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

# %% jupyter={"outputs_hidden": true}
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
