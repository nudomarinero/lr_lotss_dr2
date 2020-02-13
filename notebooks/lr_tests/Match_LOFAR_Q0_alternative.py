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
# # ML match for LOFAR and the combined PanSTARRS WISE catalogue: Compute the $Q_0$

# %% [markdown]
# In this notebook we use the LR matches of the first round of LR for the r-band to improve the accuracy of the W1 band $Q_0$
#
# ## Configuration
#
# ### Load libraries and setup

# %%
import os
import sys
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
sys.path.append(os.path.join(BASEPATH, "..", "..", "src"))
from mltier1 import Field, Q_0, parallel_process, describe

# %%
idp = os.path.join(BASEPATH, "..", "..", "data", "idata", "main")

# %%
# %load_ext autoreload

# %%
# %autoreload

# %%
from IPython.display import clear_output

# %%
import matplotlib.pyplot as plt

# %% jupyter={"outputs_hidden": false}
# %matplotlib inline

# %% [markdown]
# ### Area limits

# %%
margin_ra = 0.1
margin_dec = 0.1

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
field_optical = Field(
    ra_down - margin_ra, 
    ra_up + margin_ra, 
    dec_down - margin_dec, 
    dec_up + margin_dec)

# %% [markdown]
# ## Load data

# %%
combined_all = Table.read(os.path.join(data_path, "samples", "test_combined.fits"))

# %%
#lofar_all = Table.read("data/LOFAR_HBA_T1_DR1_catalog_v0.9.srl.fits")
#lofar_all = Table.read(os.path.join(data_path, "samples", "P005p28.fits"))
lofar_all = Table.read(os.path.join(data_path, "samples", "LoTSS_DR2_DUMMYCAT_FORPEPE_0h.srl.fits"))

# %% jupyter={"outputs_hidden": false}
np.array(combined_all.colnames)

# %% jupyter={"outputs_hidden": false}
np.array(lofar_all.colnames)

# %% [markdown]
# ### Filter catalogues

# %%
lofar = field.filter_catalogue(
    lofar_all[(lofar_all["Maj"] < 30.0)], colnames=("RA", "DEC")
)

# %%
combined = field.filter_catalogue(combined_all, colnames=("RA", "DEC"))

# %%
print(len(lofar_all), len(lofar))

# %%
print(len(combined_all), len(combined))

# %% [markdown]
# ### Sky coordinates

# %% jupyter={"outputs_hidden": false}
coords_combined = SkyCoord(
    combined["RA"], combined["DEC"], unit=(u.deg, u.deg), frame="icrs"
)

# %%
coords_lofar = SkyCoord(lofar["RA"], lofar["DEC"], unit=(u.deg, u.deg), frame="icrs")

# %% [markdown]
# ### Summary of galaxy types in the combined catalogue

# %% jupyter={"outputs_hidden": false}
combined_legacy = (
    ~np.isnan(combined["MAG_R"])
    & ~np.isnan(combined["MAG_W1"])
    & ~np.isnan(combined["MAG_W2"])
)
np.sum(combined_legacy)  # Matches

# %%
combined_matched = ~np.isnan(combined["MAG_R"]) & (combined["UNWISE_OBJID"] != "N/A")
np.sum(combined_matched)

# %% jupyter={"outputs_hidden": false}
combined_legacy_only = ~np.isnan(combined["MAG_R"]) & (
    combined["UNWISE_OBJID"] == "N/A"
)
np.sum(combined_legacy_only)  # Only Legacy

# %%
print(np.sum(combined_legacy))
print(np.sum(combined_matched) + np.sum(combined_legacy_only))

# %%
combined_wise = np.isnan(combined["MAG_R"]) & (~np.isnan(combined["MAG_W1"]))
np.sum(combined_wise)  # Only WISE

# %%
combined_wise2 = np.isnan(combined["MAG_R"]) & np.isnan(combined["MAG_W1"])
np.sum(combined_wise2)  # Only WISE2

# %%
print(len(combined))
print(np.sum(combined_legacy) + np.sum(combined_wise) + np.sum(combined_wise2))

# %% [markdown]
# ## $Q_0$ dependence on the radius
#
# We will iterate 10 times for each radius.

# %%
n_iter = 10

# %%
rads = list(range(1,26))

# %% [markdown]
# ### r-band

# %%
q_0_comp_r = Q_0(coords_lofar, coords_combined[combined_legacy], field)

# %% jupyter={"outputs_hidden": false}
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

# %% jupyter={"outputs_hidden": false}
plt.rcParams["figure.figsize"] = (5, 5)
plt.plot(rads, q_0_rad_r)
plt.plot(rads, np.array(q_0_rad_r) + 3 * np.array(q_0_rad_r_std), ls=":", color="b")
plt.plot(rads, np.array(q_0_rad_r) - 3 * np.array(q_0_rad_r_std), ls=":", color="b")
plt.xlabel("Radius (arcsecs)")
plt.ylabel("$Q_0 r-band$")
plt.ylim([0, 1])

# %% [markdown]
# ### W1-band

# %%

# %%
lr_saved = np.load(os.path.join(idp, "lr_r.npz"))["lr_r"]

# %%
len(lr_saved)

# %%
len(coords_lofar)

# %%
coords_lofar_alt = coords_lofar[lr_saved["lr_r"] <= 0.8584698995739861]

# %%
len(coords_lofar_alt)

# %%
base_factor = len(coords_lofar_alt)/len(coords_lofar)

# %%
q_0_comp_w1 = Q_0(coords_lofar_alt, coords_combined[combined_wise], field)

# %% jupyter={"outputs_hidden": false}
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

# %% jupyter={"outputs_hidden": false}
plt.rcParams["figure.figsize"] = (5, 5)
plt.plot(rads, base_factor * q_0_rad_w1)
plt.plot(rads, base_factor * q_0_rad_w1 + 3 * q_0_rad_w1_std, ls=":", color="b")
plt.plot(rads, base_factor * q_0_rad_w1 - 3 * q_0_rad_w1_std, ls=":", color="b")
plt.xlabel("Radius (arcsecs)")
plt.ylabel("$Q_0 W1-band$")
plt.ylim([0, 0.2])

# %%
0.41648 * base_factor

# %% [markdown]
# ### W2-band

# %% [markdown]
# Load the LR of W1. These, combined with those of r-band will be used to discard the galaxies that have already a good match above the likelihood threshold.

# %%
lr_w1_saved = np.load(os.path.join(idp, "lr_w1.npz"))["lr_w1"]

# %%
len(lr_w1_saved)

# %%
coords_lofar_alt2 = coords_lofar[
    (lr_saved["lr_r"] <= 0.8584698995739861) &
    (lr_w1_saved["lr_w1"] <= 3.38560307030821)
]

# %%
base_factor2 = len(coords_lofar_alt2)/len(coords_lofar)

# %%
print(base_factor2)

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
plt.plot(rads, base_factor2 * q_0_rad_w2)
plt.plot(rads, base_factor2 * q_0_rad_w2 + 3 * q_0_rad_w2_std, ls=":", color="b")
plt.plot(rads, base_factor2 * q_0_rad_w2 - 3 * q_0_rad_w2_std, ls=":", color="b")
plt.xlabel("Radius (arcsecs)")
plt.ylabel("$Q_0 W2-band$")
plt.ylim([0, 0.03])

# %%
0.06319 * base_factor2
