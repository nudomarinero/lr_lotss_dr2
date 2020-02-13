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
# # Explore combined catalogue

# %% [markdown]
# Explore the sky coverage, magnitudes and colours of the Legacy and unWISE combined catalogues.

# %%
import os
import sys
import numpy as np
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky

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
import matplotlib.pyplot as plt

# %%
# %matplotlib inline

# %% [markdown]
# ## Load data

# %%
combined = Table.read(os.path.join(data_path, "samples", "test_combined.fits"))

# %%
np.array(combined.colnames)

# %%
coords_combined = SkyCoord(
    combined["RA"], combined["DEC"], unit=(u.deg, u.deg), frame="icrs"
)

# %% [markdown]
# ### Define subsamples

# %%
combined_legacy = (
    ~np.isnan(combined["MAG_R"])
    & ~np.isnan(combined["MAG_W1"])
    & ~np.isnan(combined["MAG_W2"])
)
np.sum(combined_legacy)  # Matches

# %%
combined_matched = ~np.isnan(combined["MAG_R"]) & (combined["UNWISE_OBJID"] != "N/A")
np.sum(combined_matched)

# %%
combined_legacy_only = ~np.isnan(combined["MAG_R"]) & (
    combined["UNWISE_OBJID"] == "N/A"
)
np.sum(combined_legacy_only)  # Only Legacy

# %%
combined_wise = np.isnan(combined["MAG_R"]) & (~np.isnan(combined["MAG_W1"]))
np.sum(combined_wise)  # Only WISE

# %%
combined_wise2 = np.isnan(combined["MAG_R"]) & np.isnan(combined["MAG_W1"])
np.sum(combined_wise2)  # Only WISE2

# %%
ra_legacy = coords_combined[combined_legacy][::1000].ra.wrap_at(180 * u.deg).radian
dec_legacy = coords_combined[combined_legacy][::1000].dec.radian
ra_matched = coords_combined[combined_matched][::10000].ra.wrap_at(180 * u.deg).radian
dec_matched = coords_combined[combined_matched][::10000].dec.radian
ra_legacy_only = coords_combined[combined_legacy_only][::1000].ra.wrap_at(180 * u.deg).radian
dec_legacy_only = coords_combined[combined_legacy_only][::1000].dec.radian
ra_wise = coords_combined[combined_wise][::1000].ra.wrap_at(180 * u.deg).radian
dec_wise = coords_combined[combined_wise][::1000].dec.radian
ra_wise2 = coords_combined[combined_wise2][::1000].ra.wrap_at(180 * u.deg).radian
dec_wise2 = coords_combined[combined_wise2][::1000].dec.radian

# %%
plt.subplot(111, projection="aitoff")
plt.scatter(ra_matched, dec_matched)
plt.grid(True)

# %%
#plt.plot(ra_matched, dec_matched, marker=".", ls="", alpha=0.1)
#plt.plot(ra_legacy_only, dec_legacy_only, marker=".", ls="", alpha=0.1)
plt.plot(ra_wise, dec_wise, marker=".", ls="", alpha=0.1)
plt.plot(ra_wise2, dec_wise2, marker=".", ls="", alpha=0.1)

# %% [markdown]
# ## Compare with the radio coverage

# %%
lofar_all = Table.read(os.path.join(data_path, "samples", "LoTSS_DR2_DUMMYCAT_FORPEPE_0h.srl.fits"))

# %%
coords_lofar = SkyCoord(lofar_all['RA'], 
                       lofar_all['DEC'], 
                       unit=(u.deg, u.deg), 
                       frame='icrs')

# %%
ra_lofar = coords_lofar.ra.wrap_at(180 * u.deg).radian
dec_lofar = coords_lofar.dec.radian

# %%
plt.plot(ra_matched, dec_matched, marker=".", ls="", alpha=1)
plt.plot(ra_lofar, dec_lofar, marker=".", ls="", alpha=0.01)

# %%
plt.plot(ra_lofar, dec_lofar, marker=",", ls="", alpha=1)

# %%
