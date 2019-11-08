# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Check special cases in the cross-match between Legacy and unWISE

# %%
import os 
from astropy.table import Table
import numpy as np
from astropy.coordinates import SkyCoord 
from astropy import units as u
import matplotlib.pyplot as plt

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

# %% [markdown]
# ## Load the data from the sample catalogues

# %%
t_unwise = Table.read(os.path.join(data_path, "samples", "0016p272.cat.fits"))

# %%
t_sweep = Table.read(os.path.join(data_path, "samples", "sweep-000p025-010p030-reduced.fits"))

# %%
c_unwise = SkyCoord(ra=t_unwise["ra"]*u.deg, dec=t_unwise["dec"]*u.deg)

# %%
c_sweep = SkyCoord(ra=t_sweep["RA"]*u.deg, dec=t_sweep["DEC"]*u.deg)

# %%
print(len(t_unwise), len(t_sweep))

# %% [markdown]
# Match the unWISE sources to Legacy sources

# %%
idx, d2d, d3d = c_unwise.match_to_catalog_sky(c_sweep)

# %%
t_sweep[idx][((d2d.arcsec >= 2) & (d2d.arcsec < 2.5))][::40]

# %%
t_sweep[idx][((d2d.arcsec >= 1.5) & (d2d.arcsec < 2))][::60]

# %%
results = t_sweep[idx][((d2d.arcsec >= 2) & (d2d.arcsec < 2.5))]
print(results[results["BRICKNAME"] == "0015p272"]["RA", "DEC"])

# %%
import urllib.request

# %%
from astropy.wcs import WCS
from astropy.io import fits
import pickle
from subprocess import call

# %%
name_template = "https://portal.nersc.gov/project/cosmo/data/legacysurvey/dr8/south/coadd/{}/{}/{}"

class LegacyImage():
    def __init__(self, brick_id, ra, dec, cache_area=None):
        self.brick = brick_id
        self.pre_brick = brick_id[:3]
        self.ra = ra
        self.dec = dec
        if cache_area is None:
            cache_area = os.path.join("..", "..", "data", "im_cache")
        self.cache_area = cache_area
        ## Load wcs 
        self.wcs = {"g": None, "W1": None}
        self.wcs_fits = {"g": "legacysurvey-{}-image-g.fits.fz".format(brick_id), 
                         "W1": "legacysurvey-{}-image-W1.fits.fz".format(brick_id)}
        self._load_wcs()
        # Names of master images
        self.im_name = {"g": "legacysurvey-{}-image.jpg".format(brick_id),
                        "W1": "legacysurvey-{}-wise.jpg".format(brick_id)}
        self.model_name = {"g": "legacysurvey-{}-model.jpg".format(brick_id),
                           "W1": "legacysurvey-{}-wisemodel.jpg".format(brick_id)}
        self._download_master_images()
        self._get_centre()
        
    
    def _load_wcs(self):
        """Load the WCS objects"""
        for band in ["W1"]: # We can use "W1" only as it is smaller
            wcs_file = os.path.join(self.cache_area, "legacysurvey-{}-image-{}.wcs".format(self.brick, band))
            if not os.path.exists(wcs_file):
                fits_compressed = os.path.join(self.cache_area, "legacysurvey-{}-image-{}.fits.fz".format(self.brick, band))
                fits_name = os.path.join(self.cache_area, "legacysurvey-{}-image-{}.fits".format(self.brick, band))
                fits_url = name_template.format(self.pre_brick, self.brick, self.wcs_fits[band])
                urllib.request.urlretrieve(fits_url, fits_compressed)
                command = "funpack {}".format(fits_compressed)
                call(command, shell=True)
                with fits.open(fits_name) as hdu:
                    header = hdu[0].header
                    wcs = WCS(header)
                    pickle.dump(wcs, open(wcs_file, "wb"))
                # Remove all temporary files
                try:
                    os.remove(fits_compressed)
                except OSError:
                    pass
                try:
                    os.remove(fits_name)
                except OSError:
                    pass
            self.wcs[band] = pickle.load(open(wcs_file, "rb"))
        
    def _download_master_images(self):
        """Load the image from the Internet or from the cache"""
        for band in ["g", "W1"]:
            im_file = os.path.join(self.cache_area, self.im_name[band])
            model_file = os.path.join(self.cache_area, self.model_name[band])
            if not os.path.exists(im_file):
                im_url = name_template.format(self.pre_brick, self.brick, self.im_name[band])
                urllib.request.urlretrieve(im_url, im_file)
            if not os.path.exists(model_file):
                model_url = name_template.format(self.pre_brick, self.brick, self.model_name[band])
                urllib.request.urlretrieve(model_url, model_file)
    
    def _get_centre(self):
        self.x, self.y = self.wcs["W1"].all_world2pix([self.ra], [self.dec], 1, ra_dec_order=True)
        self.x_w1 = int(round(self.x[0]))
        self.y_w1 = int(round(self.y[0]))
        self.x_g = int(round(self.x[0]*3600/342.))
        self.y_g = int(round(self.y[0]*3600/342.))
            
    def save_images(self, path=None, size=200):
        """Save the legacy and wise images in """
        if path is None:
            path = self.cache_area
        if not os.path.exists(path):
            os.mkdirs(path)
        x = self.x_g - size//2
        y = 3600 - self.y_g - size//2
        command_template_w1 = "convert {inim} -resize 3600x3600 -crop {size}x{size}+{x}+{y}\\! {outim}"
        # Image W1
        inim_w1 = os.path.join(self.cache_area, self.im_name["W1"])
        outim_w1 = os.path.join(path, "legacysurvey-{}-{:.4f}-{:.4f}-wise.jpg".format(self.brick, self.ra, self.dec))
        command_w1 = command_template_w1.format(size=size, x=x, y=y, inim=inim_w1, outim=outim_w1)
        #print(command_w1)
        call(command_w1, shell=True)
        # Model W1
        inim_w1_model = os.path.join(self.cache_area, self.model_name["W1"])
        outim_w1_model = os.path.join(path, "legacysurvey-{}-{:.4f}-{:.4f}-wise_model.jpg".format(self.brick, self.ra, self.dec))
        command_w1_model = command_template_w1.format(size=size, x=x, y=y, inim=inim_w1_model, outim=outim_w1_model)
        #print(command_w1_model)
        call(command_w1_model, shell=True)
        ##
        command_template_l = "convert {inim} -crop {size}x{size}+{x}+{y}\\! {outim}"
        # Image legacy
        inim_l = os.path.join(self.cache_area, self.im_name["g"])
        outim_l = os.path.join(path, "legacysurvey-{}-{:.4f}-{:.4f}-legacy.jpg".format(self.brick, self.ra, self.dec))
        command_l = command_template_l.format(size=size, x=x, y=y, inim=inim_l, outim=outim_l)
        #print(command_l)
        call(command_l, shell=True)
        # Model legacy
        inim_l_model = os.path.join(self.cache_area, self.model_name["g"])
        outim_l_model = os.path.join(path, "legacysurvey-{}-{:.4f}-{:.4f}-legacy_model.jpg".format(self.brick, self.ra, self.dec))
        command_l_model = command_template_l.format(size=size, x=x, y=y, inim=inim_l_model, outim=outim_l_model)
        #print(command_l_model)
        call(command_l_model, shell=True)                       

# %%
l1 = LegacyImage("0015p272", 1.6746197960348672, 27.16906855479661)

# %%
results = t_sweep[idx][((d2d.arcsec >= 2) & (d2d.arcsec < 2.5))][::40]
path = os.path.join(data_path, "..", "reports", "figures", "examples_20_25") 
for i, r in enumerate(results):
    print(i, r["BRICKNAME"])
    l = LegacyImage(r["BRICKNAME"], r["RA"], r["DEC"])
    l.save_images(path=path)

# %%
results = t_sweep[idx][((d2d.arcsec >= 1.5) & (d2d.arcsec < 2.0))][::60]
path = os.path.join(data_path, "..", "reports", "figures", "examples_15_20") 
for i, r in enumerate(results):
    print(i, r["BRICKNAME"])
    l = LegacyImage(r["BRICKNAME"], r["RA"], r["DEC"])
    l.save_images(path=path)

# %%
