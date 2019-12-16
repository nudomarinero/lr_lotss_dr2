import os
from glob import glob
from tqdm import tqdm
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


output_path = os.path.join(UNWISE_DATA_PATH, "combined")
unwise_path = os.path.join(UNWISE_DATA_PATH, "band_merged")

list_unwise = sorted(glob(os.path.join(UNWISE_DATA_PATH, 
    "band_merged", "*.cat_processed.fits")))

# print("Load", list_unwise[0])
# t0 = Table.read(list_unwise[0])
# for unwise in tqdm(list_unwise[1:]):
#     #print("Load", unwise)
#     t = Table.read(unwise)
#     #print("Combine")
#     t0 = vstack([t0, t])
# del t

t = [Table.read(u) for u in tqdm(list_unwise)]
t0 = vstack(t)

os.makedirs(output_path, exist_ok=True)
output_name = os.path.join(output_path, "master_unwise.fits")

print("Start output")
t0.write(output_name, overwrite=True)