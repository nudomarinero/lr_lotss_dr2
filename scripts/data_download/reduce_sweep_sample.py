 
import os 
from astropy.table import Table
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



t_unwise = Table.read(os.path.join(data_path, "samples", "0016p272.cat.fits"))
t_sweep = Table.read(os.path.join(data_path, "samples", "sweep-000p025-010p030.fits"))

print("N unwise sources:", len(t_unwise))
print("N sweep sources:", len(t_sweep))


edge = 1./60

cond = (
    (t_sweep["RA"] <= (max(t_unwise["ra"]) + edge)) &
    (t_sweep["RA"] >= (min(t_unwise["ra"]) - edge)) &
    (t_sweep["DEC"] <= (max(t_unwise["dec"]) + edge)) &
    (t_sweep["DEC"] >= (min(t_unwise["dec"]) - edge))
)

reduced = t_sweep[cond]
print("N reduced sources:", len(reduced))

reduced.write(os.path.join(data_path, "samples", "sweep-000p025-010p030-reduced.fits"))


