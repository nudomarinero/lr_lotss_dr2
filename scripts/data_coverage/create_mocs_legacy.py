"""
Create the coverage MOCs for the Legacy survey.

See:
* http://www.ivoa.net/documents/MOC/20130910/WD-MOC-1.0-20130910.html
* https://cds-astro.github.io/mocpy/
* 
"""
import os
from astropy.table import Table, join
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from mocpy import MOC
from pymoc import MOC as MOC2
import numpy as np
from tqdm import tqdm

BASEPATH = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(BASEPATH, "..", "..", "data")


# MOCS: https://cds-astro.github.io/mocpy/examples/examples.html#space-time-coverages
brick_table = Table.read(os.path.join(data_path, 
    "raw", "survey-bricks.fits.gz"))
brick_table_north = Table.read(os.path.join(data_path, 
    "raw", "survey-bricks-dr8-north.fits.gz"))
brick_table_south = Table.read(os.path.join(data_path, 
    "raw", "survey-bricks-dr8-south.fits.gz"))

brick_table.rename_column("BRICKNAME", "brickname") 
bricks_north = join(
    brick_table_north, 
    brick_table, 
    join_type="left", 
    keys="brickname")[["RA1", "DEC1", "RA2", "DEC2"]]
bricks_south = join(
    brick_table_south, 
    brick_table, 
    join_type="left", 
    keys="brickname")[["RA1", "DEC1", "RA2", "DEC2"]]

def moc_from_row(row):
    """Create a MOC from a row with coordinates"""
    brick_vertices = [
            (row["RA1"], row["DEC1"]), 
            (row["RA1"], row["DEC2"]), 
            (row["RA2"], row["DEC2"]), 
            (row["RA2"], row["DEC1"]),
        ]
    vertices = SkyCoord(brick_vertices, unit="deg", frame="icrs")
    return MOC.from_polygon_skycoord(vertices, max_depth=10)

if os.path.exists("/run/user/1000/test.fits"):
    os.remove("/run/user/1000/test.fits")

# print("North:", len(bricks_north))
# brick_moc_north = MOC2()
# for row in tqdm(bricks_north):
#     brick_moc = moc_from_row(row)
#     brick_moc.write("/run/user/1000/test.fits")
#     brick_moc_north_aux = MOC2()
#     brick_moc_north_aux.read("/run/user/1000/test.fits")
#     brick_moc_north += brick_moc_north_aux
#     os.remove("/run/user/1000/test.fits")

# brick_moc_north.normalize()

# brick_moc_north.write("moc_north.moc.fits", overwrite=True)
# brick_moc_north.write("moc_north.moc", filetype="fits", overwrite=True)
# brick_moc_north.write("moc_north.moc.json", filetype="json")

print("South:", len(bricks_south))
brick_moc_south = MOC2()
for row in tqdm(bricks_south):
    brick_moc = moc_from_row(row)
    try:
        brick_moc.write("/run/user/1000/test.fits")
        brick_moc_south_aux = MOC2()
        brick_moc_south_aux.read("/run/user/1000/test.fits")
        brick_moc_south += brick_moc_south_aux
        os.remove("/run/user/1000/test.fits")
    except ValueError:
        print("Error saving", row)
        continue

brick_moc_south.normalize()

brick_moc_south.write("moc_south.moc.fits", overwrite=True)
brick_moc_south.write("moc_south.moc", filetype="fits", overwrite=True)
brick_moc_south.write("moc_south.moc.json", filetype="json")
