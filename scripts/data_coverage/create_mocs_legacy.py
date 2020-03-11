import os
from astropy.table import Table, join
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from mocpy import MOC
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

print("North:", len(bricks_north))
brick_moc_north = moc_from_row(bricks_north[0])
for row in tqdm(bricks_north[1:]):
    brick_moc = moc_from_row(row)
    brick_moc_north = brick_moc_north.union(brick_moc)

print("South:", len(bricks_south))
brick_moc_south = moc_from_row(bricks_south[0])
for row in tqdm(bricks_south[1:]):
    brick_moc = moc_from_row(row)
    brick_moc_south = brick_moc_south.union(brick_moc)

brick_moc_north.write("moc_north.moc", format="fits")
brick_moc_south.write("moc_south.moc", format="fits")

brick_moc_north.write("moc_north.moc.json", format="json")
brick_moc_south.write("moc_south.moc.json", format="json")

brick_moc_north.write("moc_north.moc.fits")
brick_moc_south.write("moc_south.moc.fits")