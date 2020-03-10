import os
from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from mocpy import MOC
import numpy as np

BASEPATH = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(BASEPATH, "..", "..", "data")


# MOCS: https://cds-astro.github.io/mocpy/examples/examples.html#space-time-coverages
brick_table = Table.read(os.path.join(data_path, 
    "raw", "survey-bricks.fits.gz"))
brick_table_north = Table.read(os.path.join(data_path, 
    "raw", "survey-bricks-dr8-north.fits.gz"))
brick_table_south = Table.read(os.path.join(data_path, 
    "raw", "survey-bricks-dr8-south.fits.gz"))

brick_mocs_south = []
brick_mocs_north = []

print(len(brick_table))
for row in brick_table:
#    if int((row["RA1"]-row["RA2"])*10000)%3600000 != 0:
    if ((row["BRICKNAME"] in brick_table_north["brickname"]) or
        (row["BRICKNAME"] in brick_table_south["brickname"])):
        brick_vertices = [
            (row["RA1"], row["DEC1"]), 
            (row["RA1"], row["DEC2"]), 
            (row["RA2"], row["DEC2"]), 
            (row["RA2"], row["DEC1"]),
        ]
        vertices = SkyCoord(brick_vertices, unit="deg", frame="icrs")
        brick_moc = MOC.from_polygon_skycoord(vertices, max_depth=9)
        if row["BRICKNAME"] in brick_table_north["brickname"]:
            brick_mocs_north.append(brick_moc)
        if row["BRICKNAME"] in brick_table_south["brickname"]:
            brick_mocs_south.append(brick_moc)

brick_moc_north = brick_mocs_north[0]
for i in range(1, len(brick_mocs_north)):
    brick_moc_north += brick_mocs_north[i]
brick_moc_north.normalize()

brick_moc_south = brick_mocs_south[0]
for i in range(1, len(brick_mocs_south)):
    brick_moc_south += brick_mocs_south[i]
brick_moc_south.normalize()

brick_moc_north.write("moc_north.moc")
brick_moc_north.write("moc_north.moc.fits")
brick_moc_north.write("moc_north.moc.json")

brick_moc_north.write("moc_north.moc")
brick_moc_north.write("moc_north.moc.fits")
brick_moc_north.write("moc_north.moc.json")