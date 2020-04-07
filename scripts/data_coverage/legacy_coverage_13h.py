import os
from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from matplotlib import pyplot as plt
from mocpy import MOC, WCS
import numpy as np


BASEPATH = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(BASEPATH, "..", "..", "data")


# MOCS: https://cds-astro.github.io/mocpy/examples/examples.html#space-time-coverages

# Get the boundaries of the Sweeps
file_name = os.path.join(data_path, "raw", "legacysurvey_dr8_south_sweep_8.0.txt")
with open(file_name, "r") as inp: 
    sweep_lines = inp.readlines()

sweeps = np.array([s[:-1] for s in sweep_lines])

min_ra = np.array([int(s[6:9]) for s in sweep_lines])
max_ra = np.array([int(s[14:17]) for s in sweep_lines])
min_dec = np.array([int(s[10:13]) if s[9] == "p" else -int(s[10:13]) 
    for s in sweep_lines])
max_dec = np.array([int(s[18:21]) if s[17] == "p" else -int(s[18:21]) 
    for s in sweep_lines])

sweep_mocs_south = []
for i in range(len(sweeps)):
    sweep_vertices = [
        (min_ra[i], min_dec[i]), 
        (max_ra[i], min_dec[i]), 
        (max_ra[i], max_dec[i]), 
        (min_ra[i], max_dec[i]),
    ]
    vertices = SkyCoord(sweep_vertices, unit="deg", frame="icrs")
    sweep_moc = MOC.from_polygon_skycoord(vertices, max_depth=9)
    sweep_mocs_south.append(sweep_moc)

file_name = os.path.join(data_path, "raw", "legacysurvey_dr8_north_sweep_8.0.txt")
with open(file_name, "r") as inp: 
    sweep_lines = inp.readlines()

sweeps = np.array([s[:-1] for s in sweep_lines])

min_ra = np.array([int(s[6:9]) for s in sweep_lines])
max_ra = np.array([int(s[14:17]) for s in sweep_lines])
min_dec = np.array([int(s[10:13]) if s[9] == "p" else -int(s[10:13]) 
    for s in sweep_lines])
max_dec = np.array([int(s[18:21]) if s[17] == "p" else -int(s[18:21]) 
    for s in sweep_lines])

sweep_mocs_north = []
for i in range(len(sweeps)):
    sweep_vertices = [
        (min_ra[i], min_dec[i]), 
        (max_ra[i], min_dec[i]), 
        (max_ra[i], max_dec[i]), 
        (min_ra[i], max_dec[i]),
    ]
    vertices = SkyCoord(sweep_vertices, unit="deg", frame="icrs")
    sweep_moc = MOC.from_polygon_skycoord(vertices, max_depth=9)
    sweep_mocs_north.append(sweep_moc)

# Get the bricks
brick_table = Table.read(os.path.join(data_path, 
    "raw", "survey-bricks.fits.gz"))
brick_table_north = Table.read(os.path.join(data_path, 
    "raw", "survey-bricks-dr8-north.fits.gz"))
brick_table_south = Table.read(os.path.join(data_path, 
    "raw", "survey-bricks-dr8-south.fits.gz"))

brick_mocs_south = []
brick_mocs_north = []
cond = (
    (brick_table["DEC"] <= 70.5) & 
    (brick_table["DEC"] >= 22.5) &
    (brick_table["RA"] <= 324) &
    (brick_table["RA"] >= 108) 
)
print(len(brick_table[cond]))
for row in brick_table[cond]:
#    if int((row["RA1"]-row["RA2"])*10000)%3600000 != 0:
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
#    else:
#        print(row["BRICKID"], row["RA1"], row["RA2"])

print("brick moc finished")


# Boundaries for the LoTSS DR2 13h region
boundaries = {"ra0" : 108, "ra1": 324, "dec0": 22.5, "dec1": 70.5}
ra_line = np.linspace(boundaries["ra0"], boundaries["ra1"], 100)
low_edge = [(x, boundaries["dec0"]) for x in ra_line]
high_edge = [(x, boundaries["dec1"]) for x in ra_line[::-1]]
area_vertices = ([
    (boundaries["ra0"], boundaries["dec0"])] +  
    low_edge +
    [   (boundaries["ra1"], boundaries["dec0"]), 
        (boundaries["ra1"], boundaries["dec1"])] +  
    high_edge +
    [(boundaries["ra0"], boundaries["dec1"])
])
area_moc = MOC.from_polygon_skycoord(
    SkyCoord(area_vertices, unit="deg", frame="icrs"),
    max_depth=9)

fig = plt.figure(1, figsize=(10, 10))
# Define a astropy WCS easily
with WCS(fig, 
        fov=100 * u.deg,
        center=SkyCoord(13*15, 45, unit='deg', frame='icrs'),
        coordsys="icrs",
        rotation=Angle(0, u.degree),
        # The gnomonic projection transforms great circles into straight lines. 
        projection="AIT") as wcs:
    ax = fig.add_subplot(1, 1, 1, projection=wcs)
    #sweep_mocs[0].fill(ax=ax, wcs=wcs, alpha=0.5, fill=True, color="red", linewidth=1)
    # Call fill with a matplotlib axe and the `~astropy.wcs.WCS` wcs object.
    for brick in brick_mocs_south:
        #moc.fill(ax=ax, wcs=wcs, alpha=0.5, fill=True, color="red", linewidth=1)
        brick.border(ax=ax, wcs=wcs, alpha=1, color="green")
    for moc in sweep_mocs_south:
        #moc.fill(ax=ax, wcs=wcs, alpha=0.5, fill=True, color="red", linewidth=1)
        moc.border(ax=ax, wcs=wcs, alpha=1, color="red")
    area_moc.border(ax=ax, wcs=wcs, alpha=1, color="blue")

plt.title('South')
plt.xlabel('RA')
plt.ylabel('Dec')
plt.grid(color="black", linestyle="dotted")

fig2 = plt.figure(2, figsize=(10, 10))
# Define a astropy WCS easily
with WCS(fig2, 
        fov=100 * u.deg,
        center=SkyCoord(13*15, 45, unit='deg', frame='icrs'),
        coordsys="icrs",
        rotation=Angle(0, u.degree),
        # The gnomonic projection transforms great circles into straight lines. 
        projection="AIT") as wcs:
    ax = fig2.add_subplot(1, 1, 1, projection=wcs)
    #sweep_mocs[0].fill(ax=ax, wcs=wcs, alpha=0.5, fill=True, color="red", linewidth=1)
    # Call fill with a matplotlib axe and the `~astropy.wcs.WCS` wcs object.
    for brick in brick_mocs_north:
        #moc.fill(ax=ax, wcs=wcs, alpha=0.5, fill=True, color="red", linewidth=1)
        brick.border(ax=ax, wcs=wcs, alpha=1, color="green")
    for moc in sweep_mocs_north:
        #moc.fill(ax=ax, wcs=wcs, alpha=0.5, fill=True, color="red", linewidth=1)
        moc.border(ax=ax, wcs=wcs, alpha=1, color="red")
    area_moc.border(ax=ax, wcs=wcs, alpha=1, color="blue")

plt.title('North')
plt.xlabel('RA')
plt.ylabel('Dec')
plt.grid(color="black", linestyle="dotted")
plt.show()


# cond = (
#     (min_dec <= boundaries["dec1"]) & 
#     (max_dec >= boundaries["dec0"]) & 
#     (
#         (max_ra >= boundaries["ra1"]) | 
#         (min_ra <= boundaries["ra0"])
#     )
# )

# selected_sweeps = sweeps[cond]
