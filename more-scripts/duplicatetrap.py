from scipy.spatial import Delaunay, delaunay_plot_2d
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools


traps = pd.read_csv( '../test.csv' )[['Trap', 'Longitude', 'Latitude' ]]
traplocations = traps.groupby('Trap')
traplocations.get_group( 'T035' )['Latitude']

longs = pd.unique(traplocations.get_group( 'T035' )['Longitude'])
lats = pd.unique(traplocations.get_group( 'T035' )['Latitude'])



### plot map
mapdata = np.loadtxt("../mapdata_copyright_openstreetmap_contributors.txt")
aspect = mapdata.shape[0] * 1.0 / mapdata.shape[1]
lon_lat_box = (-88, -87.5, 41.6, 42.1)

fig = plt.figure(figsize=(10,14))
ax = fig.gca()
plt.imshow(mapdata,
           cmap=plt.get_cmap('gray'),
           extent=lon_lat_box,
           aspect=aspect)


### plot traps
plt.scatter(longs[0], lats[0], marker='x')
plt.scatter(longs[1], lats[1], marker='x')

# set image boundaries
ax.set_xlim(-88, -87.5)
ax.set_ylim(41.6, 42.1)

plt.savefig('duplicatelocation.png')
