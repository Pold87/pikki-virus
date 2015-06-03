from scipy.spatial import Delaunay, delaunay_plot_2d
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools


traps = pd.read_csv('../train.csv')[['Longitude', 'Latitude', 'Species', 'NumMosquitos']]
locations = traps[['Longitude', 'Latitude']].values

points = locations

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

### make delaunay diagram
tri = Delaunay(points)

### plot delaunay vertices which have end point
plt.triplot(points[:,0], points[:,1], tri.simplices.copy(), 'k-')

### plot traps
locations = traps[['Longitude', 'Latitude']].values
plt.scatter(locations[:,0], locations[:,1], marker='x')

### mark points
# for j, p in enumerate(points):
# 	plt.text(p[0]-0.03, p[1]+0.03, j, ha='right') # label the points

# set image boundaries
ax.set_xlim(-88, -87.5)
ax.set_ylim(41.6, 42.1)

plt.savefig('delaunyzed.png')
