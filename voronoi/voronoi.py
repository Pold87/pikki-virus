from scipy.spatial import Delaunay
from collections import defaultdict
import pandas as pd
import numpy as np
import itertools


traps = pd.read_csv('../train.csv')[['Longitude', 'Latitude', 'Species', 'NumMosquitos']]
locations = traps[['Longitude', 'Latitude']].values

points = locations

### make voronoi diagram
vor = Voronoi(points)

### Visualize voronoi on map
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

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

### plot voronoi vertices which have end point
for simplex in vor.ridge_vertices:
    simplex = np.asarray(simplex)
    if np.all(simplex >= 0):
        plt.plot(vor.vertices[simplex,0], vor.vertices[simplex,1], 'k-')

### plot voronoi vertices which go into infinity
center = points.mean(axis=0)
for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
    simplex = np.asarray(simplex)
    if np.any(simplex < 0):
        i = simplex[simplex >= 0][0] # finite end Voronoi vertex
        t = points[pointidx[1]] - points[pointidx[0]] # tangent
        t /= np.linalg.norm(t)
        n = np.array([-t[1], t[0]]) # normal
        midpoint = points[pointidx].mean(axis=0)
        far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 100
        plt.plot([vor.vertices[i,0], far_point[0]],
                 [vor.vertices[i,1], far_point[1]], 'k--')

### plot traps
locations = traps[['Longitude', 'Latitude']].values
plt.scatter(locations[:,0], locations[:,1], marker='x')

### mark points
# for j, p in enumerate(points):
# 	plt.text(p[0]-0.03, p[1]+0.03, j, ha='right') # label the points

# set image boundaries
ax.set_xlim(-88, -87.5)
ax.set_ylim(41.6, 42.1)

plt.savefig('voronoized.png')