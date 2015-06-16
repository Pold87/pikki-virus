from scipy.spatial import Delaunay, ConvexHull
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools


locations = pd.read_csv('../unique_train.csv')[['Longitude', 'Latitude']]
locations = locations.drop_duplicates(['Longitude', 'Latitude'])
locations = locations.values
points = locations

### make delaunay diagram
tri = Delaunay(points)

### remove convex hull for more sensible graph
hull = ConvexHull(points)
delete_indices = [];
for tidx in range(tri.simplices.shape[0]):
	for hidx in range(hull.simplices.shape[0]):
		is_hull = (int((tri.simplices[tidx] == hull.simplices[hidx][0]).any()) + int((tri.simplices[tidx] == hull.simplices[hidx][1]).any())) == 2
		if is_hull:
			delete_indices.append(tidx)
delete_indices = sorted(np.sort(delete_indices), reverse=True)
for idx in delete_indices:
	tri.simplices = np.delete(tri.simplices, idx, 0)

### list neighbours
find_neighbours = lambda x,triang: list(set(indx for simplex in triang.simplices if x in simplex for indx in simplex if indx !=x))

### print graph
for pidx in range(points.size//2):
	neighbours = find_neighbours(pidx, tri)
	if neighbours != []:
		print("%d : %s\n" % (pidx, ' '.join([str(neighbour) for neighbour in neighbours])))

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

### plot delaunay vertices which have end point
plt.triplot(locations[:,0], locations[:,1], tri.simplices, 'k-')

### plot traps
plt.scatter(locations[:,0], locations[:,1], marker='x')

# set image boundaries
ax.set_xlim(-88, -87.5)
ax.set_ylim(41.6, 42.1)

plt.savefig('plot.png')