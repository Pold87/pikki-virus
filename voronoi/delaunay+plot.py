from scipy.spatial import Delaunay, delaunay_plot_2d
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
plt.triplot(locations[:,0], locations[:,1], tri.simplices.copy(), 'k-')

### plot traps
plt.scatter(locations[:,0], locations[:,1], marker='x')

### mark points
# for j, p in enumerate(points):
# 	plt.text(p[0]-0.03, p[1]+0.03, j, ha='right') # label the points

# set image boundaries
ax.set_xlim(-88, -87.5)
ax.set_ylim(41.6, 42.1)

plt.savefig('delaunyzed.png')
