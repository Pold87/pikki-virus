from scipy.spatial import Delaunay, ConvexHull, distance
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import sys


locations = pd.read_csv('../unique_train.csv')[['Longitude', 'Latitude']]
locations = locations.drop_duplicates(['Longitude', 'Latitude'])
locations = locations.values
points = locations

threshold = float(sys.argv[1])

### make delaunay diagram
tri = Delaunay(points, incremental=True)
simplices = tri.simplices	

### list neighbours
find_neighbours = lambda x,simplices: list(set(indx for simplex in simplices if x in simplex for indx in simplex if indx !=x))
neighbours = {}
for pidx in range(points.size//2):
	local_neighbours = find_neighbours(pidx, simplices)
	neighbours[pidx] = local_neighbours
	# print("%d : %s\n" % (pidx, ' '.join([str(neighbour) for neighbour in local_neighbours])))

### find points which have a distance larger than threshold
delete_indices = [];
distances = []
for point, local_neighbours in neighbours.items():
	for neighbour in local_neighbours:
		distances.append(distance.euclidean(locations[point], locations[neighbour]))
		if distances[-1] > threshold:
			for tidx in range(simplices.shape[0]):
				is_too_long = (int((simplices[tidx] == point).any()) + int((simplices[tidx] == neighbour).any())) == 2
				if is_too_long:
					delete_indices.append(tidx)

### delete those points
delete_indices = sorted(np.sort(np.unique(delete_indices)), reverse=True)
for idx in delete_indices:
	simplices = np.delete(simplices, idx, 0)

### update neighbours
for pidx in range(points.size//2):
	local_neighbours = find_neighbours(pidx, simplices)
	neighbours[pidx] = local_neighbours

### print graph
for pidx in range(points.size//2):
	print("%d : %d %s\n" % (pidx, len(neighbours[pidx]), ' '.join([str(neighbour) for neighbour in neighbours[pidx]])))

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
plt.triplot(locations[:,0], locations[:,1], tri.simplices, 'g-')
for simplex in simplices:
	plt.plot([locations[simplex[0],0], locations[simplex[1],0]], [locations[simplex[0],1], locations[simplex[1],1]], 'k-')
	plt.plot([locations[simplex[1],0], locations[simplex[2],0]], [locations[simplex[1],1], locations[simplex[2],1]], 'k-')
	plt.plot([locations[simplex[0],0], locations[simplex[2],0]], [locations[simplex[0],1], locations[simplex[2],1]], 'k-')

### plot traps
plt.scatter(locations[:,0], locations[:,1], marker='x')

# set image boundaries
ax.set_xlim(-88, -87.5)
ax.set_ylim(41.6, 42.1)

plt.savefig('plot.png')