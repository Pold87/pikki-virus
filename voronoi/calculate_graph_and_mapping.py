from scipy.spatial import Delaunay, ConvexHull
import pandas as pd
import numpy as np


locations = pd.read_csv('../unique_train.csv')[['Longitude', 'Latitude']]
locations = locations.drop_duplicates(['Longitude', 'Latitude'])
locations = locations.values
points = locations

### make delaunay diagram
tri = Delaunay(points)
find_neighbours = lambda x,triang: list(set(indx for simplex in triang.simplices if x in simplex for indx in simplex if indx !=x))

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

### write graph file
graphfile=open('./graph.graph', 'w+')
for pidx in range(points.size//2):
	neighbours = find_neighbours(pidx, tri)
	if neighbours != []:
		graphfile.write("%d : %d %s\n" % (pidx, len(neighbours), ' '.join([str(neighbour) for neighbour in neighbours])))

mappingfile=open('./graph.mapping', 'w+')
for pidx in range(points.size//2):
	neighbours = find_neighbours(pidx, tri)
	if neighbours != []:
		mappingfile.write("(%f , %f) : %d\n" % (points[pidx][0], points[pidx][1], pidx))