import pandas as pd
import numpy as np


data = pd.read_csv('../unique_train.csv')
da
locations = locations.drop_duplicates(['Longitude', 'Latitude'])
locations = locations.values
points = locations

### make delaunay diagram
tri = Delaunay(points)
find_neighbours = lambda x,triang: list(set(indx for simplex in triang.simplices if x in simplex for indx in simplex if indx !=x))

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
		mappingfile.write("%d : %f %f\n" % (pidx, points[pidx][0], points[pidx][1]))