from scipy.spatial import Delaunay, distance, Voronoi
from collections import defaultdict
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

mapdata = np.loadtxt("../mapdata_copyright_openstreetmap_contributors.txt")
aspect = mapdata.shape[0] * 1.0 / mapdata.shape[1]
lon_lat_box = (-88, -87.5, 41.6, 42.1)

fig = plt.figure(figsize=(10,14))
ax = fig.gca()
plt.imshow(mapdata,
           cmap=plt.get_cmap('gray'),
           extent=lon_lat_box,
           aspect=aspect)

traps = pd.read_csv('../train.csv')[['Longitude', 'Latitude']]
locations = traps[['Longitude', 'Latitude']].values
points = locations

ax.set_xlim(-88, -87.5)
ax.set_ylim(41.6, 42.1)

plt.savefig('1.png')

plt.scatter(locations[:,0], locations[:,1], marker='x')

plt.savefig('2.png')

vor = Voronoi(points)

for simplex in vor.ridge_vertices:
    simplex = np.asarray(simplex)
    if np.all(simplex >= 0):
        plt.plot(vor.vertices[simplex,0], vor.vertices[simplex,1], 'w-')

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
                 [vor.vertices[i,1], far_point[1]], 'w--')

plt.scatter(locations[:,0], locations[:,1], marker='x')

plt.savefig('3.png')

tri = Delaunay(points)
simplices = tri.simplices

### plot delaunay vertices which have end point
plt.triplot(locations[:,0], locations[:,1], tri.simplices, 'k-')

plt.scatter(locations[:,0], locations[:,1], marker='x')

plt.savefig('4.png')

### list neighbours
threshold = .077
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

plt.triplot(locations[:,0], locations[:,1], tri.simplices, 'r-')
for simplex in simplices:
    plt.plot([locations[simplex[0],0], locations[simplex[1],0]], [locations[simplex[0],1], locations[simplex[1],1]], 'k-')
    plt.plot([locations[simplex[1],0], locations[simplex[2],0]], [locations[simplex[1],1], locations[simplex[2],1]], 'k-')
    plt.plot([locations[simplex[0],0], locations[simplex[2],0]], [locations[simplex[0],1], locations[simplex[2],1]], 'k-')


plt.scatter(locations[:,0], locations[:,1], marker='x')

plt.savefig('5.png')


