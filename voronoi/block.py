from scipy.spatial import Delaunay, ConvexHull
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# locations = pd.read_csv('../unique_train.csv')[['Longitude', 'Latitude', 'Block']]
# locations = locations.drop_duplicates(['Longitude', 'Latitude', 'Block'])	

# grouped = locations.groupby( 'Block')

# points = grouped.mean().values

# ### plot map
# mapdata = np.loadtxt("../mapdata_copyright_openstreetmap_contributors.txt")
# aspect = mapdata.shape[0] * 1.0 / mapdata.shape[1]
# lon_lat_box = (-88, -87.5, 41.6, 42.1)

# fig = plt.figure(figsize=(10,14))
# ax = fig.gca()
# plt.imshow(mapdata,
#            cmap=plt.get_cmap('gray'),
#            extent=lon_lat_box,
#            aspect=aspect)

# ### plot traps
# plt.scatter(points[:,0], points[:,1], marker='x')

# # set image boundaries
# # ax.set_xlim(-88, -87.5)
# # ax.set_ylim(41.6, 42.1)

# plt.savefig('plot.png')

graph = {}
block = input('block number (q to quit): ')

while block != 'q':
	try:
		int(block)
		try:
			graph[block]
		except KeyError:
			graph[block] = []

		neighbours = input('neighbours: ').split(' ')

		for neighbour in neighbours:
			try:
				graph[neighbour]
			except KeyError:
				graph[neighbour] = []
			graph[block].append(neighbour)
			graph[block] = pd.unique(graph[block]).tolist()
			graph[neighbour].append(block)
			graph[neighbour] = pd.unique(graph[neighbour]).tolist()

		print("%d\n" % len(graph.keys()))
		for block in graph:
			print("%s %d %s\n" % (block, len(graph[block]), ' '.join(map(str, graph[block]))))

	except ValueError:
		print("enter int or q")

	block = input('block number (q to quit): ')