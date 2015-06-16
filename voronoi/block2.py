from scipy.spatial import Delaunay, ConvexHull
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools

data = pd.read_csv('../train.csv')[['Longitude', 'Latitude', 'AddressNumberAndStreet']]
locations = data.drop_duplicates(['Longitude', 'Latitude', 'AddressNumberAndStreet'])
blocks = data.drop_duplicates(['Longitude', 'Latitude', 'AddressNumberAndStreet'])
locations = locations[['Longitude', 'Latitude']].values
points = locations

### plot map
mapdata = np.loadtxt("../mapdata_copyright_openstreetmap_contributors.txt")
aspect = mapdata.shape[0] * 1.0 / mapdata.shape[1]
lon_lat_box = (-88, -87.5, 41.6, 42.1)

fig = plt.figure(figsize=(4*10,4*14))
ax = fig.gca()
plt.imshow(mapdata,
           cmap=plt.get_cmap('gray'),
           extent=lon_lat_box,
           aspect=aspect)

### plot traps
plt.scatter(locations[:,0], locations[:,1], marker='x')

for idx, block in blocks.iterrows():
	plt.text(block['Longitude'], block['Latitude'], '(%f, %f)' % (block['Longitude'], block['Latitude']))

# set image boundaries
ax.set_xlim(-88, -87.5)
ax.set_ylim(41.6, 42.1)

plt.savefig('plot2.png')