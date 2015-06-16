from scipy.spatial import Delaunay, ConvexHull
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools

data = pd.read_csv('../train.csv')[['Longitude', 'Latitude']]
locations = data.drop_duplicates(['Longitude', 'Latitude'])
locations = locations[['Longitude', 'Latitude']].values
points = locations
areas = np.c_[locations, np.zeros(locations.shape[0])]

for idx in range(locations.shape[0]):
	while True:
		try:
			area = int(input('%d/%d\tarea for (%f, %f): ' % (idx, locations.shape[0]-1, locations[idx,0], locations[idx,1])))
			areas[idx,2] = area
			for idx in range(areas.shape[0]):
				print('(%f, %f) : %d' % (areas[idx,0], areas[idx,1], areas[idx,2]))
			break
		except ValueError:
			print('Must be int value')

for idx in range(areas.shape[0]):
	print('(%f, %f) : %d' % (areas[idx,0], areas[idx,1], areas[idx,2]))
