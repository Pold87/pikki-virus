import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.neighbors import KernelDensity

#plot map
mapdata = np.loadtxt("../mapdata_copyright_openstreetmap_contributors.txt")
aspect = mapdata.shape[0] * 1.0 / mapdata.shape[1]
lon_lat_box = (-88, -87.5, 41.6, 42.1)
plt.figure(figsize=(10,14))
plt.imshow(mapdata,
           cmap=plt.get_cmap('gray'),
           extent=lon_lat_box,
           aspect=aspect)

###read data, group by mosquito counts
traps = pd.read_csv('../train.csv')[['Longitude', 'Latitude', 'Species', 'NumMosquitos']]
mosquis = traps.groupby(['Species', 'Longitude', 'Latitude']).count()['NumMosquitos']
mosquistypes = mosquis.index.levels[0]

###plot CULEX <num>
#  0: CULEX ERRATICUS
#  1: CULEX PIPIENS
#  2: CULEX PIPIENS/RESTUANS
#  3: CULEX RESTUANS
#  4: CULEX SALINARIUS
#  5: CULEX TARSALIS
#  6: CULEX TERRITANS

for num in range(0, 7):
	num = 6
	alpha_cm = plt.cm.Reds
	alpha_cm._init()
	alpha_cm._lut[:-3,-1] = abs(np.logspace(0, 1, alpha_cm.N) / 10 - 1)[::-1]
	
	X = mosquis[mosquistypes[num]].reset_index()[['Longitude', 'Latitude']].values
	kd = KernelDensity(bandwidth=0.02)
	kd.fit(X)
	
	xv,yv = np.meshgrid(np.linspace(-88, -87.5, 100), np.linspace(41.6, 42.1, 100))
	gridpoints = np.array([xv.ravel(),yv.ravel()]).T
	zv = np.exp(kd.score_samples(gridpoints).reshape(100,100))
	plt.imshow(zv,
	          origin='lower',
	          cmap=alpha_cm,
	          extent=lon_lat_box,
	          aspect=aspect)
	
	#mark traps
	locations = traps[['Longitude', 'Latitude']].values
	plt.scatter(locations[:,0], locations[:,1], marker='x')
	
	plt.savefig('heatmap-species' + str(num) + '.png')
