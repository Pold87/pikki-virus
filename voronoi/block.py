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