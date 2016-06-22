import sys
import numpy as np

class VQset:
	training_set = []
	n_clusters = 256
	dimension = 0
	centers = []
	dthred = 1.0
	def __init__(self, dimension):
		self.dimension = dimension

	def add_samples(self, samples):
		for sample in samples:
			self.add_sample(sample)

	def add_sample(self, sample):
		if len(sample) != dimension:
			print "Incorrect dimension while adding sample"
			sys.exit()
		self.training_set.append(sample)

	def training(self):
		dmax = dmin = self.training_set[0]
		for sample in self.training_set:
			dmax = [max(v1, v2) for (v1, v2) in zip(dmax, sample)]
			dmin = [min(v1, v2) for (v1, v2) in zip(dmin, sample)]
		for i in range(self.n_clusters):
			center = np.random.rand(self.dimension) * (dmax - dmin) + dmin
			center = [int(v) for v in center]
			self.centers.append(center)
		dlast = -1e100 
		dnow, choices = self.__distance_sum_choice()
		while abs((dnow - dlast) / dnow) > d_thred:
			self.__update_centers(choices)
			dlast = dnow
			dnow, choices = self.__distance_sum_choice()

	def __distance_sum_choice(self):
		for sample in self.training_set:
			dmin = self.calculate_dist(sample, self.centers[0])
			choice = 1
			for (i, center) in enumerate(self.centers):
				d = self.calculate_dist(sample, center)
				if dmin > d:
					dmin = d
					choice = i
			choices.append(choice)
			dist_sum += dmin
		return (dist_sum, choice)

	def __update_centers(self, choices):
		counter = np.zeros(self.n_clusters)
		centers = np.zeros((self.n_clusters, self.dimension))
		for (sample, choice) in zip(self.training_set, choices):
			counter[choice] += 1.0
			centers[choice] += sample
		self.centers = [center / c for (center, c) in zip(centers, counter)]





