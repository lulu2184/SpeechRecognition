import sys
import numpy as np

class VQset:
	training_set = []
	n_clusters = 20
	dimension = 0
	centers = []
	d_thred = 1.0
	def __init__(self, dimension):
		self.dimension = dimension

	def __init__(self):
		pass

	def set_dimension(self, x):
		self.dimension = x

	def add_samples(self, samples):
		for sample in samples:
			self.add_sample(sample)

	def add_sample(self, sample):
		if self.dimension == 0:
			self.dimension = len(sample)
		if len(sample) != self.dimension:
			print "Incorrect dimension while adding sample"
			sys.exit()
		self.training_set.append(sample)

	def training(self):
		dmax = dmin = self.training_set[0]
		for sample in self.training_set:
			dmax = [max(v1, v2) for (v1, v2) in zip(dmax, sample)]
			dmin = [min(v1, v2) for (v1, v2) in zip(dmin, sample)]
		dmax = np.array(dmax)
		dmin = np.array(dmin)
		for i in range(self.n_clusters):
			center = np.random.rand(self.dimension) * (dmax - dmin) + dmin
			center = [int(v) for v in center]
			self.centers.append(center)
		dlast = -1e100 
		dnow, choices = self.__distance_sum_choice()
		while abs((dnow - dlast) / dnow) > self.d_thred:
			self.__update_centers(choices)
			dlast = dnow
			dnow, choices = self.__distance_sum_choice()

	def quantization(self, feature):
		dmin = self.calculate_dist(feature, self.centers[0])
		choice = 1
		for (i, center) in enumerate(self.centers):
			d = self.calculate_dist(feature, center)
			if dmin > d:
				dmin = d
				choice = i
		return (dmin, choice)

	def calculate_dist(self, v1, v2):
		return np.sum(np.square(v1 - v2))

	def __distance_sum_choice(self):
		choices = []
		dist_sum = 0
		for sample in self.training_set:
			dmin, choice = self.quantization(sample)
			choices.append(choice)
			dist_sum += dmin
		return (dist_sum, choices)

	def __update_centers(self, choices):
		counter = np.zeros(self.n_clusters)
		centers = np.zeros((self.n_clusters, self.dimension))
		for (sample, choice) in zip(self.training_set, choices):
			counter[choice] += 1.0
			centers[choice] += sample
		for (i, (center, c)) in enumerate(zip(centers, counter)):
			if c > 0:
				self.centers[i] = center / c





