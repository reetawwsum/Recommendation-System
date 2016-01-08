from similarity import Similarity

class Recommendation:
	'Various recommendation systems'

	def __init__(self, dataset):
		'''
		Structure of dataset
		dataset = {
			key1: {
				inner_key1: inner_value1,
				inner_key2: inner_value2,
				.......
			},
			key2: {
				inner_key1: inner_value1,
				inner_key2: inner_value2,
				.......
			},
			.......
		}
		'''
		self.dataset = dataset

	def collaborativeRecommendation(self, key, n=3):
		'''Return n scores along with inner keys which are top match'''

		dataset = self.dataset
		weighted_inner_values = {}
		total_scores = {}

		for other_key in dataset:
			if other_key == key:
				continue

			# Fetching common inner keys to calculate similarity score
			common_inner_keys = self.fetchCommonInnerKeys(key, other_key)

			x = [dataset[key][inner_key] for inner_key in common_inner_keys]
			y = [dataset[other_key][inner_key] for inner_key in common_inner_keys]

			# Finding similarity score
			sim = Similarity()
			score = sim.pearson(x, y)

			# Ignoring scores of zero or below
			if score <= 0:
				continue

			for inner_key in dataset[other_key]:
				if inner_key not in dataset[key] or dataset[key][inner_key] == 0:
					weighted_inner_values.setdefault(inner_key, 0)
					weighted_inner_values[inner_key] += score * dataset[other_key][inner_key]
					total_scores.setdefault(inner_key, 0)
					total_scores[inner_key] += score

		scores = [(weighted_inner_values[inner_key]/total_scores[inner_key], inner_key) for inner_key in weighted_inner_values]

		# Sorting the list so that highest score appear at the top
		scores.sort()
		scores.reverse()

		return scores[0:n]

	def collaborativeFiltering(self, key, n=3):
		'''Return n scores along with other keys which are top match'''

		dataset = self.dataset
		scores = []

		for other_key in dataset:
			if other_key == key:
				continue

			# Fetching common inner keys to calculate similarity score
			common_inner_keys = self.fetchCommonInnerKeys(key, other_key)

			x = [dataset[key][inner_key] for inner_key in common_inner_keys]
			y = [dataset[other_key][inner_key] for inner_key in common_inner_keys]

			# Appending the similarity score to a list
			sim = Similarity()
			scores.append((sim.pearson(x, y), other_key))

		# Sorting the list so the highest score appear at the top
		scores.sort()
		scores.reverse()

		return scores[0:n]

	# Helper functions

	def fetchCommonInnerKeys(self, key1, key2):

		dataset = self.dataset

		inner_keys = [inner_key for inner_key in dataset[key1] if inner_key in dataset[key2]]

		return inner_keys

