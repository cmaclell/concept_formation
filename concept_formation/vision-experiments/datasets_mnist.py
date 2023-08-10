import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
import copy
import random



def MNIST_dataset(split='train', pad=False, normalize=False, permutation=False, 
				download=True, verbose=True):
	"""
	Load the original MNIST training/test dataset.
	"""

	dataset_class = datasets.MNIST
	transform = [transforms.ToTensor(), transforms.Pad(2)] if pad else [transforms.ToTensor()]
	if normalize:
		transform.append(transforms.Normalize((0.1307,), (0.3081,)))  # mean and std for all pixels in MNIST
		# transform.append(transforms.Normalize((0.5,), (0.5,)))
	if permutation:
		transform.append(transforms.Lambda(lambda x, p=permutation: permutate_image_pixels(x, p)))
	dataset_transform = transforms.Compose(transform)

	# Load dataset:
	dataset = dataset_class('./datasets/MNIST',
							train=False if split=='test' else True,
							download=download,
							transform=dataset_transform)
	if verbose:
		print("MNIST {} dataset consisting of {} samples.".format(split, len(dataset)))

	return dataset



def get_data_loader(dataset, batch_size, cuda=False, drop_last=False, shuffle=False):
	"""
	Return <DataLoader> object for the provided dataset object.
	"""
	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
					  **({'num_workers': 0, 'pin_memory': True} if cuda else {}))


def test_loaders(te_dataset, general_config, data_config, verbose=True):
	"""
	Generate test DataLoaders for all experiment cases.
	"""
	label = general_config['label']
	cuda = general_config['cuda']
	seed = general_config['seed']

	drop_last = data_config['drop_last']
	shuffle = data_config['shuffle']
	split_size = data_config['split_size']
	batch_size_te = data_config['batch_size_te']

	# Set random seeds:
	np.random.seed(seed)
	torch.manual_seed(seed)
	random.seed(seed)

	if verbose:
		print("\n\n " +' Loading test DataLoaders '.center(70, '*'))

	test_datasets = []
	for d in range(10):
		single_digit_indices = [i for i in range(len(te_dataset)) if te_dataset[i][1] == d]
		test_datasets.append(Subset(te_dataset, single_digit_indices))
	test_datasets.append(te_dataset)

	test_loaders = [get_data_loader(
		test_dataset,
		batch_size=batch_size_te,
		cuda=cuda,
		drop_last=drop_last,
		shuffle=shuffle) for test_dataset in test_datasets]

	if verbose:
		print(' Loading test DataLoaders successful '.center(70, '*'))

	labels_te = []
	for i in range(10):
		labels_te.append(i)
	labels_te.append(tuple(range(10)))

	return test_loaders, labels_te



class dataloaders_0(object):
	"""
	Corresponds to the case:
	All trained data are shuffled and trained in sequential splits.
	"""

	def __init__(self, general_config, data_config, dataset_tr, dataset_te, verbose):

		self.cuda = general_config['cuda']
		self.seed = general_config['seed']

		self.shuffle = data_config['shuffle']
		self.batch_size_tr = data_config['batch_size_tr']
		self.batch_size_te = data_config['batch_size_te']
		self.split_size = data_config['split_size']
		if self.split_size > len(dataset_tr):
			raise ValueError("split_size exceeds the size of the initial training dataset.")
		self.normalize = data_config['normalize']
		self.pad = data_config['pad']
		self.permutation = data_config['permutation']
		self.drop_last = data_config['drop_last']

		# Set random seeds:
		np.random.seed(self.seed)
		torch.manual_seed(self.seed)
		random.seed(self.seed)

		self.verbose = verbose
		self.training_loaders, self.n_splits, self.labels_tr, self.size_subsets = self.training_loaders(dataset_tr)
		self.test_loaders, self.labels_te = test_loaders(dataset_te, general_config, data_config, verbose=verbose)
		if verbose:
			print("\nAn overview of the dataloaders for experiments 0:")
			print("Training dataloaders: (0-9: 1), ..., (0-9: {})".format(self.n_splits) + ", each split with {} data.".format(self.split_size))
			print("Test dataloaders: (0), (1), ..., (9), (0-9), each with all the test data available,")
			print("(Approx. 6000 for each label and 10000 for all labels).")
			print("The actual size of each tr subset:", self.size_subsets)


	def training_loaders(self, tr_dataset):

		if self.verbose:
			print("\n\n " +' Loading Training DataLoader for Experiments 0 '.center(70, '*'))

		split_size = self.split_size

		tr_dataset_indices = list(range(len(tr_dataset)))
		if self.shuffle:
			random.shuffle(tr_dataset_indices)

		subset_datasets = []
		n_splits = len(tr_dataset) // split_size
		if len(tr_dataset) % split_size != 0:
			n_splits += 1

		for i in range(n_splits):
			start = i * split_size
			end = (i + 1) * split_size if i < n_splits - 1 else len(tr_dataset_indices)
			subsets_indices = tr_dataset_indices[start:end]
			subset_datasets.append(Subset(tr_dataset, subsets_indices))

		subset_loaders = [get_data_loader(
			subset_dataset, 
			batch_size=self.batch_size_tr, 
			cuda=self.cuda, 
			drop_last=self.drop_last, 
			shuffle=self.shuffle) for subset_dataset in subset_datasets]

		if self.verbose:
			print(' Loading training DataLoaders successful '.center(70, '*'))

		# labels exist among the tr-datasets:
		labels_tr = [list(range(10))] * 10

		# size of each tr dataset:
		size_subsets = [len(subset_dataset) for subset_dataset in subset_datasets]

		return subset_loaders, n_splits, labels_tr, size_subsets



class dataloaders_1(object):
	"""
	Corresponds to the case:
	First train all data from the specified label (0 for instance)
	Then train all shuffled data from remaining labels in sequential splits.
	"""

	def __init__(self, general_config, data_config, dataset_tr, dataset_te, verbose):

		self.cuda = general_config['cuda']
		self.seed = general_config['seed']
		self.label = general_config['label']

		self.shuffle = data_config['shuffle']
		self.batch_size_tr = data_config['batch_size_tr']
		self.batch_size_te = data_config['batch_size_te']
		self.n_split = data_config['n_split']
		# if self.split_size > len(dataset_tr):
		# 	raise ValueError("split_size exceeds the size of the initial training dataset.")
		self.normalize = data_config['normalize']
		self.pad = data_config['pad']
		self.permutation = data_config['permutation']
		self.drop_last = data_config['drop_last']

		# Set random seeds:
		np.random.seed(self.seed)
		torch.manual_seed(self.seed)
		random.seed(self.seed)

		self.verbose = verbose
		self.training_loaders, self.n_rest_split, self.labels_tr, self.size_subsets = self.training_loaders(dataset_tr)
		self.test_loaders, self.labels_te = test_loaders(dataset_te, general_config, data_config, verbose=verbose)
		if verbose:
			print("\nAn overview of the dataloaders for experiments 1:")
			print("Training dataloaders:")
			print("\t({}) with all available data,".format(self.label))
			print("\t(0-9 except {}: 1), ..., (0-9 except {}: {})".format(self.label, self.label, self.n_rest_split))
			print("Test dataloaders: (0), (1), ..., (9), (0-9), each with all the test data available,")
			print("\t(Approx. 6000 for each label and 10000 for all labels).")
			print("The actual size of each tr subset:", self.size_subsets)

	def training_loaders(self, tr_dataset):
		if self.verbose:
			print("\n\n " +' Loading Training DataLoader for Experiments 1 '.center(70, '*'))

		n_split = self.n_split

		# Trial one:
		single_indices = [i for i in range(len(tr_dataset)) if tr_dataset[i][1] == self.label]
		# if self.shuffle:
		# 	random.shuffle(single_indices)
		single_dataset_tr = Subset(tr_dataset, single_indices)

		# Trial two:
		remaining_indices = set(range(len(tr_dataset)))
		remaining_indices -= set(single_indices)
		remaining_indices = list(remaining_indices)
		if self.shuffle:
			random.shuffle(remaining_indices)

		subset_datasets = [single_dataset_tr]
		n_rest_split = n_split - 1  # nbr of splits for rest labels
		split_size = len(remaining_indices) // n_rest_split
		# n_splits = len(remaining_indices) // split_size
		# if len(tr_dataset) % split_size != 0:
		# 	n_splits += 1
		for i in range(n_rest_split):
			start = i * split_size
			end = (i + 1) * split_size if i < n_rest_split - 1 else len(remaining_indices)
			subsets_indices = remaining_indices[start:end]
			subset_datasets.append(Subset(tr_dataset, subsets_indices))

		subset_loaders = []
		for subset_dataset in subset_datasets:
			subset_loaders.append(
					get_data_loader(
					subset_dataset, 
					batch_size=self.batch_size_tr, 
					cuda=self.cuda, 
					drop_last=self.drop_last, 
					shuffle=self.shuffle,
					)
				)
		if self.verbose:
			print(' Loading training DataLoaders successful '.center(70, '*'))

		# labels exist among the tr-datasets:
		labels_tr = [[self.label]]
		labels_tr_splits = list(range(10))
		labels_tr_splits.remove(self.label)
		labels_tr += [labels_tr_splits] * n_rest_split

		# size of each tr dataset:
		size_subsets = [len(subset_dataset) for subset_dataset in subset_datasets]

		return subset_loaders, n_rest_split, labels_tr, size_subsets



class dataloaders_2(object):
	"""
	Corresponds to the case:
	First train all data from the specified label (0 for instance) along with the same portion of other labels,
	Then train all remaining shuffled data from remaining labels in sequential splits.
	"""

	def __init__(self, general_config, data_config, dataset_tr, dataset_te, verbose):

		self.cuda = general_config['cuda']
		self.seed = general_config['seed']
		self.label = general_config['label']

		self.shuffle = data_config['shuffle']
		self.batch_size_tr = data_config['batch_size_tr']
		self.batch_size_te = data_config['batch_size_te']
		self.n_split = data_config['n_split']
		# if self.split_size > len(dataset_tr):
		# 	raise ValueError("split_size exceeds the size of the initial training dataset.")
		self.normalize = data_config['normalize']
		self.pad = data_config['pad']
		self.permutation = data_config['permutation']
		self.drop_last = data_config['drop_last']
		self.size_all_tr_each = data_config['size_all_tr_each']

		# Set random seeds:
		np.random.seed(self.seed)
		torch.manual_seed(self.seed)
		random.seed(self.seed)

		self.verbose = verbose
		self.training_loaders, self.n_rest_split, self.labels_tr, self.size_subsets = self.training_loaders(dataset_tr)
		self.test_loaders, self.labels_te = test_loaders(dataset_te, general_config, data_config, verbose=verbose)
		if verbose:
			print("\nAn overview of the dataloaders for experiments 2:")
			print("Training dataloaders:")
			print("\t({}) with all available data plus {} samples from each other labels,".format(self.label, self.size_all_tr_each))
			print("\t(0-9 except {}: 1), ..., (0-9 except {}: {})".format(self.label, self.label, self.n_rest_split))
			print("Test dataloaders: (0), (1), ..., (9), (0-9), each with all the test data available,")
			print("\t(Approx. 6000 for each label and 10000 for all labels).")
			print("The actual size of each tr dataset:", self.size_subsets)

	def training_loaders(self, tr_dataset):
		if self.verbose:
			print("\n\n " +' Loading Training DataLoader for Experiments 2 '.center(70, '*'))

		n_split = self.n_split

		# Trial one:
		digit_subsets = {}  # (digit, dataset.Subset)
		for digit in range(10):
			digit_indices = [i for i in range(len(tr_dataset)) if tr_dataset[i][1] == digit]
			if digit == self.label:
				digit_subset_indices = digit_indices
			else:
				digit_subset_indices = digit_indices[:self.size_all_tr_each]
			digit_subsets[digit] = Subset(tr_dataset, digit_subset_indices)
		all_prior_dataset_tr = ConcatDataset([digit_subsets[digit] for digit in range(10)])

		# Trial two:
		remaining_indices = set(range(len(tr_dataset)))
		for digit in range(10):
			remaining_indices -= set(digit_subsets[digit].indices)
		remaining_indices = list(remaining_indices)

		if self.shuffle:
			random.shuffle(remaining_indices)

		subset_datasets = [all_prior_dataset_tr]
		# n_splits = len(remaining_indices) // split_size
		# split_size = len(remaining_indices) // n_split
		# if len(tr_dataset) % split_size != 0:
		# 	n_splits += 1
		n_rest_split = n_split - 1
		split_size = len(remaining_indices) // n_rest_split
		for i in range(n_rest_split):
			start = i * split_size
			end = (i + 1) * split_size if i < n_rest_split - 1 else len(remaining_indices)
			subsets_indices = remaining_indices[start:end]
			subset_datasets.append(Subset(tr_dataset, subsets_indices))

		subset_loaders = []
		for subset_dataset in subset_datasets:
			# print(n_splits)
			# print(split_size)
			# print(len(subset_dataset))
			subset_loaders.append(
					get_data_loader(
						subset_dataset, 
						batch_size=self.batch_size_tr, 
						cuda=self.cuda, 
						drop_last=self.drop_last, 
						shuffle=self.shuffle,
					)
				)
		if self.verbose:
			print(' Loading training DataLoaders successful '.center(70, '*'))

		# labels exist among the tr-datasets:
		labels_tr = [list(range(10))]
		labels_tr_splits = list(range(10))
		labels_tr_splits.remove(self.label)
		labels_tr += [labels_tr_splits] * n_rest_split

		# size of each tr dataset:
		size_subsets = [len(subset_dataset) for subset_dataset in subset_datasets]

		return subset_loaders, n_rest_split, labels_tr, size_subsets


class dataloaders_3(object):
	"""
	Corresponds to the `relearning` case:
	Similar to experiments 2, but inlude about 50% specified-label data in the first loader only,
	and after some splits (say 4), include the rest 50% of specified-label data by spliting evenly to the rest splits.
	"""
	def __init__(self, general_config, data_config, dataset_tr, dataset_te, verbose):

		self.cuda = general_config['cuda']
		self.seed = general_config['seed']
		self.label = general_config['label']

		self.shuffle = data_config['shuffle']
		self.batch_size_tr = data_config['batch_size_tr']
		self.batch_size_te = data_config['batch_size_te']
		self.n_split = data_config['n_split']
		# if self.split_size > len(dataset_tr):
		# 	raise ValueError("split_size exceeds the size of the initial training dataset.")
		self.normalize = data_config['normalize']
		self.pad = data_config['pad']
		self.permutation = data_config['permutation']
		self.drop_last = data_config['drop_last']
		self.size_all_tr_each = data_config['size_all_tr_each']
		self.n_relearning = data_config['n_relearning']  # Number of splits for relearning

		# Set random seeds:
		np.random.seed(self.seed)
		torch.manual_seed(self.seed)
		random.seed(self.seed)

		self.verbose = verbose
		self.training_loaders, self.n_rest_split, self.labels_tr, self.size_subsets = self.training_loaders(dataset_tr)
		self.test_loaders, self.labels_te = test_loaders(dataset_te, general_config, data_config, verbose=verbose)
		if verbose:
			print("\nAn overview of the dataloaders for experiments 3:")
			print("Training dataloaders:")
			print("\t({}) with half available data plus {} samples from each other labels,".format(self.label, self.size_all_tr_each))
			print("\t(0-9 except {}: 1), ..., (0-9 except {}: {})".format(self.label, self.label, self.n_rest_split - self.n_relearning))
			print("\t(0-9: 1), ..., (0-9: {})".format(self.label, self.label, self.n_relearning))
			print("Test dataloaders: (0), (1), ..., (9), (0-9), each with all the test data available,")
			print("\t(Approx. 6000 for each label and 10000 for all labels).")
			print("The actual size of each tr dataset:", self.size_subsets)


	def training_loaders(self, tr_dataset):
		if self.verbose:
			print("\n\n " +' Loading Training DataLoader for Experiments 3 '.center(70, '*'))

		n_split = self.n_split

		# First have the subset of all the ones with specified label:
		single_indices = [i for i in range(len(tr_dataset)) if tr_dataset[i][1] == self.label]
		# if self.shuffle:
		# 	random.shuffle(single_indices)
		single_dataset_tr = Subset(tr_dataset, single_indices)

		single_former_size = len(single_indices) // 2  # nbr of samples with specified label included in the first subset
		single_indices_former = single_indices[:single_former_size]
		single_indices_latter = single_indices[single_former_size:]
		single_latter_size = len(single_indices_latter) // self.n_relearning  # nbr of samples with specified label included in the last n_relearning subsets

		# Trial 1:
		digit_subsets = {}  # (digit, dataset.Subset)
		for digit in range(10):
			digit_indices = [i for i in range(len(tr_dataset)) if tr_dataset[i][1] == digit]
			if digit == self.label:
				digit_subset_indices = single_indices_former
			else:
				digit_subset_indices = digit_indices[:self.size_all_tr_each]
			digit_subsets[digit] = Subset(tr_dataset, digit_subset_indices)
		all_prior_dataset_tr = ConcatDataset([digit_subsets[digit] for digit in range(10)])

		# Trial 2:
		remaining_indices = set(range(len(tr_dataset)))
		for digit in range(10):
			remaining_indices -= set(digit_subsets[digit].indices)
		remaining_indices -= set(single_indices_latter)
		remaining_indices = list(remaining_indices)

		if self.shuffle:
			random.shuffle(remaining_indices)

		subset_datasets = [all_prior_dataset_tr]

		n_rest_split = n_split - 1
		split_size = len(remaining_indices) // n_rest_split
		# if len(tr_dataset) % split_size != 0:
		# 	n_splits += 1
		if self.n_relearning > n_rest_split:
			raise ValueError("The number of splits assigned for relearning of the specified label exceeds the total number of rest splits.")
		index_start_relearning = n_split - self.n_relearning
		
		for i in range(n_rest_split):
			start = i * split_size
			end = (i + 1) * split_size if i < n_rest_split - 1 else len(remaining_indices)
			subsets_indices = remaining_indices[start:end]
			subset_datasets.append(Subset(tr_dataset, subsets_indices))

		# print(index_start_relearning)
		# print(n_splits)
		# print(single_latter_size)
		# Trial 3:
		for i in range(self.n_relearning):
			start = i * single_latter_size
			end = (i + 1) * single_latter_size if i < self.n_relearning - 1 else len(single_indices_latter)
			relearning_indices = single_indices_latter[start:end]
			# print(start, end)
			subset_datasets[index_start_relearning + i] = ConcatDataset([subset_datasets[index_start_relearning + i], Subset(tr_dataset, relearning_indices)])

		subset_loaders = []
		for subset_dataset in subset_datasets:
			subset_loaders.append(
					get_data_loader(
					subset_dataset, 
					batch_size=self.batch_size_tr, 
					cuda=self.cuda, 
					drop_last=self.drop_last, 
					shuffle=self.shuffle,
					)
				)
		if self.verbose:
			print(' Loading training DataLoaders successful '.center(70, '*'))

		# labels exist among the tr-datasets:
		labels_tr = [list(range(10))]
		labels_tr_splits = list(range(10))
		labels_tr_splits.remove(self.label)
		labels_tr += [labels_tr_splits] * (n_rest_split - self.n_relearning)
		labels_tr += [list(range(10))] * self.n_relearning

		# size of each tr dataset:
		size_subsets = [len(subset_dataset) for subset_dataset in subset_datasets]

		return subset_loaders, n_rest_split, labels_tr, size_subsets



class dataloaders_4(object):
	"""
	This is a `bonus` one that also corresponds to the `relearning` case:
	Similar to experiments 3, but choose the splits with chosen labels in the  middle only instead of last splits.
	"""
	def __init__(self, general_config, data_config, dataset_tr, dataset_te, verbose):

		self.cuda = general_config['cuda']
		self.seed = general_config['seed']
		self.label = general_config['label']

		self.shuffle = data_config['shuffle']
		self.batch_size_tr = data_config['batch_size_tr']
		self.batch_size_te = data_config['batch_size_te']
		self.n_split = data_config['n_split']
		# if self.split_size > len(dataset_tr):
		# 	raise ValueError("split_size exceeds the size of the initial training dataset.")
		self.normalize = data_config['normalize']
		self.pad = data_config['pad']
		self.permutation = data_config['permutation']
		self.drop_last = data_config['drop_last']
		self.size_all_tr_each = data_config['size_all_tr_each']
		self.n_relearning = data_config['n_relearning']  # Number of splits for relearning

		# Set random seeds:
		np.random.seed(self.seed)
		torch.manual_seed(self.seed)
		random.seed(self.seed)

		self.verbose = verbose
		self.training_loaders, self.n_rest_split, self.labels_tr, self.size_subsets = self.training_loaders(dataset_tr)
		self.test_loaders, self.labels_te = test_loaders(dataset_te, general_config, data_config, verbose=verbose)
		if verbose:
			print("\nAn overview of the dataloaders for experiments 4:")
			print("Training dataloaders:")
			print("\t({}) with half available data plus {} samples from each other labels,".format(self.label, self.size_all_tr_each))
			print("\t(0-9 except {}: 1), ..., (0-9 except {}: {})".format(self.label, self.label, self.n_rest_split))
			print("\tWhile the middle {} splits are added with the rest data from label {}.".format(self.n_relearning, self.label))
			# print("\t(0-9: 1), ..., (0-9: {})".format(self.label, self.label, self.n_relearning))
			print("Test dataloaders: (0), (1), ..., (9), (0-9), each with all the test data available,")
			print("\t(Approx. 6000 for each label and 10000 for all labels).")
			print("The actual size of each tr dataset:", self.size_subsets)


	def training_loaders(self, tr_dataset):
		if self.verbose:
			print("\n\n " +' Loading Training DataLoader for Experiments 4 '.center(70, '*'))

		n_split = self.n_split

		# First have the subset of all the ones with specified label:
		single_indices = [i for i in range(len(tr_dataset)) if tr_dataset[i][1] == self.label]
		# if self.shuffle:
		# 	random.shuffle(single_indices)
		single_dataset_tr = Subset(tr_dataset, single_indices)

		single_former_size = len(single_indices) // 2  # nbr of samples with specified label included in the first subset
		single_indices_former = single_indices[:single_former_size]
		single_indices_latter = single_indices[single_former_size:]
		single_latter_size = len(single_indices_latter) // self.n_relearning  # nbr of samples with specified label included in the last n_relearning subsets

		# Decide which splits are chosen as the relearning ones:
		middle_split = n_split // 2
		if self.n_relearning > n_split - 2:
			raise ValueError("The number of splits for relearning is not valid. It should be less than the number of splits - 2 for this experiment set.")
		if self.n_relearning % 2 != 0:
			index_start_relearning = int(middle_split - (self.n_relearning // 2))
			# index_end_relearning = int(middle_split + (self.n_relearning // 2))
		else:
			index_start_relearning = int(middle_split - (self.n_relearning / 2 - 1))
			# index_end_relearning = int(middle_split + (self.n_relearning / 2))
		# print(middle_split)
		# print(index_start_relearning)

		# Trial 1:
		digit_subsets = {}  # (digit, dataset.Subset)
		for digit in range(10):
			digit_indices = [i for i in range(len(tr_dataset)) if tr_dataset[i][1] == digit]
			if digit == self.label:
				digit_subset_indices = single_indices_former
			else:
				digit_subset_indices = digit_indices[:self.size_all_tr_each]
			digit_subsets[digit] = Subset(tr_dataset, digit_subset_indices)
		all_prior_dataset_tr = ConcatDataset([digit_subsets[digit] for digit in range(10)])

		# Trial 2:
		remaining_indices = set(range(len(tr_dataset)))
		for digit in range(10):
			remaining_indices -= set(digit_subsets[digit].indices)
		remaining_indices -= set(single_indices_latter)
		remaining_indices = list(remaining_indices)

		if self.shuffle:
			random.shuffle(remaining_indices)

		subset_datasets = [all_prior_dataset_tr]

		n_rest_split = n_split - 1
		split_size = len(remaining_indices) // n_rest_split
		# if len(tr_dataset) % split_size != 0:
		# 	n_splits += 1
		# if self.n_relearning > n_rest_split:
		# 	raise ValueError("The number of splits assigned for relearning of the specified label exceeds the total number of rest splits.")
		# index_start_relearning = n_split - self.n_relearning
		
		for i in range(n_rest_split):
			start = i * split_size
			end = (i + 1) * split_size if i < n_rest_split - 1 else len(remaining_indices)
			subsets_indices = remaining_indices[start:end]
			subset_datasets.append(Subset(tr_dataset, subsets_indices))

		# print(index_start_relearning)
		# print(n_splits)
		# print(single_latter_size)

		# Trial 3 & 4:
		for i in range(self.n_relearning):
			start = i * single_latter_size
			end = (i + 1) * single_latter_size if i < self.n_relearning - 1 else len(single_indices_latter)
			relearning_indices = single_indices_latter[start:end]
			# print(start, end)
			subset_datasets[index_start_relearning + i] = ConcatDataset([subset_datasets[index_start_relearning + i], Subset(tr_dataset, relearning_indices)])

		subset_loaders = []
		for subset_dataset in subset_datasets:
			subset_loaders.append(
					get_data_loader(
					subset_dataset, 
					batch_size=self.batch_size_tr, 
					cuda=self.cuda, 
					drop_last=self.drop_last, 
					shuffle=self.shuffle,
					)
				)
		if self.verbose:
			print(' Loading training DataLoaders successful '.center(70, '*'))

		# labels exist among the tr-datasets:
		labels_tr = [list(range(10))]
		labels_tr_splits = list(range(10))
		labels_tr_splits.remove(self.label)
		labels_tr += [labels_tr_splits] * (n_rest_split - self.n_relearning)
		labels_tr += [list(range(10))] * self.n_relearning

		# size of each tr dataset:
		size_subsets = [len(subset_dataset) for subset_dataset in subset_datasets]

		return subset_loaders, n_rest_split, labels_tr, size_subsets
