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
	Generate test DataLoader(s) for all experiment cases.
	"""
	label = general_config['label']
	cuda = general_config['cuda']
	seed = general_config['seed']
	test = general_config['test']

	drop_last = data_config['drop_last']
	shuffle = data_config['shuffle']
	batch_size_te = data_config['batch_size_te']

	# Set random seeds:
	np.random.seed(seed)
	torch.manual_seed(seed)
	random.seed(seed)

	if verbose:
		print("\n\n " +' Loading test DataLoaders '.center(70, '*'))

	test_datasets = []

	if test == 'entire':
		# load the entire MNIST test set only.
		test_loaders = [get_data_loader(
			te_dataset,
			batch_size=batch_size_te,
			cuda=cuda,
			drop_last=drop_last,
			shuffle=shuffle)]
		labels_te = [tuple(range(10))]

	elif test == 'chosen':
		# Load the test set comprised of chosen digit only.
		chosen_digit_indices = [i for i in range(len(te_dataset)) if te_dataset[i][1] == label]
		test_dataset = Subset(te_dataset, chosen_digit_indices)
		test_loaders = [get_data_loader(
			test_dataset,
			batch_size=batch_size_te,
			cuda=cuda,
			drop_last=drop_last,
			shuffle=shuffle)]
		labels_te = [label]

	elif test == 'all':
		# Load all test sets (11), entire MNIST test set, chosen-digit test set, and non-chosen-digit test sets (9).
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

		labels_te = []
		for i in range(10):
			labels_te.append(i)
		labels_te.append(tuple(range(10)))

	else:
		# Load the non-chosen-digit test sets only.
		for d in range(10):
			if d != label:
				single_digit_indices = [i for i in range(len(te_dataset)) if te_dataset[i][1] == d]
				test_datasets.append(Subset(te_dataset, single_digit_indices))
		test_loaders = [get_data_loader(
			test_dataset,
			batch_size=batch_size_te,
			cuda=cuda,
			drop_last=drop_last,
			shuffle=shuffle) for test_dataset in test_datasets]
		labels_te = []
		for i in range(10):
			if i != label:
				labels_te.append(i)

	if len(labels_te) != len(test_loaders):
		raise ValueError("len(labels_te) != len(test_loaders)")

	if verbose:
		print(' Loading test DataLoaders successful '.center(70, '*'))
	return test_loaders, labels_te




class dataloaders_0(object):
	"""
	Load train and test data loaders for Experiment 0.
	Train splits: All trained data are shuffled and trained in sequential splits.
	"""

	def __init__(self, general_config, data_config, dataset_tr, dataset_te, verbose):

		self.cuda = general_config['cuda']
		self.seed = general_config['seed']

		self.shuffle = data_config['shuffle']
		self.batch_size_tr = data_config['batch_size_tr']
		self.batch_size_te = data_config['batch_size_te']
		self.normalize = data_config['normalize']
		self.pad = data_config['pad']
		self.permutation = data_config['permutation']
		self.drop_last = data_config['drop_last']
		self.n_split = data_config['n_split']

		# Set random seeds:
		np.random.seed(self.seed)
		torch.manual_seed(self.seed)
		random.seed(self.seed)

		self.verbose = verbose
		self.training_loaders, self.labels_tr, self.size_subsets = self.training_loaders(dataset_tr)
		self.test_loaders, self.labels_te = test_loaders(dataset_te, general_config, data_config, verbose=verbose)
		if verbose:
			print("\nAn overview of the dataloaders for Experiment 0:")
			print("Number of training splits:", self.n_split)
			print("Test set(s) after every splits:", general_config['test'])
			print("Actual size of each training split:", self.size_subsets)


	def training_loaders(self, tr_dataset):

		if self.verbose:
			print("\n\n " +' Loading Training DataLoader for Experiments 0 '.center(70, '*'))

		n_split = self.n_split

		# Shuffle the entire original dataset before making splits
		tr_dataset_indices = list(range(len(tr_dataset)))
		if self.shuffle:
			random.shuffle(tr_dataset_indices)

		subset_datasets = []  # the collection of data splits
		split_size = len(tr_dataset) // n_split  # the (least) size of each training split
		# Make splits:
		for i in range(n_split):
			start = i * split_size
			end = (i + 1) * split_size if i < n_split - 1 else len(tr_dataset_indices)
			subsets_indices = tr_dataset_indices[start:end]
			subset_datasets.append(Subset(tr_dataset, subsets_indices))

		# Turn each split from <DataSet> to <DataLoader>
		subset_loaders = [get_data_loader(
			subset_dataset, 
			batch_size=self.batch_size_tr, 
			cuda=self.cuda, 
			drop_last=self.drop_last, 
			shuffle=self.shuffle) for subset_dataset in subset_datasets]

		if self.verbose:
			print(' Loading training DataLoaders successful '.center(70, '*'))

		# Digit labels that included in each split:
		labels_tr = [list(range(10))] * n_split

		# Size of each training split:
		size_subsets = [len(subset_dataset) for subset_dataset in subset_datasets]

		return subset_loaders, labels_tr, size_subsets




class dataloaders_1(object):
	"""
	Load train and test data for Experiment 1.
	"""

	def __init__(self, general_config, data_config, dataset_tr, dataset_te, verbose):

		self.cuda = general_config['cuda']
		self.seed = general_config['seed']
		self.label = general_config['label']

		self.shuffle = data_config['shuffle']
		self.batch_size_tr = data_config['batch_size_tr']
		self.batch_size_te = data_config['batch_size_te']
		self.n_split = data_config['n_split']
		if self.n_split > len(dataset_tr):
			raise ValueError("# of training splits exceeds the size of the initial training dataset.")
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
			print("\nAn overview of the dataloaders for Experiment 1:")
			print("Number of training splits:", self.n_split)
			print("Test set(s) after every splits:", general_config['test'])
			print("Chosen digit:", self.label)
			print("# of samples from each non-chosen digits in the 1st training split:", self.size_all_tr_each)
			print("Actual size of each training split:", self.size_subsets)

	def training_loaders(self, tr_dataset):
		if self.verbose:
			print("\n\n " +' Loading Training DataLoader for Experiments 1 '.center(70, '*'))

		n_split = self.n_split

		# ==========
		# First generate the first split.
		# Comprised of data from all chosen label and the same number of sampples from each non-chosen label.

		digit_subsets = {}  # (digit, dataset.Subset)
		for digit in range(10):
			digit_indices = [i for i in range(len(tr_dataset)) if tr_dataset[i][1] == digit]
			if digit == self.label:
				digit_subset_indices = digit_indices
			else:
				digit_subset_indices = digit_indices[:self.size_all_tr_each]
			digit_subsets[digit] = Subset(tr_dataset, digit_subset_indices)
		all_prior_dataset_tr = ConcatDataset([digit_subsets[digit] for digit in range(10)])

		# ==========
		# Then divide the rest randomly.

		remaining_indices = set(range(len(tr_dataset)))
		for digit in range(10):
			remaining_indices -= set(digit_subsets[digit].indices)
		remaining_indices = list(remaining_indices)

		if self.shuffle:
			random.shuffle(remaining_indices)

		subset_datasets = [all_prior_dataset_tr]
		n_rest_split = n_split - 1
		split_size = len(remaining_indices) // n_rest_split
		for i in range(n_rest_split):
			start = i * split_size
			end = (i + 1) * split_size if i < n_rest_split - 1 else len(remaining_indices)
			subsets_indices = remaining_indices[start:end]
			subset_datasets.append(Subset(tr_dataset, subsets_indices))

		# Turn each split from <DataSet> to <DataLoader>
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

		# Digit labels that included in each split:
		labels_tr = [list(range(10))]
		labels_tr_splits = list(range(10))
		labels_tr_splits.remove(self.label)
		labels_tr += [labels_tr_splits] * n_rest_split

		# Size of each training split:
		size_subsets = [len(subset_dataset) for subset_dataset in subset_datasets]

		return subset_loaders, n_rest_split, labels_tr, size_subsets




class dataloaders_2(object):
	"""
	Load train and test data for Experiment 1.
	"""
	def __init__(self, general_config, data_config, dataset_tr, dataset_te, verbose):

		self.cuda = general_config['cuda']
		self.seed = general_config['seed']
		self.label = general_config['label']

		self.shuffle = data_config['shuffle']
		self.batch_size_tr = data_config['batch_size_tr']
		self.batch_size_te = data_config['batch_size_te']
		self.n_split = data_config['n_split']
		if self.n_split > len(dataset_tr):
			raise ValueError("# of training splits exceeds the size of the initial training dataset.")
		self.normalize = data_config['normalize']
		self.pad = data_config['pad']
		self.permutation = data_config['permutation']
		self.drop_last = data_config['drop_last']
		self.size_all_tr_each = data_config['size_all_tr_each']
		self.relearning_split = data_config['relearning_split']
		if max(self.relearning_split) > self.n_split:
			raise ValueError("The relearning split pointed exceeds the indices available.")

		# Set random seeds:
		np.random.seed(self.seed)
		torch.manual_seed(self.seed)
		random.seed(self.seed)

		self.verbose = verbose
		self.training_loaders, self.n_rest_split, self.labels_tr, self.size_subsets = self.training_loaders(dataset_tr)
		self.test_loaders, self.labels_te = test_loaders(dataset_te, general_config, data_config, verbose=verbose)
		if verbose:
			print("\nAn overview of the dataloaders for Experiment 2:")
			print("Number of training splits:", self.n_split)
			print("Test set(s) after every splits:", general_config['test'])
			print("Chosen digit:", self.label)
			print("# of samples from each non-chosen digits in the 1st training split:", self.size_all_tr_each)
			print("Relearning splits:", self.relearning_split)
			print("Actual size of each training split:", self.size_subsets)


	def training_loaders(self, tr_dataset):
		if self.verbose:
			print("\n\n " +' Loading Training DataLoader for Experiments 2 '.center(70, '*'))

		n_split = self.n_split
		if self.relearning_split is not None:
			n_relearning = len(self.relearning_split)
			relearning_split = self.relearning_split
		else:
			n_relearning = n_split // 2  # by default, one half
			relearning_split = [range(1, n_split + 1)][:-n_relearning]  # by default, the last half
		self.n_relearning = n_relearning

		# First have the subset of all the ones with specified label:
		single_indices = [i for i in range(len(tr_dataset)) if tr_dataset[i][1] == self.label]
		single_dataset_tr = Subset(tr_dataset, single_indices)

		single_former_size = len(single_indices) // 2  # nbr of samples with specified label included in the first subset
		single_indices_former = single_indices[:single_former_size]
		single_indices_latter = single_indices[single_former_size:]
		single_latter_size = len(single_indices_latter) // n_relearning  # nbr of samples with chosen digit label included in each relearning split

		# ==========
		# First generate the first split.
		# Comprised of data from all chosen label and the same number of sampples from each non-chosen label.

		digit_subsets = {}  # (digit, dataset.Subset)
		for digit in range(10):
			digit_indices = [i for i in range(len(tr_dataset)) if tr_dataset[i][1] == digit]
			if digit == self.label:
				digit_subset_indices = single_indices_former
			else:
				digit_subset_indices = digit_indices[:self.size_all_tr_each]
			digit_subsets[digit] = Subset(tr_dataset, digit_subset_indices)
		all_prior_dataset_tr = ConcatDataset([digit_subsets[digit] for digit in range(10)])

		# ==========
		# Then randonly divide the remaining data from non-chosen labels.

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
		
		for i in range(n_rest_split):
			start = i * split_size
			end = (i + 1) * split_size if i < n_rest_split - 1 else len(remaining_indices)
			subsets_indices = remaining_indices[start:end]
			subset_datasets.append(Subset(tr_dataset, subsets_indices))


		# ==========
		# Finally add data from the chosen label to each relearning split.

		for i in range(n_relearning):
			start = i * single_latter_size
			end = (i + 1) * single_latter_size if i < n_relearning - 1 else len(single_indices_latter)
			relearning_indices = single_indices_latter[start:end]
			# print(start, end)
			subset_datasets[relearning_split[i] - 1] = ConcatDataset(
				[subset_datasets[relearning_split[i] - 1], 
				Subset(tr_dataset, relearning_indices)])

		# Turn each split from <DataSet> to <DataLoader>
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

		# Digit labels that included in each split:
		labels_tr = [list(range(10))]
		labels_tr_splits = list(range(10))
		labels_tr_splits.remove(self.label)
		labels_tr += [labels_tr_splits] * n_rest_split
		for index in relearning_split:
			labels_tr[index - 1].append(self.label)

		# Size of each training split:
		size_subsets = [len(subset_dataset) for subset_dataset in subset_datasets]

		return subset_loaders, n_rest_split, labels_tr, size_subsets
