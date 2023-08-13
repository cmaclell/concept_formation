import os
import pandas as pd
import numpy as np
import time
import torch
from torch import optim
import argparse
from tqdm import tqdm

from options import general_options, data_options, model_options
from datasets_mnist import MNIST_dataset, CIFAR_dataset
from experiment_0 import experiment_0_nn, experiment_0_cobweb
from experiment_1 import experiment_1_nn, experiment_1_cobweb
from experiment_2 import experiment_2_nn, experiment_2_cobweb
from experiment_3 import experiment_3_nn, experiment_3_cobweb
from experiment_4 import experiment_4_nn, experiment_4_cobweb
from experiment_5 import experiment_5_nn, experiment_5_cobweb


def checkattr(args, attr):
	'''Check whether attribute exists, whether it's a boolean and whether its value is True.'''
	return hasattr(args, attr) and type(getattr(args, attr))==bool and getattr(args, attr)



def accuracy2csv(general_config, model_config, data_config, test_accs):

	model = model_config['type']
	if model == 'cobweb':
		model = 'cobweb4v'
	seed = general_config['seed']
	label = general_config['label']
	experiment = general_config['type']
	n_split = data_config['n_split']

	trainset = []
	if experiment == 0:
		for i in range(1, n_split + 1):
			trainset += ["S%d" % i] * 11
	elif experiment == 1 or experiment == 2:
		for i in range(0, n_split):
			trainset += ["S%d" % i] * 11
	elif experiment == 3:
		n_relearning = data_config['n_relearning']
		for i in range(0, n_split - n_relearning):
			trainset += ["S%d" % i] * 11
		for i in range(1, n_relearning + 1):
			trainset += ["R%d" % i] * 11
	else:
		n_relearning = data_config['n_relearning']
		middle_split = n_split // 2
		if n_relearning % 2 != 0:
			index_start_relearning = int(middle_split - (n_relearning // 2))
		else:
			index_start_relearning = int(middle_split - (n_relearning / 2 - 1))
		for i in range(0, index_start_relearning):
			trainset += ["S%d" % i] * 11
		for i in range(1, n_relearning + 1):
			trainset += ["R%d" % i] * 11
		for i in range(n_split - n_relearning - index_start_relearning):
			trainset += ["S%d" % (i + index_start_relearning)] * 11


	testset = [f"L{i}" for i in range(10)]
	testset[label] = "L0"
	for i in range(len(testset)):
		if i != label:
			testset[i] = f"L{i + 1}"
		else:
			break
	testset.append("All")
	# print(testset)
	testset = testset * n_split

	# print(len(trainset))
	# print(len(testset))
	# print(len(test_accs))
	# print(len(trainset))
	# print(len(trainset))
	# print(len(trainset))

	data = {
	'TrainSet': trainset,
	'TestSet': testset,
	'Model': [model] * len(trainset),
	'Seed': [seed] * len(trainset),
	'Experiment': [experiment] * len(trainset),
	'TestAccuracy': test_accs,
	}
	if experiment == 5:
		data = {
		'TrainSet': trainset * 2,
		'TestSet': testset * 2,
		'Model': ([model] * len(trainset)) * 2,
		'Seed': ([seed] * len(trainset)) * 2,
		'Experiment': ([experiment] * len(trainset)) * 2,
		'TestAccuracy': test_accs,
		}
	df = pd.DataFrame(data)

	folder_name = 'test_accs'
	experiment_name = f'exp{experiment}'

	if not os.path.exists(folder_name):
		os.makedirs(folder_name)
	folder_path = os.path.join(folder_name, experiment_name)
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)

	# file_name = 'e' + str(experiment) + '_' + model + '_s' + str(seed) + '_l' + str(label) + "_accs.csv"
	file_name = f"L{label}_{model}_S{seed}.csv"
	file_path = os.path.join(folder_path, file_name)
	df.to_csv(file_path, index=False)


	# data = {'test_acc': test_accs}
	# df = pd.DataFrame(data)

	# model = model_config['type']
	# seed = general_config['seed']
	# label = general_config['label']
	# experiment = general_config['type']

	# folder_name = 'test_accs'
	# if not os.path.exists(folder_name):
	# 	os.makedirs(folder_name)

	# file_name = 'e' + str(experiment) + '_' + model + '_s' + str(seed) + '_l' + str(label) + "_accs.csv"
	# file_path = os.path.join(folder_name, file_name)
	# df.to_csv(file_path, index=False)



def experiments(args):

	verbose = True

	general_config = {
	'type': args.experiment,
	'label': args.label,
	'seed': args.seed,
	'cuda': False,
	}

	model_config = {
	'type': args.model_type,
	'lr': args.lr,
	'epoch': args.epoch,
	'log_interval': args.log_interval,
	'momentum': args.momentum,
	'kernel': args.kernel,
	'n_hidden': args.n_hidden,
	'n_nodes': args.n_nodes,
	}

	data_config = {
	# Basic info of the initial dataset:
	'dataset': args.dataset,
	'available_labels': 10,
	'image_size': 28,

	# Size of each dataset:
	'size_all_tr_each': args.size_all_tr_each,
	'n_split': args.n_split,
	'split_size': args.split_size,
	'n_relearning': args.n_relearning,

	# settings:
	'shuffle': True,
	'normalize': False,
	'batch_size_tr': args.batch_size_tr,
	'batch_size_te': args.batch_size_te,
	'drop_last': False,
	'pad': False,
	'permutation': False,
	}

	if checkattr(args, 'no_shuffle'):
		data_config['shuffle'] = False
	if checkattr(args, 'normalize'):
		data_config['normalize'] = True
	if checkattr(args, 'drop_last'):
		data_config['drop_last'] = True
	if checkattr(args, 'pad'):
		data_config['pad'] = True
	if checkattr(args, 'permutation'):
		data_config['permutation'] = True

	# Cuda:
	if torch.cuda.is_available():
		model_config['cuda'] = True
	if checkattr(args, 'no_cuda'):
		model_config['cuda'] = False
	device = torch.device("cuda" if general_config['cuda'] else "cpu")
	
	# Load initial datasets:
	if data_config['dataset'] == 'mnist':
		dataset_tr = MNIST_dataset(split='train', 
			pad=data_config['pad'], 
			normalize=data_config['normalize'], 
			permutation=data_config['permutation'], 
			download=True, verbose=verbose)
		dataset_te = MNIST_dataset(split='test', 
			pad=data_config['pad'], 
			normalize=data_config['normalize'], 
			permutation=data_config['permutation'], 
			download=True, verbose=verbose)
	else:
		dataset_tr = CIFAR_dataset(split='train', 
			pad=data_config['pad'], 
			normalize=data_config['normalize'], 
			permutation=data_config['permutation'], 
			download=True, verbose=verbose)
		dataset_te = CIFAR_dataset(split='test', 
			pad=data_config['pad'], 
			normalize=data_config['normalize'], 
			permutation=data_config['permutation'], 
			download=True, verbose=verbose)

	experiment = general_config['type']
	model_type = model_config['type']
	label = general_config['label']
	split_size = data_config['split_size']
	log_interval = model_config['log_interval']

	if experiment == 0:
		if model_type == 'cobweb':
			test_accs = experiment_0_cobweb(general_config, model_config, data_config, dataset_tr, dataset_te, verbose)
		else:
			test_accs = experiment_0_nn(general_config, model_config, data_config, dataset_tr, dataset_te, verbose)
	if experiment == 1:
		if model_type == 'cobweb':
			test_accs = experiment_1_cobweb(general_config, model_config, data_config, dataset_tr, dataset_te, verbose)
		else:
			test_accs = experiment_1_nn(general_config, model_config, data_config, dataset_tr, dataset_te, verbose)
	if experiment == 2:
		if model_type == 'cobweb':
			test_accs = experiment_2_cobweb(general_config, model_config, data_config, dataset_tr, dataset_te, verbose)
		else:
			test_accs = experiment_2_nn(general_config, model_config, data_config, dataset_tr, dataset_te, verbose)
	if experiment == 3:
		if model_type == 'cobweb':
			test_accs = experiment_3_cobweb(general_config, model_config, data_config, dataset_tr, dataset_te, verbose)
		else:
			test_accs = experiment_3_nn(general_config, model_config, data_config, dataset_tr, dataset_te, verbose)
	if experiment == 4:
		if model_type == 'cobweb':
			test_accs = experiment_4_cobweb(general_config, model_config, data_config, dataset_tr, dataset_te, verbose)
		else:
			test_accs = experiment_4_nn(general_config, model_config, data_config, dataset_tr, dataset_te, verbose)
	if experiment == 5:
		if model_type == 'cobweb':
			test_accs = experiment_5_cobweb(general_config, model_config, data_config, dataset_tr, dataset_te, verbose)
		else:
			test_accs = experiment_5_nn(general_config, model_config, data_config, dataset_tr, dataset_te, verbose)
	accuracy2csv(general_config, model_config, data_config, test_accs)


if __name__ == '__main__':

	parser = argparse.ArgumentParser("./main_mnist.py",
									 description="Run sets of experiments based on the MNIST dataset.")
	parser = general_options(parser)
	parser = data_options(parser)
	parser = model_options(parser)
	args = parser.parse_args()

	experiments(args)

