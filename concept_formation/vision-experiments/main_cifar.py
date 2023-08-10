import os
import pandas as pd
import numpy as np
import time
import torch
from torch import optim
import argparse
from tqdm import tqdm

from options import general_options, data_options, model_options
from datasets_cifar import CIFAR_dataset
from experiment_0 import experiment_0_nn, experiment_0_cobweb
from experiment_1 import experiment_1_nn, experiment_1_cobweb
from experiment_2 import experiment_2_nn, experiment_2_cobweb
from experiment_3 import experiment_3_nn, experiment_3_cobweb


def checkattr(args, attr):
	'''Check whether attribute exists, whether it's a boolean and whether its value is True.'''
	return hasattr(args, attr) and type(getattr(args, attr))==bool and getattr(args, attr)



def accuracy2csv(general_config, model_config, test_accs):
	data = {'test_acc': test_accs}
	df = pd.DataFrame(data)

	model = model_config['type']
	seed = general_config['seed']
	label = general_config['label']
	experiment = general_config['type']

	folder_name = 'test_accs'
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)

	file_name = 'cifar_e' + str(experiment) + '_' + model + '_s' + str(seed) + '_l' + str(label) + "_accs.csv"
	file_path = os.path.join(folder_name, file_name)
	df.to_csv(file_path, index=False)



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
	}

	data_config = {
	# Basic info of the initial dataset:
	'dataset': 'cifar',
	'available_labels': 10,
	'image_size': 32,

	# Size of each dataset:
	'size_all_tr_each': args.size_all_tr_each,
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
	accuracy2csv(general_config, model_config, test_accs)


if __name__ == '__main__':

	parser = argparse.ArgumentParser("./main_cifar.py",
									 description="Run sets of experiments based on the MNIST dataset.")
	parser = general_options(parser)
	parser = data_options(parser)
	parser = model_options(parser)
	args = parser.parse_args()

	experiments(args)

