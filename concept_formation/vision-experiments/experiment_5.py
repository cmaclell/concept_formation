import torch

from concept_formation.cobweb_torch import CobwebTorchTree
from datasets_mnist import dataloaders_4
import models_nn
import models_cobweb


def reconfig(general_config, model_config, data_config):
	"""
	Re-initialization for model_config and data_config based on requirements of experiments 0.
	"""
	if data_config['dataset'] == 'mnist':
		data_config['split_size'] = 5400
		data_config['size_all_tr_each'] = 600
	else:
		data_config['split_size'] = 4500
		data_config['size_all_tr_each'] = 500
	if model_config['type'] == 'cobweb':
		general_config['cuda'] == False
		data_config['batch_size_tr'] = 60000
		data_config['batch_size_te'] = 10000


def experiment_5_nn(general_config, model_config, data_config, dataset_tr, dataset_te, verbose):

	reconfig(general_config, model_config, data_config)
	label = general_config['label']
	size_all_tr_each = data_config['size_all_tr_each']

	# Dataloaders:
	dataloaders = dataloaders_4(general_config, data_config, dataset_tr, dataset_te, verbose=verbose)
	loaders_tr = dataloaders.training_loaders
	loaders_te = dataloaders.test_loaders

	# Models and optimizers:
	device = torch.device("cuda" if general_config['cuda'] else "cpu")
	model, optimizer = models_nn.build_model(model_config, data_config, device)

	# Store the test accuracies
	test_accs = []

	if verbose:
		print('\n\n' + ' START EXPERIMENTS '.center(70, '~'))
		print("Experiments type: 5")
		print("Experiments description: Train half data from label {} and some data ({}) from each remaining label first,"
			" then train the rest with sequential splits.".format(label, size_all_tr_each))
		print("\tIn the middle {} splits, fit in the remaining training data from label {} evenly. (ReLearning)".format(dataloaders.n_relearning, label))
		print("\tAfter all the process, repeated them again.")
		print("Number of Train-test trials:", len(loaders_tr))
		print("Model:", model_config['type'])  # fc or cnn
		print("Seed:", general_config['seed'])
		print("The selected label:", label)
		print("Epochs:", model_config['epoch'])
		print("\nModel overview:")
		print(model)
		print("\nOptimizer:")
		print(optimizer)
		print("\nCUDA is {}used.".format("" if general_config['cuda'] else "NOT "))

	for i in range(len(loaders_tr)):
		if verbose:
			print("\n\n" + " Trial {} ".format(i+1).center(70, '='))
		
		for epoch in range(1, model_config['epoch'] + 1):
			if verbose:
				print("\n\n [Epoch {}]".format(epoch))
				print("\n====> Model Training with labels {} <====".format(dataloaders.labels_tr[i]))
			models_nn.train(model, optimizer, loaders_tr[i], epoch, model_config['log_interval'], device)
			
			for j in range(len(loaders_te)):
				if verbose:
					print("\n----> Model Testing with labels {} <----".format(dataloaders.labels_te[j]))
				acc = models_nn.test(model, loaders_te[j], device)
				test_accs.append(acc.item())

	# Repeat the process:
	for i in range(len(loaders_tr)):
		if verbose:
			print("\n\n" + " Trial {} ".format(i+len(loaders_tr)+1).center(70, '='))
		
		for epoch in range(1, model_config['epoch'] + 1):
			if verbose:
				print("\n\n [Epoch {}]".format(epoch))
				print("\n====> Model Training with labels {} <====".format(dataloaders.labels_tr[i]))
			models_nn.train(model, optimizer, loaders_tr[i], epoch, model_config['log_interval'], device)
			
			for j in range(len(loaders_te)):
				if verbose:
					print("\n----> Model Testing with labels {} <----".format(dataloaders.labels_te[j]))
				acc = models_nn.test(model, loaders_te[j], device)
				test_accs.append(acc.item())

	print("\n\nThis is the end of the experiments.")
	print("There are {} test accuracy data in total.".format(len(test_accs)))
	return test_accs


def experiment_5_cobweb(general_config, model_config, data_config, dataset_tr, dataset_te, verbose):

	reconfig(general_config, model_config, data_config)
	label = general_config['label']
	size_all_tr_each = data_config['size_all_tr_each']

	# Dataloaders:
	dataloaders = dataloaders_4(general_config, data_config, dataset_tr, dataset_te, verbose=verbose)
	loaders_tr = dataloaders.training_loaders
	loaders_te = dataloaders.test_loaders

	# Models and optimizers:
	example_imgs, _ = next(iter(loaders_tr[0]))
	model = CobwebTorchTree(example_imgs.shape[1:])

	# Store the test accuracies
	test_accs = []

	if verbose:
		print('\n\n' + ' START EXPERIMENTS '.center(70, '~'))
		print("Experiments type: 5")
		print("Experiments description: Train half data from label {} and some data ({}) from each remaining label first,"
			" then train the rest with sequential splits.".format(label, size_all_tr_each))
		print("\tIn the middle {} splits, fit in the remaining training data from label {} evenly. (ReLearning)".format(dataloaders.n_relearning, label))
		print("\tAfter all the process, repeated them again.")
		print("Number of Train-test trials:", len(loaders_tr))
		print("Model:", model_config['type'])  # cobweb
		print("Seed:", general_config['seed'])
		print("The selected label:", label)
		print("\nCUDA is {}used.".format("" if general_config['cuda'] else "NOT "))

	for i in range(len(loaders_tr)):
		if verbose:
			print("\n\n" + " Trial {} ".format(i+1).center(70, '='))
			print("\n====> Model Training with labels {} <====".format(dataloaders.labels_tr[i]))
		models_cobweb.train(model, loaders_tr[i])
		
		for j in range(len(loaders_te)):
			if verbose:
				print("\n----> Model Testing with labels {} <----".format(dataloaders.labels_te[j]))
			acc = models_cobweb.test(model, loaders_te[j])
			print("Test accuracy: {}".format(acc))
			test_accs.append(acc)

	# Repeat the process:
	for i in range(len(loaders_tr)):
		if verbose:
			print("\n\n" + " Trial {} ".format(i+len(loaders_tr)+1).center(70, '='))
			print("\n====> Model Training with labels {} <====".format(dataloaders.labels_tr[i]))
		models_cobweb.train(model, loaders_tr[i])
		
		for j in range(len(loaders_te)):
			if verbose:
				print("\n----> Model Testing with labels {} <----".format(dataloaders.labels_te[j]))
			acc = models_cobweb.test(model, loaders_te[j])
			print("Test accuracy: {}".format(acc))
			test_accs.append(acc)

	print("\n\nThis is the end of the experiments.")
	print("There are {} test accuracy data in total.".format(len(test_accs)))
	return test_accs
