import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import torch.nn.functional as F
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from tqdm import tqdm
import random

from models_nn import FC
from datasets_mnist import MNIST_dataset, get_data_loader


def train_eval(train_loader, val_loader, n_hidden, n_nodes, lr, n_epoch, batch_size, momentum):

	model = FC(n_hidden=n_hidden, n_nodes=n_nodes, available_labels=10, image_size=28)
	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

	for epoch in range(n_epoch):
		model.train()
		with tqdm(train_loader, unit='batch') as tepoch:  # include a progress bar
			tepoch.set_description(f"Epoch {epoch}")
			for batch_id, (imgs, labels) in enumerate(train_loader):
				optimizer.zero_grad()
				outputs = model(imgs)
				loss = F.nll_loss(outputs, labels)
				loss.backward()
				optimizer.step()

	model.eval()
	predictions = []
	true_labels = []
	with torch.no_grad():
		for data, target in val_loader:
			output = model(data)
			_, predicted = torch.max(output.data, 1)
			predictions.extend(predicted)
			true_labels.extend(target)

	accuracy = accuracy_score(true_labels, predictions)
	return accuracy


# Training set and validation set (part of the original training set)
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

dataset_ori = MNIST_dataset(split='train', pad=False, normalize=False, permutation=False, download=True, verbose=True)

indices = list(range(len(dataset_ori)))
random.shuffle(indices)
point = int(len(indices) * 0.8)
dataset_tr = Subset(dataset_ori, indices[:point])
dataset_val = Subset(dataset_ori, indices[point:])



# X_tr = dataset_tr.data.view(-1, 28 * 28).float()
# X_tr = dataset_tr.data.float()
# # for x in X_tr:
# # 	x = x.unsqueeze(0)
# print(X_tr[0].shape)
# print(dataset_tr[0][0].shape)
# y_tr = dataset_tr.targets

# X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=0)
# dataset_tr = data.TensorDataset(X_tr, y_tr)
# dataset_val = data.TensorDataset(X_val, y_val)
# for i in range(len(dataset_tr)):
# 	data, label = dataset_tr[i]
# 	data = data.unsqueeze(0)  # Add a channel dimension to data
# 	label = torch.tensor(label)  # Convert label to tensor (optional)
# 	dataset_tr[i] = (data, label)  # Update the data and label
# for i in range(len(dataset_val)):
# 	data, label = dataset_val[i]
# 	data = data.unsqueeze(0)  # Add a channel dimension to data
# 	label = torch.tensor(label)  # Convert label to tensor (optional)
# 	dataset_val[i] = (data, label)  # Update the data and label

# print(dataset_tr[0][0].shape)


# Hyperparameter grid:
param_grid = {
	'n_hidden': [1, 2, 3],
	'n_nodes': [16, 32, 64, 128],
	'lr': list(np.logspace(np.log10(0.001), np.log10(0.5), num=20)),
	'epoch': [1, 2, 3, 4, 5],
	'batch_size': [32, 64, 128, 256],
	'momentum': [0.0, 0.5, 0.9, 0.99],
}

# Perform grid search for fc:
best_accuracy = 0.0
best_params = {}

for batch_size in param_grid['batch_size']:
	# Data loaders:
	train_loader = data.DataLoader(dataset_tr, batch_size=batch_size, shuffle=True, drop_last=False)
	val_loader = data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=False)

	for n_hidden in param_grid['n_hidden']:
		for n_nodes in param_grid['n_nodes']:
			for lr in param_grid['lr']:
				for n_epoch in param_grid['epoch']:
					for momentum in param_grid['momentum']:
						accuracy = train_eval(train_loader, val_loader, n_hidden, n_nodes, lr, n_epoch, batch_size, momentum)
						if accuracy > best_accuracy:
							best_accuracy = accuracy
							best_params = {
							'n_hidden': n_hidden,
							'n_nodes': n_nodes,
							'lr': lr,
							'epoch': n_epoch,
							'batch_size': batch_size,
							'momentum': momentum,
							}

print("Best hyperparameters for FC:", best_params)
print("Best Accuracy:", best_accuracy)

# Perform grid search for fc-cnn:
# best_accuracy = 0.0
# best_params = {}

# for batch_size in param_grid['batch_size']:
# 	# Data loaders:
# 	train_loader = data.DataLoader(dataset_tr, batch_size=batch_size, shuffle=True, drop_last=False)
# 	val_loader = data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=False)

# 	for n_hidden in param_grid['n_hidden']:
# 		for n_nodes in param_grid['n_nodes']:
# 			for lr in param_grid['lr']:
# 				for n_epoch in param_grid['epoch']:
# 					for momentum in param_grid['momentum']:
# 						accuracy = train_eval(train_loader, val_loader, n_hidden, n_nodes, lr, n_epoch, batch_size, momentum, model='cnn')
# 						if accuracy > best_accuracy:
# 							best_accuracy = accuracy
# 							best_params = {
# 							'n_hidden': n_hidden,
# 							'n_nodes': n_nodes,
# 							'lr': lr,
# 							'epoch': n_epoch,
# 							'batch_size': batch_size,
# 							'momentum': momentum,
# 							}

# print("Best hyperparameters for FC-CNN:", best_params)
# print("Best Accuracy:", best_accuracy)




