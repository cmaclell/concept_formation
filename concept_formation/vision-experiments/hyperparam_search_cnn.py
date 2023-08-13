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


class FC_CNN(nn.Module):
	def __init__(self, n_hidden, n_nodes, available_labels=10, kernel_size=5):
		super(FC_CNN, self).__init__()

		# CNN Layers:
		self.conv1 = nn.Conv2d(1, 10, kernel_size=kernel_size)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=kernel_size)
		# self.conv2_dropout = nn.Dropout2d()

		# FC layers:
		self.n_hidden = n_hidden
		self.hidden_layers = nn.ModuleList([nn.Linear(320, n_nodes)])
		if self.n_hidden > 1:
			for _ in range(self.n_hidden - 1):
				self.hidden_layers.append(nn.Linear(n_nodes, n_nodes))
		self.fc_out = nn.Linear(n_nodes, available_labels)

		# self.fc1 = nn.Linear(320, 50)
		# self.fc2 = nn.Linear(50, available_labels)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2(x), 2))
		x = x.view(-1, 320)
		for layer in self.hidden_layers:
			x = F.relu(layer(x))
		# x = F.relu(self.fc1(x))
		# x = F.dropout(x, training=self.training)
		# x = self.fc2(x)
		x = self.fc_out(x)
		return F.log_softmax(x)



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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_eval(train_loader, val_loader, n_hidden, n_nodes, lr, n_epoch, batch_size, momentum):

	model = FC_CNN(n_hidden=n_hidden, n_nodes=n_nodes, available_labels=10, kernel_size=5).to(device)
	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

	for epoch in range(n_epoch):
		model.train()
		with tqdm(train_loader, unit='batch') as tepoch:  # include a progress bar
			tepoch.set_description(f"Epoch {epoch}")
			for batch_id, (imgs, labels) in enumerate(train_loader):
				optimizer.zero_grad()
				outputs = model(imgs.to(device))
				loss = F.nll_loss(outputs, labels.to(device))
				loss.backward()
				optimizer.step()

	model.eval()
	predictions = []
	true_labels = []
	with torch.no_grad():
		for data, target in val_loader:
			output = model(data.to(device))
			_, predicted = torch.max(output.data, 1)
			predictions.extend(predicted.cpu())
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

# Hyperparameter grid:
param_grid = {
	'n_hidden': [1, 2, 3],
	'n_nodes': [16, 32, 64, 128],
	'lr': list(np.logspace(np.log10(0.001), np.log10(0.5), num=20)),
	'epoch': [1, 2, 3, 4, 5],
	'batch_size': [32, 64, 128, 256],
	'momentum': [0.0, 0.5, 0.9, 0.99],
}

# Perform grid search for fc-cnn:
best_accuracy = 0.0
best_params = {}

for batch_size in param_grid['batch_size']:
	# Data loaders:
	train_loader = data.DataLoader(dataset_tr, batch_size=batch_size, shuffle=True, drop_last=False)
	val_loader = data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=False)
	# for inputs, labels in train_loader:
	# 	inputs = inputs.to(device)
	# 	labels = labels.to(device)
	# for inputs, labels in val_loader:
	# 	inputs = inputs.to(device)
	# 	labels = labels.to(device)

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

print("Best hyperparameters for FC-CNN:", best_params)
print("Best Accuracy:", best_accuracy)


