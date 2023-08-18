import torch
from torch import optim, nn
import torch.nn.functional as F
from tqdm import tqdm


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


class FC(nn.Module):
	def __init__(self, n_hidden, n_nodes, available_labels=10, image_size=28):
		super(FC, self).__init__()
		# self.fc1 = nn.Linear(image_size * image_size, 100)
		# self.fc2 = nn.Linear(100, available_labels)
		self.n_hidden = n_hidden
		self.hidden_layers = nn.ModuleList([nn.Linear(image_size * image_size, n_nodes)])
		for _ in range(self.n_hidden):
			self.hidden_layers.append(nn.Linear(n_nodes, n_nodes))
		self.fc_out = nn.Linear(n_nodes, available_labels)

	def forward(self, x):
		x = x.view(x.size(0), -1)
		# x = self.fc1(x)
		# x = F.relu(self.fc1(x))
		# # x = F.dropout(x, training=self.training)
		# x = self.fc2(x)

		for layer in self.hidden_layers:
			x = F.relu(layer(x))
		x = self.fc_out(x)
		return F.log_softmax(x)


def build_model(model_config, data_config, device):

	available_labels = data_config['available_labels']
	image_size = data_config['image_size']
	kernel_size = model_config['kernel']
	lr = model_config['lr']
	momentum = model_config['momentum']
	n_hidden = model_config['n_hidden']
	n_nodes = model_config['n_nodes']

	if model_config['type'] == 'fc':
		model = FC(n_hidden=n_hidden, n_nodes=n_nodes,
		 available_labels=available_labels, image_size=image_size).to(device)
	else:
		model = FC_CNN(n_hidden=n_hidden, n_nodes=n_nodes,
		 available_labels=available_labels, kernel_size=kernel_size).to(device)
	# if cuda:
	# 	model = model.cuda()
	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
	return model, optimizer


def train(model, optimizer, train_loader, epoch, log_interval, device):
	
	# batch_size = data_config['batch_size_tr']
	# log_interval = model_config['log_interval']

	model.train()
	running_loss = 0.0
	# train_losses = []
	# train_counter = []
	# train_loader = tqdm(train_loader)  # include a progress bar
	with tqdm(train_loader, unit='batch') as tepoch:  # include a progress bar
		tepoch.set_description(f"Epoch {epoch}")
		for batch_id, (imgs, labels) in enumerate(train_loader):
			optimizer.zero_grad()
			outputs = model(imgs)
			loss = F.nll_loss(outputs, labels)
			loss.backward()
			optimizer.step()
			if batch_id % log_interval == 0:
				print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
					epoch, 
					batch_id * len(imgs), 
					len(train_loader.dataset), 
					100. * batch_id / len(train_loader), 
					loss.item()))
				# train_losses.append(loss.item())
				# train_counter.append((batch_id * batch_size) + ((epoch - 1) * len*(train_loader.dataset)))
				# torch.save(model.state_dict(), '/results/model.pth')
				# torch.save(optimizer.state_dict(), '/results/optimizer/pth')
			running_loss += loss.item()
			tepoch.set_postfix(loss=running_loss / len(tepoch))


def test(model, test_loader, device):
	# test_loader = tqdm(test_loader)  # include a progress bar
	model.eval()
	# test_losses = []
	test_loss = 0
	correct = 0
	accuracy = 0.
	with tqdm(test_loader, unit='batch') as tepoch:  # include a progress bar
		tepoch.set_description("Testing")
		with torch.no_grad():
			for imgs, labels in test_loader:
				imgs, labels = imgs.to(device), labels.to(device)
				outputs = model(imgs)
				test_loss += F.nll_loss(outputs, labels, size_average=False).item()
				preds = outputs.data.max(1, keepdim=True)[1]
				correct += preds.eq(labels.data.view_as(preds)).sum()
		test_loss /= len(test_loader.dataset)
		# test_losses.append(test_loss)
		accuracy = correct / len(test_loader.dataset)
		print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
			test_loss, correct, len(test_loader.dataset), 100. * accuracy))
		tepoch.set_postfix(accuracy=accuracy.item())
	return accuracy

