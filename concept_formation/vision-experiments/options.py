"""
Define options.
"""

def general_options(parser):
	parser.add_argument('--experiment', type=int, default=0, choices=[0, 1, 2, 3, 4, 5], help='type of experiments')
	# parser.add_argument('--label', type=int, default=0, help='the label to be separated from in the experiments')
	parser.add_argument('--label', type=lambda x: int(x) if x.isdigit() else x, default=0, help='the label to be separated from in the experiments')
	parser.add_argument('--no-cuda', action='store_true', help='do NOT use cuda (even if the condition holds)')
	return parser


def data_options(parser):
	parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar'], help='Initial dataset chosen')
	parser.add_argument('--no-shuffle', action='store_true', help='do NOT shuffle the data when generating a dataset')
	parser.add_argument('--normalize', action='store_true', help='normalize every data')
	parser.add_argument('--seed', type=int, default=123, help='random seed')
	parser.add_argument('--n_split', type=int, default=10, help='the number of training splits in an experiment')
	parser.add_argument('--split-size', type=int, default=6000, help='# of training data in every split of remained labels training')
	parser.add_argument('--batch-size-tr', type=int, default=64, help='batch size of every training data loader')
	parser.add_argument('--batch-size-te', type=int, default=64, help='batch size of every test data loader')
	parser.add_argument('--pad', action='store_true', help='whether pad each image by 2 pixels')
	parser.add_argument('--drop-last', action='store_true', help='whether drop the last batch if its size is less than batch size')
	parser.add_argument('--permutation', action='store_true', help='whether permute the pixels of all images')
	parser.add_argument('--size-all-tr-each', type=int, default=600, 
		help='the size of training data from each label that forms the concatenated training set (which has the size n_label * size_all_tr_each)')
	parser.add_argument('--n-relearning', type=int, default=5, help='number of splits that include relearning')
	return parser


def model_options(parser):
	parser.add_argument('--model-type', type=str, default='cobweb', choices=['cobweb', 'fc', 'fc-cnn'], help='type of model')
	parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
	parser.add_argument('--epoch', type=int, default=1, help='# of epochs of training')
	parser.add_argument('--log-interval', type=int, default=10, help='size of log interval')
	parser.add_argument('--momentum', type=float, default=0.5, help='the value of momentum parameter in optimizer SGD')
	parser.add_argument('--kernel', type=int, default=5, help='kernel size of the CNN model layers')
	parser.add_argument('--n-hidden', type=int, default=1, help='# of hidden layers in fc model')
	parser.add_argument('--n-nodes', type=int, default=100, help='# of nodes in each hidden layer in fc model')
	return parser



