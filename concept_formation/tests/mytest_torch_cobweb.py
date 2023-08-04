import torch
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from concept_formation.multinomial_cobweb import MultinomialCobwebTree
from concept_formation.cobweb_torch import CobwebTorchTree
from concept_formation.cobweb_torch import CobwebTorchNode
from concept_formation.visualize import visualize

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def one_hot_encode_label(label):
        one_hot_label = torch.zeros(10)
        one_hot_label[label] = 1.0
        return one_hot_label

def train_transform_with_one_hot_encoding(training_images, training_labels):
    for i in range(len(training_images)):
        label = training_labels[i]
        # Insert the one-hot encoding of the label at the beginning of the first row
        one_hot_label = one_hot_encode_label(label)
        training_images[i][0][0][:10] = one_hot_label
    return training_images

def test_transform(testing_images):
    for i in range(len(testing_images)):
        testing_images[i][0][0][:10] = 0.5
    return testing_images

if __name__ == "__main__":
    transform = transforms.Compose(
      [transforms.ToTensor()])
    #   [transforms.ToTensor(),
    #    transforms.Normalize((0.1307,), (0.3081,))])
    #[transforms.ToTensor(),
    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    ## MNIST
    train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                               download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False,
                                               download=True, transform=transform)
    
    print(len(train_set))
    print(len(test_set))
    ## CIFAR
    # trainSet = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                            download=True, transform=transform)


    #torch.manual_seed(0)
    train_data_loader = torch.utils.data.DataLoader(train_set,
                                          batch_size=len(train_set),
                                          shuffle=True,
                                          num_workers=3)
    test_data_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=len(test_set),
                                          shuffle=True,
                                          num_workers=3)
    
    
    train_images, train_labels = next(iter(train_data_loader))
    test_images, test_labels = next(iter(test_data_loader))

    # train_images = train_transform_with_one_hot_encoding(train_images,train_labels)
    # test_images = test_transform(test_images)

    # train_images = train_images[:10000]
    # test_images = test_images[:10000]
 
    # print(train_labels[0])
    # print(test_labels[0])

    ## Whole Images
    print(train_images.shape[0])
    tree = CobwebTorchTree(train_images.shape[1:])
    for i in tqdm(range(train_images.shape[0])):
        tree.ifit(train_images[i], train_labels[i].item())

    # ## Test on one image
    # print("actual_label:", test_labels[5])
    # o = tree.categorize(test_images[5])
    # out = o.predict()
    # out_np = out[0][0][:10].numpy()
    # L = np.argmax(out_np)
    # L_t = torch.tensor(L)
    # print("predicted_label:",L_t)

    ## Test on test set
    correct_prediction = 0
    for i in tqdm(range(test_images.shape[0])):
        actual_label = test_labels[i]
        o = tree.categorize(test_images[i])
        # out, out_label = o.get_best(test_images[i]).predict()
        # out, out_label = o.get_basic().predict()
        out, out_label = o.predict()
        # if(i % 100 == 0):
        #      print("actual:", actual_label,"\n")

        # get label
        # out_np = out[0][0][:10].numpy()
        # label = np.argmax(out_np)
        # predicted_label = torch.tensor(label)
        predicted_label = torch.tensor(out_label)

        if (predicted_label == actual_label):
             correct_prediction += 1

    Accuracy = (correct_prediction/len(test_images))*100
    print("Accuracy: ",Accuracy)

    # # 8x8 Patches, MUCH SLOWER
    # tree = CobwebTorchTree((1,8,8))
    # for i in tqdm(range(train_images.shape[0])):
    #     for x in range(train_images.shape[2] - 8):
    #         for y in range(train_images.shape[3] - 8):
    #             tree.ifit(train_images[i, :, x:x+8, y:y+8])

    # data = [{"{},{}".format(j,k): {str(round(train_images[i][0][j][k].item() * 4) / 4): 1} for j in range(train_images.shape[2]) for k in range(train_images.shape[3])} for i in range(train_images.shape[0])]
    # print(data[0])

    # tree = MultinomialCobwebTree()
    # futures = []
    # for instance in tqdm(data):
    #     futures.append(tree.ifit(instance))
    
    # for fut in tqdm(futures):
    #     fut.wait()

    visualize(tree)
