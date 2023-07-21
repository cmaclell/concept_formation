import torch
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from concept_formation.cobweb_torch import CobwebTorchTree
from concept_formation.visualize import visualize

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == "__main__":
    transform = transforms.Compose(
      [transforms.ToTensor()])
    # [transforms.ToTensor(),
    #  transforms.Normalize((0.5), (0.5))])
    #[transforms.ToTensor(),
    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainSet = torchvision.datasets.MNIST(root='./data', train=True,
                                               download=True, transform=transform)
    # trainSet = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                            download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(trainSet,
                                          # batch_size=len(mnistTrainSet),
                                          batch_size=1000,
                                          shuffle=True,
                                          num_workers=3)

    train_images, train_labels = next(iter(data_loader))
    print(train_images.shape)

    ## Whole Images
    tree = CobwebTorchTree(train_images.shape[1:])
    for i in tqdm(range(train_images.shape[0])):
        tree.ifit(train_images[i])

    ## 8x8 Patches, MUCH SLOWER
    # tree = CobwebTorchTree((1,8,8))
    # for i in tqdm(range(train_images.shape[0])):
    #     for x in range(train_images.shape[2] - 8):
    #         for y in range(train_images.shape[3] - 8):
    #             tree.ifit(train_images[i, :, x:x+8, y:y+8])

    visualize(tree)
