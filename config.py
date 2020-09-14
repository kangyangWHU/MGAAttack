import torch
import numpy as np

# add your model path
model_pth = {

    'vgg16_bn_cifar10': 'models/cifar10/state_dicts/vgg16_bn_cifar10.pt',
    'vgg19_bn_cifar10': 'models/cifar10/state_dicts/vgg19_bn_cifar10.pt',
    'resnet18_cifar10': 'models/cifar10/state_dicts/resnet18_cifar10.pt',
    'googlenet_cifar10': 'models/cifar10/state_dicts/googlenet_cifar10.pt',


}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# your dataset path
local_cifar10_path = '/home/yangkang/project/datasets/'
# imagenet12_valk = 'D:/datasets/ImageNet12/val1k/'


cifar10_mean = np.array([0.4914, 0.4822, 0.4465])
cifar10_std = np.array([0.2023, 0.1994, 0.2010])
# this is compute by (0-mean)/std
cifar10_min = np.array([-2.42906574, -2.41825476, -2.22139303])
cifar10_max = np.array([2.51408799, 2.59679037, 2.75373134])

torch_mean = np.array([0.485, 0.456, 0.406])
torch_std = np.array([0.229, 0.224, 0.225])

tf_mean = np.array([0.5,0.5,0.5])
tf_std = np.array([0.5,0.5,0.5])

cifar10_labels = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
