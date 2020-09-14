__all__=['Cifar10Loader', 'get_imagenet_val_loader', "predict", "init_model",
         "batch_norm", "normalize", "clamp_by_2norm", "torch2numpy",
         "RP", "JPEG", "BitDepthReduce", "InputTransformModel"]
import torch
import torch.nn as nn
import kornia
from torchvision import datasets
import config as flags
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import PIL
import PIL.Image
from io import BytesIO
from models.cifar10 import *
from pretrainedmodels import *

# loader utils
class Cifar10Loader(object):

    def __init__(self, batch_size, train_transforms=None, val_test_transforms=None):

        self.batch_size = batch_size
        self.cifar10_path = flags.local_cifar10_path
        self.train_transforms = train_transforms

        if train_transforms is None:
            self.train_transforms = T.Compose([
                                    T.RandomRotation((-15,15)),
                                    T.RandomCrop(32, padding=4),
                                    T.RandomHorizontalFlip(),
                                    T.ToTensor()
                                ])

        # if test doesn't preprocess, it perform worse
        if val_test_transforms is None:
            self.val_test_transforms = T.Compose([
                    T.ToTensor(),
                ])
        else:
            self.val_test_transforms = val_test_transforms


        trainset = datasets.CIFAR10(self.cifar10_path, train=True, download=False, transform=self.train_transforms)
        trainset,val_set = torch.utils.data.random_split(trainset, [49000, 1000])

        # change the transforms of validation set
        val_set.dataset.transform = self.val_test_transforms

        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size)
        self.test_loader = torch.utils.data.DataLoader(
                    datasets.CIFAR10(self.cifar10_path, train=False, download=True, transform=self.val_test_transforms),
                    batch_size=self.batch_size)

    def get_train_loader(self):
        return self.train_loader
    def get_val_loader(self):
        return self.val_loader
    def get_test_loader(self):
        return self.test_loader


def get_imagenet_val_loader(path, batch_size, image_size=224,normalize=None, shuffle=False, num_workers=4):
    if normalize=='torch':
        dataset = ImageFolder(path,
                              T.Compose([
                                  T.Resize(image_size),
                                  T.CenterCrop(image_size),
                                  T.ToTensor(),
                                  T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                              ]))
    elif normalize=='tf':
        dataset = ImageFolder(path,
                              T.Compose([
                                  T.Resize(image_size),
                                  T.CenterCrop(image_size),
                                  T.ToTensor(),
                                  T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ]))
    else:
        dataset = ImageFolder(path,
                              T.Compose([
                                  T.Resize(image_size),
                                  T.CenterCrop(image_size),
                                  T.ToTensor()
                              ]))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       shuffle=shuffle, num_workers=num_workers,pin_memory=True)


# test utils
def predict(model, data, logits=False):
    '''
    :param model: classifier model
    :param data: numpy or torch train_data, with batch
    :param mean: same train_data may need normalize before input
    :return:  if logits is True, return logits, or else return label
    '''

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    data = data.to(device=flags.device, dtype=torch.float32)

    # get predicts
    with torch.no_grad():
        outs = model(data)
        if logits:
            return outs
        return outs.argmax(1)


def init_model(name):
    '''
    :param name: reflect, execute string code to get model
    :return:
    '''

    if "cifar" in name:
        pth = flags.model_pth[name]
        name += '()'
        model = eval(name)
        model.load_state_dict(torch.load(pth, map_location='cpu'))

    # imagenet model
    else:
        name += '()'
        model = eval(name)

    model = model.to(flags.device)
    model.eval()
    return model


# attack utils
def batch_norm(x, p=2):
    '''
    :param x: batch [b,c,w,h]
    :param p: p batch_norm
    :return: batch batch_norm
    '''
    return x.view(x.size(0), -1).norm(dim=1, p=p)

# x / batch_norm(x)
def normalize(x, p=2):
    '''
    :param x: batch
    :param p:
    :return: batch direction vector
    '''
    x_norm = batch_norm(x, p)
    x_norm = torch.max(x_norm, torch.ones_like(x_norm) *1e-6)        # in case divide zero
    return x / x_norm.view(-1, 1,1,1)

def clamp_by_2norm(x, epsilon):
    # avoid nan or inf if gradient is 0
    if (x == 0).any():
        x[x == 0] = torch.full_like(x[x == 0], 1e-6)
    x *= torch.clamp((epsilon * normalize(x, p=2)/ x), max=1.)
    return x


def torch2numpy(x):

    if x.device.type == 'cuda':
        x = x.detach()
    return x.cpu().numpy()


# defense utils

#  differentiable
class RP():
    def __init__(self, max_size=331, value=0.5, p=1):
        self.max_size = max_size
        self.value = value
        self.p = p

    def rp(self, x):
        rnd = np.random.randint(x.shape[-1], self.max_size)
        x = kornia.resize(x, size=(rnd, rnd))

        h_rem = self.max_size - rnd
        w_rem = self.max_size - rnd

        pad_left = np.random.randint(0, w_rem)
        pad_right = w_rem - pad_left
        pad_top = np.random.randint(0, h_rem)
        pad_bottom = h_rem - pad_top

        x = torch.nn.functional.pad(x, [pad_left, pad_right,pad_top, pad_bottom], mode='constant', value=self.value)
        return x

    def __call__(self, x:torch.Tensor):

        # perform transform
        if np.random.rand() < self.p:
            return self.rp(x)
        else:
            return x


class BitDepthReduce():

    def __init__(self, depth=3, p=1):
        '''
        :param depth: keeped bit depth
        '''
        self.depth = depth
        self.p = p

    def __call__(self, x):

        if np.random.rand() < self.p:
            x = torch.round(x*255)
            x = x.to(dtype=torch.uint8)

            shift = 8 - self.depth
            x = (x>>shift) << shift

            x = x.to(dtype=torch.float32) / 255

        return x



# non-differentiable
class JPEG():
    def __init__(self, quality=75, p=1):
        self.quality = quality
        self.p = p

    def _compression(self, x):
        x = np.transpose(x, (0, 2, 3, 1))
        res = []
        for arr in x:
            pil_image = PIL.Image.fromarray((arr * 255.0).astype(np.uint8))
            f = BytesIO()
            pil_image.save(f, format='jpeg', quality=self.quality)  # quality level specified in paper
            jpeg_image = np.asarray(PIL.Image.open(f)).astype(np.float32) / 255.0
            res.append(jpeg_image)

        x = np.stack(res, axis=0)
        return np.transpose(x, (0, 3, 1, 2))

    def __call__(self, x):

        if np.random.rand() < self.p:
            x_clone = x.detach().cpu().numpy()
            x_clone = self._compression(x_clone)

            return torch.from_numpy(x_clone).to(dtype=x.dtype, device=x.device)

        return x




class InputTransformModel(nn.Module):
    def __init__(self, model,  normalize=None, input_trans=None):
        super(InputTransformModel, self).__init__()

        self.model = model
        self.input_trans = input_trans
        self.normalize  = normalize

    def __normalize(self, x):
        mean = torch.tensor(self.normalize[0], dtype=x.dtype, device=x.device)
        std = torch.tensor(self.normalize[1], dtype=x.dtype, device=x.device)
        x = (x - mean.reshape(1, -1, 1, 1)) / std.reshape(1, -1, 1, 1)
        return x

    def forward(self, x):

        if self.input_trans:
            x = self.input_trans(x)

        if self.normalize:
            x = self.__normalize(x)

        return self.model(x)
