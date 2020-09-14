import torch.nn as nn

__all__ = [ 'vgg13_bn_cifar10', 'vgg13_bn_cifar10_trades', 'vgg16_bn_cifar10','vgg16_bn_cifar10_pgd','vgg19_bn_cifar10', 'vgg19_bn_cifar10_pgd']

class VGG(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm):

    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm))
    return model

#
# def vgg11(pretrained=False, progress=True, **kwargs):
#     """VGG 11-layer model (configuration "A")
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)

#
# def vgg11_bn(pretrained=False, progress=True, device='cpu', **kwargs):
#     """VGG 11-layer model (configuration "A") with batch normalization
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _vgg('vgg11_bn', 'A', True, pretrained, progress, device, **kwargs)
#
#
# def vgg13(pretrained=False, progress=True, **kwargs):
#     """VGG 13-layer model (configuration "B")
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn_cifar10():
    return _vgg('vgg13_bn', 'B', True)


def vgg13_bn_cifar10_trades():
    return _vgg('vgg13_bn', 'B', True)



def vgg16_bn_cifar10():
    return _vgg('vgg16_bn', 'D', True)


def vgg16_bn_cifar10_pgd():
    return _vgg('vgg16_bn', 'D', True)


def vgg19_bn_cifar10():
    return _vgg('vgg19_bn', 'E', True)

def vgg19_bn_cifar10_pgd():
    return _vgg('vgg19_bn', 'E', True)