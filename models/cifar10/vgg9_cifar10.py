from torch import nn


class VGG9_CIFAR10(nn.Module):
    """simplify the VGG model in order to adapt cifar10 83.82, when epochs == 10 or 15, lr /= 10"""

    def __init__(self, name='VGG9_CIFAR10'):
        super(VGG9_CIFAR10, self).__init__()

        self.name = name
        self.decay1 = True
        self.decay2 = True
        self.feature = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AvgPool2d(4)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(-1, 128)
        return self.classifier(x)

def vgg9_cifar10():
    return VGG9_CIFAR10()