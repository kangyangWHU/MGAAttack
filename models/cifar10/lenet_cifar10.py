from torch import nn


class LeNet_CIFAR10(nn.Module):
    """this LeNet model is used for cifar10"""
    def __init__(self, name='LeNet_CIFAR10'):
        super(LeNet_CIFAR10, self).__init__()
        self.name = name
        self.feature = nn.Sequential(
                nn.Conv2d(3,16,5),
                nn.MaxPool2d(2),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(16),
                nn.Conv2d(16,32,5),
                nn.MaxPool2d(2),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(32),
            )

        self.classifier = nn.Sequential(
                nn.Linear(32*5*5, 100),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(100),
                nn.Linear(100,10),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(-1, 32*5*5)
        return self.classifier(x)

def lenet_cifar10():
    return LeNet_CIFAR10()