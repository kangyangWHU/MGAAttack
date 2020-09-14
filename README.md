### step 1: download weights from https://github.com/kangyangWHU/PyTorch_CIFAR10

please download googlenet, inceptionv3,resnet18,vgg16_bn, and vgg19_bn to models/cifar10/state_dicts/, then rename them to googlenet_cifar10.pt, inception_v3_cifar10.pt, resnet18_cifar10.pt, vgg16_bn_cifar10.pt, and vgg19_bn_cifar10.pt, respectively.

### step 2: replace your CIFAR-10, and ImageNet path in the config.py



### untargeted on ImageNet with RP defense

```
python main.py --ensemble_models  inceptionv4 xception inceptionresnetv2 --model inceptionv3 --dataset imagenet --epsilon 0.047 --max_queries 10000 --mr 0.001 --num_attack 1000 --defense_method RP
```



###  targeted on ImageNet without defense

```
python main.py --ensemble_models  inceptionv4 xception inceptionresnetv2 --model inceptionv3 --dataset imagenet --epsilon 0.047 --max_queries 10000 --mr 0.001 --num_attack 1000 --targeted
```



### targeted on CIFAR-10

```
python main.py --ensemble_models vgg16_bn_cifar10 resnet18_cifar10 googlenet_cifar10 --model vgg19_bn_cifar10 --dataset cifar10 --epsilon 0.03137 --max_queries 10000 --mr 0.001 --num_attack 2000 --targeted
```