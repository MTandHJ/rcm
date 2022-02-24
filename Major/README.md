



## Usage



### Training



- Upsampling the inputs

> python AT.py resnet18 cifar10 --scale-factor=2 --mode=nearest --strides=1222
>
> python ALP.py resnet18 cifar10 --scale-factor=2 --mode=nearest --strides=1222
>
> python TRADES.py resnet18 cifar10 --scale-factor=2 --mode=nearest --strides=1222 --leverage=6



**Note:** AlexNet: alexnet, 1122; VGG16: vgg16_bn, 11222.



- Shrinking sliding strides

  [[ResNet18-1122](https://drive.google.com/file/d/11qOQWTRzYbttKHC2lFcUsLs4FYAeq13D/view?usp=sharing)]

> python AT.py resnet18 cifar10 --scale-factor=1 --strides=1122
>
> python ALP.py resnet18 cifar10 --scale-factor=1 --strides=1122
>
> python TRADES.py resnet18 cifar10 --scale-factor=1 --strides=1122 --leverage=6



### Evaluation



**Note:** You should keep the hyper-parameters consistent with training unless this is intentional. 



- AutoAttack

> python auto_attack.py resnet18 cifar10 /path/of/saved/models



- Other white attacks provided by FoolBox

> python white_box_attack.py resnet18 cifar10 /path/of/saved/models --attack=pgd-linf --steps=20 --stepsize=0.25

