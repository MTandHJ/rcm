

# Understanding Adversarial Robustness from Feature Maps of Convolutional Layers





## Requirements



- python==3.7.10
- torch==1.8.1
- torchvision==0.9.1



## Files



```
┌── data # the path of data
│	├── cifar10
│	└── cifar100
└── rcm
	└── AWP
        ├── autoattack # AutoAttack
        ├── infos # for saving trained model
        ├── logs # for logging
        ├── models # Architectures
        ├── src
            ├── ...
            ├── base.py # Coach, arranging the training procdure
            ├── config.py # You can specify the ROOT as the path of training data.
            ├── ...
            └── utils.py # other usful tools
        ├── auto_attack.py # Croce F.
        ├── AT.py # AT-AWP
        ├── TRADES.py # TRADES-AWP      
        └── white_box_attack.py # the white-box attacks due to foolbox
	└── Major # other defense methods       
```



## Reference Code



- AT:  https://github.com/MadryLab/cifar10_challenge
- TRADES:  https://github.com/yaodongyu/TRADES
- MART:  https://github.com/YisenWang/MART
- FAT:  https://github.com/zjfheart/Friendly-Adversarial-Training
- AWP: https://github.com/csdongxian/AWP
- Bag-AT:  https://github.com/P2333/Bag-of-Tricks-for-AT



