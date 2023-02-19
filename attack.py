
import os
import argparse



parser = argparse.ArgumentParser()
# parser.add_argument("scp", type=str)
parser.add_argument("--device", type=int, default=0)
opts = parser.parse_args()


seeds = [0, 1, 2, 3, 4]
strides = ["1122"]


command_type = "CUDA_VISIBLE_DEVICES={device} python auto_attack.py resnet18 cifar10 --strides={stride} ./infos/MART/cifar10-resnet18/{seed}\=1-nearest\=5.0\=MART-sgd-0.01-0.0035\=pgd-linf-0.0314-0.00700-10\=128\=default/ "


for stride in strides:
    for seed in seeds:
        command = command_type.format(
            device=opts.device,
            stride=stride, seed=seed
        )
        print(f"====={command}======")
        os.system(command)



