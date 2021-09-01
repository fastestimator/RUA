# RUA
Automating Augmentation Through Random Unidimensional Search. To get more detail about the approach and results, please refer to our [paper](https://arxiv.org/pdf/2106.08756.pdf)


## Pre-requisites
* TensorFlow == 2.4.1
* PyTorch == 1.7.1
* FastEstimator == 1.2.0


### Run PyramidNet on Cifar10:
* RUA search:
```
cd pyramidnet_cifar10/rua
fastestimator run pyramidnet_cifar10_rua.py
```
* After finding optimal augmentation level:
```
cd pyramidnet_cifar10/final
fastestimator train pyramidnet_cifar10_final.py
```


### Run WRN-28-10 on Cifar10:
* RUA search:
```
cd wrn2810_cifar10/rua
fastestimator run wrn2810_cifar10_rua.py
```
* After finding optimal augmentation level:
```
cd wrn2810_cifar10/final
fastestimator train wrn2810_cifar10_final.py
```


### Run WRN-28-10 on Cifar100:
* RUA search:
```
cd wrn2810_cifar100/rua
fastestimator run wrn2810_cifar100_rua.py
```
* After finding optimal augmentation level:
```
cd wrn2810_cifar100/final
fastestimator train wrn2810_cifar100_final.py
```


### Run WRN-28-2 on SVHN_Cropped:
* RUA search:
```
cd wrn282_svhn/rua
fastestimator run wrn282_svhn_rua.py
```
* After finding optimal augmentation level:
```
cd  wrn282_svhn/final
fastestimator train wrn282_svhn_final.py
```

### Run Resnet50 on ImageNet:
First please download the ImageNet dataset [here](https://image-net.org/). Then organize your folder like this:
```
- /data/imagenet/train
    |- class1
        |- image1.png
        |- image2.png
        |- ...
    |- ...
    |- class1000

- /data/imagenet/val
    |- class1
        |- image1.png
        |- image2.png
        |- ...
    |- ...
    |- class1000
```
* RUA search:
```
cd resnet50_imagenet/rua
fastestimator run resnet50_imagenet_rua.py --data_dir /data/imagenet
```
* After finding optimal augmentation level:
```
cd  wrn282_svhn/final
fastestimator train wrn282_svhn_final.py --data_dir /data/imagenet
```
