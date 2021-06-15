# RUA
Automating Augmentation Through Random Unidimensional Search.


## Pre-requisites
* TensorFlow == 2.3.0
* PyTorch == 1.6.0
* FastEstimator == 1.1.1


### Run PyramidNet on Cifar10:
* RUA search:
```
cd pyramidnet_cifar10/rua
python pyramidnet_cifar10_rua.py
```
* Test on Cifar10 after finding optimal augmentation level:
```
cd pyramidnet_cifar10/final
fastestimator train pyramidnet_cifar10_final.py --level 24
```


### Run WRN-28-10 on Cifar10:
* RUA search:
```
cd wrn2810_cifar10/rua
python wrn2810_cifar10_rua.py
```
* Test on Cifar10 after finding optimal augmentation level:
```
cd wrn2810_cifar10/final
fastestimator train wrn2810_cifar10_final.py --level 18
```


### Run WRN-28-10 on Cifar100:
* RUA search:
```
cd wrn2810_cifar100/rua
python wrn2810_cifar100_rua.py
```
* Test on Cifar100 after finding optimal augmentation level:
```
cd wrn2810_cifar100/final
fastestimator train wrn2810_cifar100_final.py --level 23
```


### Run WRN-28-2 on SVHN_Cropped:
First download the cropped [dataset](http://ufldl.stanford.edu/housenumbers/), then organize the dataset like this:
```
- /data/SVHN_Cropped
    |- 0
        |- 0.png
        |- 00.png
    |- 1
    |- 2
    |- 3
    |- 4
    |- 5
    |- 6
    |- 7
    |- 8
    |- 9
```
* RUA search:
```
cd wrn282_svhn/rua
python wrn282_svhn_rua.py  # might need to change the data_dir in the file
```
* Test on Cifar100 after finding optimal augmentation level:
```
cd  wrn282_svhn/final
fastestimator train wrn282_svhn_rda_final_best.py --level 26 --data_dir /data/SVHN_Cropped
```