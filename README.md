# RUA
Automating Augmentation Through Random Unidimensional Search. To get more detail about the approach and results, please refer to our [paper](https://arxiv.org/pdf/2106.08756.pdf)


## Pre-requisites
* Python >= 3.6
* TensorFlow == 2.4.1
* PyTorch == 1.7.1

## Installation:
You can install the nightly version of fastestimator along with the above mentioned dependencies.

* Nightly (Linux/Mac):
    ``` bash
    $ pip install fastestimator-nightly
    ```

* Nightly (Windows):

    First download zip file [here](https://github.com/fastestimator/fastestimator/archive/master.zip)
    ``` bash
    $ pip install fastestimator-master.zip
    ```

Detailed installation instructions can be referred from [here](https://github.com/fastestimator/fastestimator)

## Docker Hub
Docker containers create isolated virtual environments that share resources with a host machine. Docker provides an easy way to set up a FastEstimator environment. You can simply pull our image from [Docker Hub](https://hub.docker.com/r/fastestimator/fastestimator/tags) and get started:
* Nighly:
    * GPU:
        ``` bash
        docker pull fastestimator/fastestimator:nightly-gpu
        ```
    * CPU:
        ``` bash
        docker pull fastestimator/fastestimator:nightly-cpu
        ```


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
Here, we are using SVHN Cropped digits dataset to get MNIST-like 32-by-32 images. You can refer to [dataset](http://ufldl.stanford.edu/housenumbers/) for more information.

* RUA search:
```
cd wrn282_svhn/rua
python wrn282_svhn_rua.py
```
* Test on Cifar100 after finding optimal augmentation level:
```
cd  wrn282_svhn/final
fastestimator train wrn282_svhn_final.py --level 26
```


## Citation
Please cite RUA in your publications if it helps your research:
```
@misc{dong2021automating,
      title={Automating Augmentation Through Random Unidimensional Search},
      author={Xiaomeng Dong and Michael Potter and Gaurav Kumar and Yun-Chan Tsai and V. Ratna Saripalli},
      year={2021},
      eprint={2106.08756},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```