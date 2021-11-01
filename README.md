# Enhancing Knowledge Tracing via Adversarial Training
This repository contains source code for the paper "[Enhancing Knowledge Tracing via Adversarial Training](https://dl.acm.org/doi/pdf/10.1145/3474085.3475554)" to be presented at ACM MM 2021 (**Oral**).


## Requirements
```sh
PyTorch==1.7.0
Python==3.8.0
```

## Usage

### Cloning the repository
```
git clone git@github.com:xiaopengguo/ATKT.git
cd ATKT
```

### Running
We evaluate our method on four datasets including **Statics2011**, **ASSISTments2009**, **ASSISTments2015** and **ASSISTments2017**.

#### Statics2011
```
python main.py --dataset 'statics'
```

#### ASSISTments2009
```
python main.py --dataset 'assist2009_pid'
```

#### ASSISTments2015
```
python main.py --dataset 'assist2015'
```

#### ASSISTments2017
```
python main.py --dataset 'assist2017_pid'
```
Evaluated results (AUC scores) will be saved in **statics_test_result.txt**, **assist2009_pid_test_result.txt**, **assist2015_test_result.txt**, and **assist2017_pid_test_result.txt**, respectively.


## Acknowledgments
Code and datasets are borrowed from [AKT](https://github.com/arghosh/AKT). Adversarial training implementation is inspired by [adversarial_training](https://github.com/WangJiuniu/adversarial_training). Early stopping implementation is modified from [early-stopping-pytorch](https://github.com/Bjarten/early-stopping-pytorch).
    
### Reference

```  
@inproceedings{guo2021enhancing,
  title={Enhancing Knowledge Tracing via Adversarial Training},
  author={Guo, Xiaopeng and Huang, Zhijie and Gao, Jie and Shang, Mingyu and Shu, Maojing and Sun, Jun},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={367--375},
  year={2021}
}
``` 

