
# Expectigrad: Rectifying AdaGrad and Adam
[![License](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)

Introduction about Expectigrad.

### Pseudocode

>```
>Let x_1 be the initial network parameters
>Initialize sum of squared gradients s = 0
>
>for t = 1,...,T do
>    s += g^2
>    x_t = x_{t-1} - αg / (sqrt(s/t) + ε)
>end for
>
>return x_T
>```

### Citing

If you use this code in published work, please cite the original paper:

```
@inproceedings{daley2020expectigrad,
  title={{E}xpectigrad: Rectifying {A}da{G}rad and {A}dam},
  author={Daley, Brett and Amato, Christopher},
  booktitle={},
  pages={},
  year={2020}
}
```

## Installation

Use pip to quickly install Expectigrad:

```
pip install expectigrad
```

Alternatively, you can clone this repository and install manually:

```
git clone 
cd expectigrad
python setup.py -e .
```

## Usage

Depending on the deep learning framework you use, you will need to instantiate a different optimizer.
These optimizers plug into the frameworks as usual.
Import paths and class constructors are given below:

### Pytorch

```python
from expectigrad.pytorch import Expectigrad

Expectigrad(
    params, lr=1e-3, eps=1e-3
)
```

### Tensorflow 1.x

```python
from expectigrad.tensorflow1 import ExpectigradOptimizer

ExpectigradOptimizer(
    learning_rate=1e-3, epsilon=1e-3, use_locking=False, name='Expectigrad'
)
```

### Tensorflow 2.x

```python
from expectigrad.tensorflow2 import Expectigrad

Expectigrad(
    learning_rate=1e-3, epsilon=1e-3, name='Expectigrad', **kwargs
)
```
