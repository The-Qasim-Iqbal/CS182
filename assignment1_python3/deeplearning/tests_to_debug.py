from deeplearning.fast_layers import conv_forward_fast, conv_backward_fast
from deeplearning.layers import *
from time import time
from deeplearning.classifiers.convnet import *

import numpy as np
import matplotlib.pyplot as plt
from deeplearning.classifiers.cnn import *
from deeplearning.data_utils import get_CIFAR10_data
from deeplearning.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from deeplearning.layers import *
from deeplearning.fast_layers import *
from deeplearning.solver import Solver

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

data = get_CIFAR10_data()
for k, v in data.items():
    print ('%s: ' % k, v.shape)

np.random.seed(35)
unique_model2 = BestConvNet2(weight_scale=.001, hidden_dim=125, reg=.1, filter_size = 5, num_filters=32)
unique_solver2 = Solver(unique_model2, data,
    num_epochs=8, batch_size=64,
    update_rule='adam',
    optim_config={
      'learning_rate': .0001,
    },
    verbose=True, print_every=50)
unique_solver2.train()