import numpy as np

from deeplearning.layers import *
from deeplearning.fast_layers import *
from deeplearning.layer_utils import *
from deeplearning.classifiers.fc_net import *

# BestConvNet and BestConvNet2. The 2nd has a dropout to help with regularization
class BestConvNet(object):
    """
    A convolutional network with the following architecture:

    conv - relu - 2x2 max pool - [affine - batch_normalization - relu] x 2 - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=16, filter_size=3,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        C, H, W = input_dim
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = weight_scale * np.random.randn(int(num_filters * H * W / 4), hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, hidden_dim)
        self.params['b3'] = np.zeros(hidden_dim)
        self.params['W4'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b4'] = np.zeros(num_classes)
        # spatial bn layer CHANGE when adding spatial
        # self.params['gamma1'] =
        # self.params['beta1'] =

        # affine bn layer
        self.params['gamma1'] = np.ones(hidden_dim)
        self.params['beta1'] = np.zeros(hidden_dim)
        self.params['gamma2'] = np.ones(hidden_dim)
        self.params['beta2'] = np.zeros(hidden_dim)

        self.bn_params = [{'mode': 'train'} for i in range(2)]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

        print("finished init")

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        mode = 'test' if y is None else 'train'
        for bn_param in self.bn_params:
            bn_param[mode] = mode

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        gamma1, beta1 = self.params['gamma1'], self.params['beta1']
        gamma2, beta2 = self.params['gamma2'], self.params['beta2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        out, crp_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out2, abn_cache1 = affine_bn_relu_forward(out, W2, b2, gamma1, beta1, self.bn_params[0])
        out3, abn_cache2 = affine_bn_relu_forward(out2, W3, b3, gamma2, beta2, self.bn_params[1])
        scores, a_cache = affine_forward(out3, W4, b4)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, gradx = softmax_loss(scores, y)
        loss += 0.5 * self.reg * \
                np.sum([np.sum(np.square(self.params["W"+str(i)])) for i in range(1, 5)])


        dx4, gradW4, grads["b4"] = affine_backward(gradx, a_cache)
        dx3, gradW3, grads["b3"], grads['gamma2'], grads['beta2'] = affine_bn_relu_backward(dx4, abn_cache2)
        dx2, gradW2, grads["b2"], grads['gamma1'], grads['beta1'] = affine_bn_relu_backward(dx3, abn_cache1)
        dx1, gradW1, grads["b1"] = conv_relu_pool_backward(dx2, crp_cache)
        grads["W4"] = gradW4 + (self.reg * self.params["W4"])
        grads["W3"] = gradW3 + (self.reg * self.params["W3"])
        grads["W2"] = gradW2 + (self.reg * self.params["W2"])
        grads["W1"] = gradW1 + (self.reg * self.params["W1"])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads











##########################################################################################################################




class BestConvNet2(object):
    """
    A convolutional network with the following architecture:

    conv - relu - 2x2 max pool - [affine - batch_normalization - relu] x 2
    affine - relu - dropout - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=16, filter_size=3,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dropout = 0.1, dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        C, H, W = input_dim
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = weight_scale * np.random.randn(int(num_filters * H * W / 4), hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, hidden_dim)
        self.params['b3'] = np.zeros(hidden_dim)
        self.params['W4'] = weight_scale * np.random.randn(hidden_dim, hidden_dim)
        self.params['b4'] = np.zeros(hidden_dim)
        self.params['W5'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b5'] = np.zeros(num_classes)

        # affine bn layer
        self.params['gamma1'] = np.ones(hidden_dim)
        self.params['beta1'] = np.zeros(hidden_dim)
        self.params['gamma2'] = np.ones(hidden_dim)
        self.params['beta2'] = np.zeros(hidden_dim)

        self.dropout_param = {'mode': 'train', 'p': dropout}
        self.bn_params = [{'mode': 'train'} for i in range(2)]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

        print("finished init")

    def loss(self, X, y=None):
        """

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        mode = 'test' if y is None else 'train'
        for bn_param in self.bn_params:
            bn_param[mode] = mode

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        gamma1, beta1 = self.params['gamma1'], self.params['beta1']
        gamma2, beta2 = self.params['gamma2'], self.params['beta2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        out, crp_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out2, abn_cache1 = affine_bn_relu_forward(out, W2, b2, gamma1, beta1, self.bn_params[0])
        out3, abn_cache2 = affine_bn_relu_forward(out2, W3, b3, gamma2, beta2, self.bn_params[1])
        out4, ad_cache = affine_relu_dropout_forward(out3, W4, b4, self.dropout_param)
        scores, a_cache = affine_forward(out4, W5, b5)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, gradx = softmax_loss(scores, y)
        loss += 0.5 * self.reg * \
                np.sum([np.sum(np.square(self.params["W"+str(i)])) for i in range(1, 5)])


        dx5, grads["W5"], grads["b5"] = affine_backward(gradx, a_cache)
        dx4, grads["W4"], grads['b4'] = affine_relu_dropout_backward(dx5, ad_cache)
        dx3, grads["W3"], grads["b3"], grads['gamma2'], grads['beta2'] = affine_bn_relu_backward(dx4, abn_cache2)
        dx2, grads["W2"], grads["b2"], grads['gamma1'], grads['beta1'] = affine_bn_relu_backward(dx3, abn_cache1)
        dx1, grads["W1"], grads["b1"] = conv_relu_pool_backward(dx2, crp_cache)
        for i in range(1, 6):
            grads['W'+str(i)] += (self.reg * self.params["W" + str(i)])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # print('grads is: ', grads)
        return loss, grads



pass
