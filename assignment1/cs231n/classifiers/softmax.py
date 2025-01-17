from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_trains = X.shape[0]
    num_class = W.shape[1]

    for i in range(num_trains):
      score = X[i].dot(W)
      f = score - np.max(score)   # 减去最大值防止计算时数值错误
      softmax = np.exp(f) / np.sum(np.exp(f))
      loss_i = -np.log(softmax[y[i]])
      loss += loss_i

      for j in range(num_class):
        if j == y[i]:
          dW[:, j] += X[i] * softmax[j] - X[i]
        else:
          dW[:, j] += X[i] * softmax[j]
        
    loss /= num_trains
    dW /= num_trains

    loss += reg * np.sum(W * W)
    dW += reg * 2 * W 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_trains = X.shape[0]
    scores = X.dot(W)
    max_val = np.max(scores, axis=1).reshape(num_trains, 1)  # 挑选出每个样本中softmax的最大值
    softmax = scores - max_val
    softmax = np.exp(softmax) / np.sum(np.exp(softmax), axis=1).reshape(num_trains, 1)
    loss = -np.sum(np.log(softmax[np.arange(num_trains), y]))
  

    softmax[np.arange(num_trains), y] -= 1    # 根据循环法的公式，当j=y[i]时，需要减去X[i]；其他情况下更新softmax*X[i]
    dW = X.T.dot(softmax)

    loss /= num_trains
    dW /= num_trains

    loss += reg * np.sum(W * W)
    dW += 2 * reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
