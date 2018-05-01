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
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    scoresExp = np.exp(scores)
    loss += -np.log(scoresExp[y[i]]/np.sum(scoresExp))
    dW[:,y[i]] += -(np.sum(scoresExp)-scoresExp[y[i]])/np.sum(scoresExp)*X[i]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      dW[:,j] += scoresExp[j]/np.sum(scoresExp) * X[i]

  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*reg*W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  scoresExp = np.exp(scores)
  scoresExpSum = np.sum(scoresExp,axis=1)
  _index = np.arange(num_train)
  correct_class_scoresExp = scoresExp[_index,y]
  loss = np.sum(-np.log(correct_class_scoresExp/scoresExpSum))
  loss /= num_train
  loss += reg * np.sum(W * W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scoresExpSum = np.tile(scoresExpSum,(num_classes, 1)).T
  p = scoresExp/scoresExpSum
  p[_index, y] = p[_index,y]- 1
  dW = X.T.dot(p)
  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

