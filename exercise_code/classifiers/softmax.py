"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier

def naive_matrix_mult(x, y):
    I, K, J = *x.shape, y.shape[1]
    result = np.zeros((I, J))
    for i in range(I):
        for j in range(J):
            for k in range(K):
                result[i][j] += x[i][k] * y[k][j]
    return result

def naive_matrix_sum(x, y):
    n_row, n_col = x.shape
    for i in range(n_row):
        for j in range(n_col):
            x[i][j] += y[i][j]

def naive_matrix_scale(x, scalar):
    n_row, n_col = x.shape
    for i in range(n_row):
        for j in range(n_col):
            x[i][j] *= scalar

def cross_entropy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

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
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # DONE: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    # Matrix multiplication: Naive implementation
    n_samples = X.shape[0]
    n_classes = W.shape[1]
    y_hats = naive_matrix_mult(X, W)
    p_vals = np.zeros((1, n_classes)) # Store softmax outputs

    # Loss & gradient calculations
    for i in range(n_samples):
        actual_class_idx = y[i]      # for the ith sample
        all_class_scores = y_hats[i] # for the ith sample

        # Use epsilon to handle numerical instability in softmax
        eps = -np.max(all_class_scores)
        
        # Calculate denominator of softmax
        denom = 0.0
        for j in range(n_classes):
            denom += np.exp(all_class_scores[j] + eps)
        
        # Calculate nominator of softmax
        for j in range(n_classes):
            nom = np.exp(all_class_scores[j] + eps)
            p_vals[0, j] = nom / denom
        
        loss += -np.log(p_vals[0, actual_class_idx]) # Cross-entropy loss for ith sample
        p_vals[0, actual_class_idx] -= 1             # Subtract one from the predicted score of actual class
        dW_i = naive_matrix_mult(X[i].reshape(-1, 1), p_vals) # Calculate jacobian for ith sample
        naive_matrix_sum(dW, dW_i)                            # Add contribution of current sample to the overall gradient

    loss = loss / n_samples + reg         # Calculate the mean of losses, add regularization term
    naive_matrix_scale(dW, 1.0/n_samples) # Calculate the mean of jacobians
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW

def cross_entropy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    y_hats = X @ W # NxC
    actual_class_scores = np.choose(y, y_hats.T)        # Pick actual class scores for every sample: y_hats.T -> (CxN): Chooses the entry from each column vector wrt corresponding y entry
    sigma = np.log(np.sum(np.exp(y_hats), axis=1))      # Calculate log of sum of exp of y_hat entries
    loss = np.mean(-actual_class_scores + sigma) + reg  # Calculate the mean, add regularization term

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-7, 5e-7]
    regularization_strengths = [2.5e4, 5e4]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################

    pass

    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
