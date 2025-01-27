"""Two Layer Network."""
# pylint: disable=invalid-name
import numpy as np


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension
    of N, a hidden layer dimension of H, and performs classification over C
    classes. We train the network with a softmax loss function and L2
    regularization on the weight matrices. The network uses a ReLU nonlinearity
    after the first fully connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each
    class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each
          y[i] is an integer in the range 0 <= y[i] < C. This parameter is
          optional; if it is not passed then we only return scores, and if it is
          passed then we instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c]
        is the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of
          training samples.
        - grads: Dictionary mapping parameter names to gradients of those
          parameters  with respect to the loss function; has the same keys as
          self.params.
        """
        # pylint: disable=too-many-locals
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, _ = X.shape

        # Compute the forward pass
        scores = None
        ########################################################################
        # DONE: Perform the forward pass, computing the class scores for the   #
        # input. Store the result in the scores variable, which should be an   #
        # array of shape (N, C).                                               #         
        ########################################################################

        O = X @ W1 + b1      # First fully connected layer output -> O: NxH
        H = np.maximum(0, O) # ReLU -> H: NxH
        scores = H @ W2 + b2 # Second fully connected layer output -> scores: NxC

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        ########################################################################
        # DONE: Finish the forward pass, and compute the loss. This should     #
        # include both the data loss and L2 regularization for W1 and W2. Store#
        # the result in the variable loss, which should be a scalar. Use the   #
        # Softmax classifier loss. So that your results match ours, multiply   #
        # the regularization loss by 0.5                                       #
        ########################################################################

        row_idxs = np.arange(N)
        
        # Softmax calculation
        eps = (-np.max(scores, axis=1, keepdims=True))
        nom = np.exp(scores + eps)
        denom = np.sum(nom, axis=1, keepdims=True)
        softmax_scores = nom / denom
        
        # Cross entropy loss with regularization term calculation
        actual_class_scores = softmax_scores[row_idxs, y]
        loss = -np.mean(np.log(actual_class_scores)) + (np.sum(W1**2) + np.sum(W2**2)) * (reg/2)

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # Backward pass: compute gradients
        grads = {}
        ########################################################################
        # DONE: Compute the backward pass, computing the derivatives of the    #
        # weights and biases. Store the results in the grads dictionary. For   #
        # example, grads['W1'] should store the gradient on W1, and be a matrix#
        # of same size                                                         #
        ########################################################################

        # dL/dsoftmax * dsoftmax/dscores
        softmax_scores[row_idxs, y] -= 1
        dL_dscores = softmax_scores / N           # NxC (Renaming the term for a better understanding)

        # dL/dscores * dscores/dW2 -> Note: Matrix multiplication order is not the same
        dscores_dW2 = H.T                         # HxN
        dL_dW2 = (dscores_dW2 @ dL_dscores)       # HxN * NxC -> HxC
        dL_db2 = np.sum(dL_dscores, axis=0)       # 1xN * NxC -> C
        grads['W2'] = dL_dW2 + reg * W2           # W1 cancels out, reg term halved based on description in the task
        grads['b2'] = dL_db2
        
        # dL/dscores * dscores/dH * dH/dO * dO/dW1 -> Note: Matrix multiplication order is not the same
        dscores_dH = W2.T                         # CxH
        dH_dO = np.where(O > 0, 1, 0)             # NxH -> NxH (Note: Alternatively: (dL_dH > 0).astype(int))
        dL_dO = dH_dO * (dL_dscores @ dscores_dH) # NxH . (NxC * CxH) -> NxH
        dO_dW1 = X.T                              # DxN

        dL_dW1 = (dO_dW1 @ dL_dO)                 # DxN * NxH -> DxH
        dL_db1 = np.sum(dL_dO, axis=0)            # 1xN * NxH -> H
        grads['W1'] = dL_dW1 + reg * W1           # W2 cancels out, reg term halved based on description in the task
        grads['b1'] = dL_db1

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning
          rate after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        # pylint: disable=too-many-arguments, too-many-locals
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            ####################################################################
            # DONE: Create a random minibatch of training data and labels,     #
            # storing hem in X_batch and y_batch respectively.                 #
            ####################################################################

            idxs = np.random.choice(num_train, batch_size)
            X_batch = X[idxs]
            y_batch = y[idxs]

            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            ####################################################################
            # DONE: Use the gradients in the grads dictionary to update the    #
            # parameters of the network (stored in the dictionary self.params) #
            # using stochastic gradient descent. You'll need to use the        #
            # gradients stored in the grads dictionary defined above.          #
            ####################################################################

            for key in self.params:
              self.params[key] -= learning_rate * grads[key]

            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each
          of the elements of X. For all i, y_pred[i] = c means that X[i] is
          predicted to have class c, where 0 <= c < C.
        """
        y_pred = None

        ########################################################################
        # DONE: Implement this function; it should be VERY simple!             #
        ########################################################################

        O = X @ self.params['W1'] + self.params['b1']
        H = np.maximum(0, O)
        scores = H @ self.params['W2'] + self.params['b2']
        y_pred = np.argmax(scores, axis=1)

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return y_pred


def neuralnetwork_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    best_net = None # store the best model into this 

    ############################################################################
    # DONE: Tune hyperparameters using the validation set. Store your best     #
    # trained model in best_net.                                               #
    #                                                                          #
    # To help debug your network, it may help to use visualizations similar to #
    # the  ones we used above in the Jupyther Notebook; these visualizations   #
    # will have significant qualitative differences from the ones we saw for   #
    # the poorly tuned network.                                                #
    #                                                                          #
    # Tweaking hyperparameters by hand can be fun, but you might find it useful#
    # to  write code to sweep through possible combinations of hyperparameters #
    # automatically like we did on the previous exercises.                     #
    ############################################################################

    results = {}
    best_val = -1
    # all_nets = []

    learning_rates = [1e-2, 1e-3, 1e-4]
    regularization_strengths = [1e-2, 1e-1, 1e1, 1e2]
    hidden_sizes = [50, 100, 150, 200]
    num_iters = 100

    input_size = X_train.shape[1]
    num_classes = y_train.shape[0]

    for lr in learning_rates:
      for reg in regularization_strengths:
        for hidden_size in hidden_sizes:
          net = TwoLayerNet(input_size, hidden_size, num_classes)
          net.train(X_train, y_train, X_val, y_val, learning_rate=lr, reg=reg,
                    num_iters=num_iters, verbose=True)

          y_train_pred = net.predict(X_train)
          train_acc = np.mean(y_train == y_train_pred)

          y_val_pred = net.predict(X_val)
          val_acc = np.mean(y_val == y_val_pred)

          print('lr %e' % lr)
          print('reg %e' % reg)
          print('hs %d' % hidden_size)
          print('training accuracy: %f' % train_acc)
          print('validation accuracy: %f' % val_acc)
          
          results[(lr, reg, hidden_size)] = (train_acc, val_acc)
          # all_nets.append((net, val_acc))
          
          if val_acc > best_val:
              best_val = val_acc
              best_net = net

      # Print out results.
      for (lr, reg, hidden_size) in sorted(results):
          train_accuracy, val_accuracy = results[(lr, reg)]
          print('lr %e reg %e hs %d train accuracy: %f val accuracy: %f' % (
                lr, reg, hidden_size, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return best_net
