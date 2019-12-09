import numpy as np
from utils import SGDOptimizer,AdamOptimizer
from utils import gen_batches, accuracy_score, label_binarize
from utils import log_loss, softmax, relu, inplace_relu_derivative
import warnings
import matplotlib.pyplot as plt

class MLPClassifier():

    """
    MLP classification.
    MLPClassifier trains iteratively since at each time step
    the partial derivatives of the loss function with respect to the model
    parameters are computed to update the parameters.

    Attributes
    ----------
    classes_ : array or list of array of shape (n_classes,)
        Class labels for each output.
    loss_ : float
        The current loss computed with the loss function.
    coefs_ : list, length n_layers - 1
        The ith element in the list represents the weight matrix corresponding
        to layer i.
    intercepts_ : list, length n_layers - 1
        The ith element in the list represents the bias vector corresponding to
        layer i + 1.
    n_iter_ : int,
        The number of iterations the solver has ran.
    n_layers_ : int
        Number of layers.
    n_outputs_ : int
        Number of outputs.
    out_activation_ : string
        Name of the output activation function.
    """

    def __init__(self,hidden_layer_sizes=(100,),
                 solver='adam',
                 batch_size= 200,
                 learning_rate=0.001,
                 epoches=200,
                 random_state=1,
                 tol=1e-4,
                 verbose=True,
                 momentum=0.0,
                 nesterovs_momentum=False,
                 early_stopping=False,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-8,
                 n_iter_no_change=10):

        self.solver = solver
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epoches = epoches
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.evaluation = False

    def predict(self, X):
        """Predict using the multi-layer perceptron classifier
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.
        Returns
        -------
        y : array-like, shape (n_samples,) or (n_samples, n_classes)
            The predicted classes.
        """
        y_pred = self._predict(X)

        return np.argmax(y_pred, axis = 1)

    def _predict(self, X):
        """Predict using the trained model
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.
        Returns
        -------
        y_pred : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The decision function of the samples for each class in the model.
        """

        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        layer_units = [X.shape[1]] + hidden_layer_sizes + \
                      [self.n_outputs_]

        # Initialize layers
        activations = [X]

        for i in range(self.n_layers_ - 1):
            activations.append(np.empty((X.shape[0],
                                         layer_units[i + 1])))
        # forward propagate
        self._forward_pass(activations)
        y_pred = activations[-1]

        return y_pred

    def fit(self, X, y, evaluation_set = None):
        """Fit the model to data matrix X and target(s) y.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : returns a trained MLP model.
        """
        # set evalutation to True
        if evaluation_set != None:
            self.evaluation = True
        if self.evaluation == True:
            self._X_eval = evaluation_set['X_eval']
            self._y_eval = np.ravel(evaluation_set['y_eval'])

        return self._fit(X, y)

    def _fit(self, X, y):
        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]

        hidden_layer_sizes = list(hidden_layer_sizes)

        # check wheher hidden layer sizes valid or not
        if np.any(np.array(hidden_layer_sizes) <= 0):
            raise ValueError("hidden_layer_sizes must be > 0, got %s." %hidden_layer_sizes)

        # transform y to n_classes ndarray for multiclass
        X, y = self._validate_input(X, y)

        n_samples, n_features = X.shape      # n_samples , n_features

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        self.n_outputs_ = y.shape[1]        # n_outputs

        layer_units = ([n_features] + hidden_layer_sizes + [self.n_outputs_])   # layers

        # check random state
        self._random_state = np.random.RandomState(self.random_state)

        # initialization
        self._initialize(y, layer_units)

        # get batch_sizes - clip to ensure batchsize in (1, n_samples)
        if self.batch_size < 1 or self.batch_size > n_samples:
            warnings.warn("Got `batch_size` less than 1 or larger than "
                              "sample size. It is going to be clipped")
        batch_size = np.clip(self.batch_size, 1, n_samples)

        # Initialize lists
        activations = [X]
        activations.extend(np.empty((batch_size, n_fan_out))
                           for n_fan_out in layer_units[1:])
        deltas = [np.empty_like(a_layer) for a_layer in activations]

        coef_grads = [np.empty((n_fan_in_, n_fan_out_)) for n_fan_in_,
                                                            n_fan_out_ in zip(layer_units[:-1],
                                                                              layer_units[1:])]

        intercept_grads = [np.empty(n_fan_out_) for n_fan_out_ in
                           layer_units[1:]]

        # Run the Stochastic optimization solver
        self._fit_stochastic(X, y, activations, deltas, coef_grads,
                                 intercept_grads, layer_units)

        return self

    '''
    SGD or Adam Fit
    '''
    def _fit_stochastic(self, X, y, activations, deltas, coef_grads,
                        intercept_grads, layer_units):

        params = self.coefs_ + self.intercepts_

        if self.solver == 'sgd':
            self._optimizer = SGDOptimizer(
                params, self.learning_rate,
                self.momentum, self.nesterovs_momentum)
        elif self.solver == 'adam':
            self._optimizer = AdamOptimizer(
                params, self.learning_rate, self.beta_1, self.beta_2,
                self.epsilon)

        # early_stopping
        early_stopping = self.early_stopping
        if self.evaluation:
            X_val = self._X_eval
            y_val = self._y_eval
        else:
            X_val = None
            y_val = None

        # number  of samples
        n_samples = X.shape[0]

        # batch size
        batch_size = np.clip(self.batch_size, 1, n_samples)

        # training process
        try:
            # every epoch
            for it in range(self.epoches):
                # loss
                accumulated_loss = 0.0

                # every batches
                for batch_slice in gen_batches(n_samples, batch_size):
                    activations[0] = X[batch_slice]

                    # backpropogation
                    batch_loss, coef_grads, intercept_grads = self._backprop(
                        X[batch_slice], y[batch_slice], activations, deltas,
                        coef_grads, intercept_grads)
                    # loss
                    accumulated_loss += batch_loss * (batch_slice.stop -
                                                      batch_slice.start)

                    # update weights
                    grads = coef_grads + intercept_grads
                    self._optimizer.update_params(grads)

                self.n_iter_ += 1
                self.loss_ = accumulated_loss / X.shape[0]

                # append loss
                self.loss_curve_.append(self.loss_)

                # append training acore
                self.train_scores_.append(self.score(X, np.argmax(y, axis = 1)))


                if self.evaluation == False:
                    if self.verbose:
                        print("Iteration %d, loss = %.8f Training score: %f" % (self.n_iter_,self.loss_, self.train_scores_[-1]))
                else:
                    # compute validation score, use that for stopping
                    self.validation_scores_.append(self.score(X_val, y_val))

                    if self.verbose:
                        print("Iteration %d, loss = %.8f Training score: %f Validation score: %f" % (self.n_iter_,self.loss_, self.train_scores_[-1],
                                                                                  self.validation_scores_[-1]))

                # update no_improvement_count based on training loss or
                # validation score according to early_stopping
                self._update_no_improvement_count(early_stopping, X_val, y_val)

                if self._no_improvement_count > self.n_iter_no_change:
                    # not better than last `n_iter_no_change` iterations by tol
                    # stop or decrease learning rate
                    if early_stopping:
                        msg = ("Early stop. Validation score did not improve more than "
                               "tol=%f for %d consecutive epochs." % (
                                   self.tol, self.n_iter_no_change))
                        print(msg)
                        break

                    self._no_improvement_count = 0

        except KeyboardInterrupt:
            warnings.warn("Training interrupted by user.")

        # restore best weights
        self.coefs_ = self._best_coefs
        self.intercepts_ = self._best_intercepts

    # forward pass
    def _forward_pass(self, activations):
        """Perform a forward pass on the network by computing the values
        of the neurons in the hidden layers and the output layer.
        Parameters
        ----------
        activations : list, length = n_layers - 1
            The ith element of the list holds the values of the ith layer.
        """
        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 1):
            activations[i + 1] = np.dot(activations[i], self.coefs_[i])
            activations[i + 1] += self.intercepts_[i]

            # For the hidden layers
            if (i + 1) != (self.n_layers_ - 1):
                activations[i + 1] = relu(activations[i + 1])

        # For the last layer
        activations[i + 1] = softmax(activations[i + 1])

        return activations

    # backpropagation for one layer
    def _compute_loss_grad(self, layer, n_samples, activations, deltas,
                           coef_grads, intercept_grads):
        """Compute the gradient of loss with respect to coefs and intercept for
        specified layer.
        This function does backpropagation for the specified one layer.
        """
        coef_grads[layer] = np.dot(activations[layer].T,
                                   deltas[layer])
        coef_grads[layer] /= n_samples

        intercept_grads[layer] = np.mean(deltas[layer], 0)

        return coef_grads, intercept_grads

    # bp
    def _backprop(self, X, y, activations, deltas, coef_grads,
                  intercept_grads):
        """Compute the MLP loss function and its corresponding derivatives
        with respect to each parameter: weights and bias vectors.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data.
        y : array-like, shape (n_samples,)
            The target values.
        activations : list, length = n_layers - 1
             The ith element of the list holds the values of the ith layer.
        deltas : list, length = n_layers - 1
            The ith element of the list holds the difference between the
            activations of the i + 1 layer and the backpropagated error.
            More specifically, deltas are gradients of loss with respect to z
            in each layer, where z = wx + b is the value of a particular layer
            before passing through the activation function
        coef_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            coefficient parameters of the ith layer in an iteration.
        intercept_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            intercept parameters of the ith layer in an iteration.
        Returns
        -------
        loss : float
        coef_grads : list, length = n_layers - 1
        intercept_grads : list, length = n_layers - 1
        """
        n_samples = X.shape[0]

        # Forward propagate
        activations = self._forward_pass(activations)

        # Get loss

        loss = log_loss(y, activations[-1])

        # Backward propagate
        last = self.n_layers_ - 2

        # The calculation of delta[last] here works with following
        # combinations of output activation and loss function:
        # sigmoid and binary cross entropy, softmax and categorical cross
        # entropy, and identity with squared loss
        deltas[last] = activations[-1] - y

        # Compute gradient for the last layer
        coef_grads, intercept_grads = self._compute_loss_grad(
            last, n_samples, activations, deltas, coef_grads, intercept_grads)

        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 2, 0, -1):
            deltas[i - 1] = np.dot(deltas[i], self.coefs_[i].T)
            inplace_relu_derivative(activations[i], deltas[i - 1])

            coef_grads, intercept_grads = self._compute_loss_grad(
                i - 1, n_samples, activations, deltas, coef_grads,
                intercept_grads)

        return loss, coef_grads, intercept_grads


    """ transform y input"""
    def _validate_input(self, X, y):

        y = np.ravel(y)
        self.classes_ = list( np.arange(np.max(y)+1))
        y = label_binarize(y, self.classes_)

        return X, y

    """"initialization"""
    def _initialize(self, y, layer_units):
        # set all attributes, allocate weights etc for first call
        # Initialize parameters
        self.n_iter_ = 0
        self.n_outputs_ = y.shape[1]

        # Compute the number of layers
        self.n_layers_ = len(layer_units)

        # Initialize coefficient and intercept layers
        self.coefs_ = []
        self.intercepts_ = []

        for i in range(self.n_layers_ - 1):
            coef_init, intercept_init = self._init_coef(layer_units[i],
                                                        layer_units[i + 1])
            self.coefs_.append(coef_init)
            self.intercepts_.append(intercept_init)


        self.loss_curve_ = []
        self._no_improvement_count = 0
        self.train_scores_ = []
        if self.evaluation:
            self.validation_scores_ = []
            self.best_validation_score_ = -np.inf
        else:
            self.best_loss_ = np.inf

    """Iinitialize coefficients"""
    def _init_coef(self, fan_in, fan_out):
        # Use the initialization method recommended by
        # Glorot et al.
        factor = 6.
        init_bound = np.sqrt(factor / (fan_in + fan_out))

        # Generate weights and bias:
        coef_init = self._random_state.uniform(-init_bound, init_bound,
                                               (fan_in, fan_out))
        intercept_init = self._random_state.uniform(-init_bound, init_bound,
                                                    fan_out)
        return coef_init, intercept_init


    '''for early stopping'''
    def _update_no_improvement_count(self, early_stopping, X_val, y_val):
        if self.evaluation:
            # update best parameters
            # use validation_scores_, not loss_curve_
            last_valid_score = self.validation_scores_[-1]

            if last_valid_score < (self.best_validation_score_ +
                                   self.tol):
                self._no_improvement_count += 1
            else:
                self._no_improvement_count = 0

            if last_valid_score > self.best_validation_score_:
                self.best_validation_score_ = last_valid_score
                self._best_coefs = [c.copy() for c in self.coefs_]
                self._best_intercepts = [i.copy()
                                         for i in self.intercepts_]
        else:
            # if evalution not set use loss update no_improvement_count
            if self.loss_curve_[-1] > self.best_loss_ - self.tol:
                self._no_improvement_count += 1
            else:
                self._no_improvement_count = 0
            if self.loss_curve_[-1] < self.best_loss_:
                self.best_loss_ = self.loss_curve_[-1]
                self._best_coefs = [c.copy() for c in self.coefs_]
                self._best_intercepts = [i.copy() for i in self.intercepts_]


    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.
            Parameters
            ----------
            X : array-like, shape = (n_samples, n_features)
                Test samples.
            y : array-like, shape = (n_samples) or (n_samples, n_outputs)
                True labels for X.
            Returns
            -------
            score : float Mean accuracy of self.predict(X) wrt. y.
        """
        return accuracy_score(y, self.predict(X))


    def plot_graph(self):
        if self.evaluation:
            plt.plot(self.train_scores_, label='train accuracy')
            plt.plot(self.validation_scores_, label='val accuracy')
            plt.legend(loc='best')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title('Learning Curve')
            plt.grid()
            plt.savefig('learning_curve.png')
        else:
            plt.plot(self.train_scores_, label='train accuracy')
            plt.legend(loc='best')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title('Learning Curve')
            plt.grid()
            plt.savefig('learning_curve.png')

