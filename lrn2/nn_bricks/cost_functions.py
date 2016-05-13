import theano.tensor as T

def kl(p, q):
    """
    Kullback-Leibler divergence
    """
    l = 1e-13
    return T.sum((p+l) * T.log((p+l)/(q+l)))

def cross_entropy(y,t):
    """
    Cost function representing the Cross Entropy error function
    for targets given inputs following the Bernoulli distribution

    See Eq. (5.21) in

    Bishop, C. M. (2009) Pattern Recognition and Machine Learning. Springer Verlag

    :type y: theano.tensor.matrix
    :param y: Output of the model

    :type t: theano.tensor.matrix
    :param t: Theano tensor. Target values

    :type cost: theano.tensor variable
    :return cost: Cross entropy error function

    """

    return T.nnet.binary_crossentropy(y,t).mean()

def squared_error(y,t):
    """
    Cost function representing the Mean Squared Error

    See Eq. (5.11) in

    Bishop, C. M. (2009) Pattern Recognition and Machine Learning. Springer Verlag

    :type y: theano.tensor.matrix
    :param y: Output of the model

    :type t: theano.tensor.matrix
    :param t: Theano tensor. Target values

    :type cost: theano.tensor variable
    :return cost: mean squared error
    """
    return T.mean((y-t)**2)

def categorical_crossentropy(y,t):
    """
    Cost function for standard multiclass classification problem in which input is assigned
    to one out of K mutually exclusive classes (Binary target variables that have a 1-of-K
    codinf scheme

    See Eq. (5.24) in

    Bishop, C. M. (2009) Pattern Recognition and Machine Learning. Springer Verlag

    :type y: theano.tensor.matrix
    :param y: Output of the model

    :type t: theano.tensor.matrix
    :param t: Theano tensor. Target values

    :type cost: theano.tensor variable
    :return cost: mean squared error
    """
    return T.nnet.categorical_crossentropy(y,t).mean()


def symbolic_softmax(x):
    """
    Softmax function to be used instead of the built-in T.nnet.softmax, since this does
    not work with T.grad (yet)

    See (5.25) in

    Bishop, C. M. (2009) Pattern Recognition and Machine Learning. Springer Verlag

    :type x: theano.tensor
    :param x: Theano variable or graph

    :type softmax: theano.tensor variable
    :return softmax: Softmax function
    """

    e = T.exp(x)
    return e / T.sum(e,axis = 1).dimshuffle(0,'x')


def identity_function(x):
    """
    Identity function, to be used as the activation in the RNN and NNStack classes

    :type x: theano.tensor
    :param x: Theano variable or graph

    :type x: theano.tensor variable
    :return x: same input variable
    """
    return 1.0*x

def accuracy(y,t):
    """
    Theano function of the
    """
    return T.eq(T.round(y), t).mean()
