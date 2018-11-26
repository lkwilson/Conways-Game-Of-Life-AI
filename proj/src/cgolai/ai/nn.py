import random
import numpy as np

from .actfunc import sigmoid, sigmoid_p, identity, identity_p


class NN:  # TODO: better descent
    def __init__(self, shape, mu=0.01, h=None, h_p=None, h_out=None, h_out_p=None, hs=None, hs_p=None):
        """
        Shape is a listable object of positive integers specifying how many
        nodes are in each layer. If input or output layer is unknown, they can
        be of type None.
    
        Activation Function:
            Activation functions default to relu for hidden layers and no
            activation function on the output layer. To change this behavior,
            set h, hs, or h_out.

            Setting h changes the default hidden layer activation function from
            relu. Setting h_out sets the activation function for the output
            layer. If either is specified, its corresponding h_p or h_out_p need
            to also be set.

            Setting hs changes the initial layer's activation functions with
            those specified by hs. If hs is too short, then the remaining
            layers use the defaults (h and h_out), even if specified.
        """
        hs = [] if hs is None else hs
        hs_p = [] if hs_p is None else hs_p
        self.N = None
        self.hs = None
        self.hs_p = None
        self.total_error = None
        self.num_samples = None
        self.W = None
        self.b = None
        self.x = None
        self.z = None
        self.e = None
        self.E = None
        self.initialized = False

        self.set_shape(shape)
        self.set_hs_hs_p(h, h_p, h_out, h_out_p, hs, hs_p)

        # hyper params
        self.mu = mu

        # self.W and self.b
        self.init_weights_and_biases()

    @staticmethod
    def check_n_edge(edge):
        if edge is not None and not isinstance(edge, int):
            raise Exception("shape should be None or an integer")

    @staticmethod
    def check_prime_defined(f, f_p):
        if f is not None:
            if not callable(f) or not callable(f_p):
                raise Exception("f and f_p must be callable")

    @staticmethod
    def get_default_h_h_p(h, h_p):
        return (sigmoid, sigmoid_p) if h is None else (h, h_p)

    @staticmethod
    def get_default_h_out_h_out_p(h_out, h_out_p):
        return (identity, identity_p) if h_out is None else (h_out, h_out_p)

    @staticmethod
    def init_bias(length):
        wk = np.random.rand(length)
        return random.choice([wk, -wk])

    @staticmethod
    def init_weight(shape):
        """(-1, 1)"""
        wk = np.random.rand(shape[0]*shape[1]).reshape(shape)
        return random.choice([wk, -wk])

    def set_shape(self, shape):
        """ sets self.N """
        self.N = list(shape)
        self.check_n_edge(self.N[0])
        for N in self.N[1:-1]:
            if not isinstance(N, int):
                raise Exception("shape interior must only be integers")
        self.check_n_edge(self.N[-1])

    def extend_hs_hs_p(self, h, h_p, h_out, h_out_p):
        hs_len = len(self.N)-1
        for i in range(len(self.hs), hs_len-1):
            self.hs.append(h)
            self.hs_p.append(h_p)
        if len(self.hs) < hs_len:
            self.hs.append(h_out)
            self.hs_p.append(h_out_p)

    def change_none_to_default(self):
        for i in range(len(self.hs[:-1])):
            self.hs[i], self.hs_p[i] = self.get_default_h_h_p(self.hs[i], self.hs_p[i])
        self.hs[-1], self.hs_p[-1] = self.get_default_h_out_h_out_p(self.hs[-1], self.hs_p[-1])

    def set_hs_hs_p(self, h, h_p, h_out, h_out_p, hs, hs_p):
        # initialize and check input
        self.hs = list(hs)
        self.hs_p = list(hs_p)
        self.check_prime_defined(h, h_p)
        self.check_prime_defined(h_out, h_out_p)
        for f, f_p in zip(self.hs, self.hs_p):
            self.check_prime_defined(f, f_p)

        # build self.hs and self.hs_p
        self.extend_hs_hs_p(h, h_p, h_out, h_out_p)
        self.change_none_to_default()

    def fit(self, x, y, verbose=False, iterations=1000):
        # x.shape = (n_samples, m_features)
        # y.shape = (n_samples, k_targets)
        # will change shape of weights matrix if sizes aren't as expected
        if not self.initialized:
            self.num_samples = x.shape[0]
            self.N[0] = x.shape[1]
            self.N[-1] = y.shape[1]
            self.init_weights_and_biases()

        report_every = iterations//10
        xy_zipped = zip(x, y)
        for i in range(iterations):
            self.total_error = 0
            for _x, _y in xy_zipped:
                self._train(_x, _y)
                self.total_error += self.E
            if verbose and iterations > 10 and i % report_every == 0:
                print('iterations: {}; error: {}'.format(i, self.total_error))

    def predict(self, X):
        # x.shape = (n_samples, m_features)
        return np.array([self.feed_forward(x) for x in X])

    def init_weights_and_biases(self):
        if self.N[0] is None or self.N[-1] is None:
            return
        self.W = []
        self.b = []
        for i in range(len(self.N)-1):
            weight_shape = (self.N[i+1], self.N[i])
            bias_length = weight_shape[0]
            self.W.append(self.init_weight(weight_shape))
            self.b.append(self.init_bias(bias_length))
        self.W = np.array(self.W)
        self.b = np.array(self.b)
        self.initialized = True

    def _train(self, x, y):
        # called for every sample
        grad_w, grad_b = self.gradient(x, y)
        self.W -= self.mu*grad_w
        self.b -= self.mu*grad_b

    def feed_forward(self, x):
        self.z = []
        self.x = [x]
        for W, b, h in zip(self.W, self.b, self.hs):
            x = self.x[-1]
            z = W@x+b
            hv = h(z)
            self.z.append(z)
            self.x.append(hv)
        self.x = np.array(self.x)
        self.z = np.array(self.z)
        return self.x[-1]

    def gradient(self, x, y):
        self.feed_forward(x)
        self.calc_error(y)

        # init
        i = len(self.N)-2
        base = -2*self.e*self.hs_p[i](self.z[i])
        grad_w = [np.outer(base, self.x[i])]
        grad_b = [np.copy(base)]
        while i > 0:
            base = (base@self.W[i])*self.hs_p[i-1](self.z[i-1])
            grad_w.insert(0, np.outer(base, self.x[i-1]))
            grad_b.insert(0, np.copy(base))
            i -= 1

        return np.array(grad_w), np.array(grad_b)

    def calc_error(self, y):
        # assumes feed_forward called
        self.e = y - self.x[-1]
        self.E = self.e @ self.e
