import torch
import numpy as np


class NNTorch:
    """
    I used this as a guide:
    https://github.com/J-Yash/Creating-a-simple-NN-in-pytorch/blob/master/Creating%20a%20simple%20NN%20in%20PyTorch.ipynb

    I used it to understand pytorch, and my NN has more features than what they present.
    """
    def __init__(self, shape, mu=0.01, h=None, optim=None):  # toss args
        """
        Shape is a listable object of positive integers specifying how many
        nodes are in each layer. If input or output layer is unknown, they can
        be of type None. Activation functions default to relu for hidden layers
        and no activation function on the output layer.
        """
        self.shape = list(shape)
        self.mu = mu
        self.h = h if h is not None else torch.nn.ReLU
        self.optim = optim if optim is not None else torch.optim.Adam
        self.nn = None
        self.criterion = None
        self.optimizer = None
        self._default_fit_iterations = 1000
        self._cuda = torch.cuda.is_available()

        for n in self.shape[1:-1]:
            if not isinstance(n, int):
                raise TypeError("expected int for shape")

        # hyper params
        self._is_trained = False
        self.init_net()

    def is_trained(self):
        return self._is_trained

    def to_tensor(self, val):
        if isinstance(val, np.ndarray):
            ret = torch.from_numpy(val)
        elif isinstance(val, list):
            ret = torch.Tensor(val)
        else:
            ret = val
        if self._cuda:
            return ret.cuda().double()
        else:
            return ret.double()

    def fit(self, x, y, verbose=False, iterations=None):
        # x.shape = (n_samples, m_features)
        # y.shape = (n_samples, k_targets)
        # prep x, y and nn
        if iterations is None:
            iterations = self._default_fit_iterations
        x = self.to_tensor(x)
        y = self.to_tensor(y)
        if self.nn is None:
            self.init_net(in_layer=x.size()[1], out_layer=y.size()[1])

        report_every = iterations//10
        for i in range(iterations):
            # forward pass
            y_out = self.nn(x)
            loss = self.criterion(y_out, y)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # report
            if verbose and iterations > 10 and i % report_every == 0:
                print('iterations: {}; error: {}'.format(i, loss))
        self._is_trained = True
        return x, y  # give formatted x, y

    def predict(self, x, numpy=True):
        # x.shape = (n_samples, m_features)
        x = self.to_tensor(x)
        if not numpy:
            return x
        return np.array(self.nn(x).detach())

    def init_net(self, in_layer=None, out_layer=None):
        if self.shape[0] is None and in_layer is not None:
            self.shape[0] = in_layer
        if self.shape[-1] is None and out_layer is not None:
            self.shape[-1] = out_layer
        if self.nn is not None or self.shape[0] is None or self.shape[-1] is None:
            return  # can't or already initialized

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        layers = [torch.nn.Linear(self.shape[0], self.shape[1])]
        for i in range(2, len(self.shape)):
            layers.append(self.h())
            layers.append(torch.nn.Linear(self.shape[i-1], self.shape[i]))
        self.nn = torch.nn.Sequential(*layers).to(device).double()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = self.optim(self.nn.parameters(), lr=self.mu)
