import torch
import torch.nn as nn
import numpy as np


class NNTorch:
    """
    I used this as a guide:
    https://github.com/J-Yash/Creating-a-simple-NN-in-pytorch/blob/master/Creating%20a%20simple%20NN%20in%20PyTorch.ipynb

    I used it to understand pytorch, and my NN has more features than what they present.
    """
    def __init__(self, shape, mu=0.01, *args, **kwargs):  # throw away args and kwargs to meet NN api
        """
        Shape is a listable object of positive integers specifying how many
        nodes are in each layer. If input or output layer is unknown, they can
        be of type None. Activation functions default to relu for hidden layers
        and no activation function on the output layer.
        """
        self.N = None
        self.total_error = None
        self.num_samples = None
        self.net = None
        self.criterion = None
        self.optimizer = None

        self.set_shape(shape)

        # hyper params
        self.mu = mu

        # self.W and self.b
        self.init_net()

    @staticmethod
    def check_n_edge(edge):
        if edge is not None and not isinstance(edge, int):
            raise Exception("shape should be None or an integer")

    def set_shape(self, shape):
        """ sets self.N """
        self.N = list(shape)
        self.check_n_edge(self.N[0])
        for N in self.N[1:-1]:
            if not isinstance(N, int):
                raise Exception("shape interior must only be integers")
        self.check_n_edge(self.N[-1])

    @staticmethod
    def to_tensor(val):
        if isinstance(val, np.ndarray):
            ret = torch.from_numpy(val).to(torch.float)
        elif isinstance(val, list):
            ret = torch.Tensor(val)
        else:
            ret = val
        if len(ret) == 1:
            ret.unsqueeze(0)
        return ret

    def fit(self, x, y, verbose=False, iterations=1000):
        # x.shape = (n_samples, m_features)
        # y.shape = (n_samples, k_targets)
        # will change shape of weights matrix if sizes aren't as expected
        x = self.to_tensor(x)
        y = self.to_tensor(y)

        self.init_net(in_layer=x.size()[1], out_layer=y.size()[1])
        report_every = iterations//10
        for i in range(iterations):
            y_out = self.net.forward(x)
            self.optimizer.zero_grad()
            loss = self.criterion(y_out, y)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            if verbose and iterations > 10 and i % report_every == 0:
                print('iterations: {}; error: {}'.format(i, loss))

    def predict(self, x):
        # x.shape = (n_samples, m_features)
        x = self.to_tensor(x)
        return self.net.forward(x)

    def init_net(self, in_layer=None, out_layer=None):
        if self.N[0] is None and in_layer is not None:
            self.N[0] = in_layer
        if self.N[-1] is None and out_layer is not None:
            self.N[-1] = out_layer
        if self.net is not None or self.N[0] is None or self.N[-1] is None:
            return  # can't or already initialized
        layers = [nn.Linear(self.N[0], self.N[1])]
        for i in range(2, len(self.N)):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(self.N[i-1], self.N[i]))
        self.net = nn.Sequential(*layers)
        self.criterion = nn.MSELoss()  # TODO configurable
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.mu)  # TODO configurable
