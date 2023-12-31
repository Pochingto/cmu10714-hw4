"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np

from functools import reduce

class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(fan_in=in_features, fan_out=out_features, device=device, dtype=dtype))
        if bias:
            self.bias = init.kaiming_uniform(fan_in=out_features, fan_out=1, device=device, dtype=dtype)
            self.bias = Parameter(self.bias.transpose())
        else:
            self.bias = Parameter(init.zeros(1, out_features, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.matmul(X, self.weight) + ops.broadcast_to(self.bias, (X.shape[0], self.weight.shape[1]))
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        size = reduce(lambda a, b: a * b, X.shape)
        return X.reshape((X.shape[0], size // X.shape[0]))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        ones = init.ones_like(x, device=x.device)
        return ones / (ones + ops.exp(-x))
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)

        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        one_hot_y = init.one_hot(logits.shape[1], y, device=logits.device)
        return (ops.summation(ops.logsumexp(logits, (1,)) / logits.shape[0]) - ops.summation(one_hot_y * logits / logits.shape[0]))
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mean = ops.summation(x, axes=0) / x.shape[0]
            devi = x + (-1) * ops.broadcast_to(mean.reshape((1, x.shape[1])), x.shape)
            variance = ops.summation(devi * devi, axes=0) / x.shape[0]

            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * mean.data
            self.running_var.data = (1 - self.momentum) * self.running_var.data + self.momentum * variance.data
        else:
            mean = self.running_mean
            devi = x + (-1) * ops.broadcast_to(mean.reshape((1, x.shape[1])), x.shape)
            variance = self.running_var

        normalized = devi / ops.broadcast_to(ops.power_scalar(variance + self.eps, 0.5).reshape((1, x.shape[1])), x.shape)
        return normalized * ops.broadcast_to(self.weight.reshape((1, x.shape[1])), normalized.shape) + ops.broadcast_to(self.bias.reshape((1, x.shape[1])), normalized.shape)
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mean = ops.summation(x, axes=1) / x.shape[1]
        devi = x + (-1) * ops.broadcast_to(mean.reshape((x.shape[0], 1)), x.shape)
        variance = ops.summation(devi * devi, axes=1) / x.shape[1]

        normalized = devi / ops.broadcast_to(ops.power_scalar(variance + self.eps, 0.5).reshape((x.shape[0], 1)), x.shape)
        return normalized * ops.broadcast_to(self.weight, normalized.shape) + ops.broadcast_to(self.bias, normalized.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            flattened = Flatten()(x)
            mask = init.randb(flattened.shape[0] * flattened.shape[1], p=(1 - self.p)).reshape(x.shape)
            return (mask * x) / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(fan_in=(self.kernel_size*self.kernel_size*self.in_channels), fan_out=(self.kernel_size*self.kernel_size*self.out_channels)
                                                      , shape=(self.kernel_size, self.kernel_size, self.in_channels, self.out_channels)), device=device)
        if bias:
            bound = 1/(in_channels*kernel_size**2)**0.5
            self.bias = Parameter(init.rand(self.out_channels, low=-bound, high=bound, device=device))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x = x.transpose((1, 2)).transpose((2, 3))
        x = ops.conv(x, self.weight, stride=self.stride, padding=(self.kernel_size-1)//2)
        if self.bias:
            x += self.bias.reshape((1,1,1,self.out_channels)).broadcast_to(x.shape)
        x = x.transpose((2, 3)).transpose((1, 2))
        return x
        ### END YOUR SOLUTION


class ConvBN(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size, stride=stride, bias=bias, device=device, dtype=dtype)
        self.bn = BatchNorm2d(dim=out_channels, device=device)
        self.f = ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.f(x)
        return x


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        
        k = 1/hidden_size
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-np.sqrt(k), high=np.sqrt(k), device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-np.sqrt(k), high=np.sqrt(k), device=device, dtype=dtype, requires_grad=True))
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, low=-np.sqrt(k), high=np.sqrt(k), device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(hidden_size, low=-np.sqrt(k), high=np.sqrt(k), device=device, dtype=dtype, requires_grad=True))
        else:
            self.bias_ih = None
            self.bias_hh = None

        if nonlinearity == "tanh":
            self.f = Tanh()
        elif nonlinearity == "relu":
            self.f = ReLU()
        else:
            raise ValueError("Non-linearity of RNNCell can only be tanh or relu!")
        
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        if h is None:
            h = init.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
        output = X @ self.W_ih + h @ self.W_hh
        if self.bias_ih and self.bias_hh:
            output += ops.broadcast_to((self.bias_ih + self.bias_hh).reshape((1, self.hidden_size)), (batch_size, self.hidden_size))
        return self.f(output)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        self.rnn_cells = [RNNCell(input_size=input_size, hidden_size=hidden_size, bias=bias, nonlinearity=nonlinearity, device=device, dtype=dtype)]
        for i in range(num_layers - 1):
            self.rnn_cells.append(RNNCell(input_size=hidden_size, hidden_size=hidden_size, bias=bias, nonlinearity=nonlinearity, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, batch_size, input_size = X.shape
        if h0 is None:
            h0 = init.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
        h0 = ops.split(h0, axis=0) # for iteration, Tensor not allow slicing

        hn = []
        ht = list(ops.split(X, axis=0))

        for i, rnn_cell in enumerate(self.rnn_cells):
            h = h0[i]
            for t, x in enumerate(ht):
                h = rnn_cell(x, h)
                ht[t] = h
    
            hn.append(h)
        return ops.stack(ht, axis=0), ops.stack(hn, axis=0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.device = device
        self.dtype = dtype

        k = 1 / hidden_size
        self.W_ih = Parameter(init.rand(input_size, 4 * hidden_size, low=-np.sqrt(k), high=np.sqrt(k), device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, 4 * hidden_size, low=-np.sqrt(k), high=np.sqrt(k), device=device, dtype=dtype, requires_grad=True))

        self.bias_ih = None
        self.bias_hh = None
        if bias:
            self.bias_ih = Parameter(init.rand(4 * hidden_size, low=-np.sqrt(k), high=np.sqrt(k), device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(4 * hidden_size, low=-np.sqrt(k), high=np.sqrt(k), device=device, dtype=dtype, requires_grad=True))

        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        batch_size, _ = X.shape
        if h is None:
            h = init.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype), init.zeros(batch_size, self.hidden_size, device=self.device, dtype=self.dtype)

        h0, c0 = h
        out = X @ self.W_ih + h0 @ self.W_hh
        if self.bias:
            out += self.bias_ih.reshape((1, 4 * self.hidden_size)).broadcast_to((batch_size, 4 * self.hidden_size))
            out += self.bias_hh.reshape((1, 4 * self.hidden_size)).broadcast_to((batch_size, 4 * self.hidden_size))

        assert out.shape[1] == 4 * self.hidden_size
        splited = list(ops.split(out, axis=1))
        stacked = [ops.stack(splited[slice(i * self.hidden_size, (i + 1) * self.hidden_size)], axis=1) for i in range(4)]

        i, f, g, o = stacked
        # print(i.shape, f.shape, g.shape, o.shape)
        i, f, g, o = self.sigmoid(i), self.sigmoid(f), self.tanh(g), self.sigmoid(o)
        c_out = f * c0 + i * g
        h_out = o * self.tanh(c_out)
        return h_out, c_out
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.device = device
        self.dtype = dtype

        self.lstm_cells = [LSTMCell(input_size, hidden_size, bias=bias, device=device, dtype=dtype)]
        self.lstm_cells += [LSTMCell(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype)] * (num_layers - 1)
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, batch_size, input_size = X.shape
        if h is None:
            h = init.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device, dtype=self.dtype), \
                init.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device, dtype=self.dtype)
            
        h0, c0 = h
        h0, c0 = list(ops.split(h0, axis=0)), list(ops.split(c0, axis=0))
        output, h_n, c_n = list(ops.split(X, axis=0)), [], []
        for l, lstm_cell in enumerate(self.lstm_cells):
            h, c = h0[l], c0[l]
            for t in range(seq_len):
                h, c = lstm_cell(output[t], (h, c))
                output[t] = h
            h_n.append(h)
            c_n.append(c)

        return ops.stack(output, axis=0), (ops.stack(h_n, axis=0), ops.stack(c_n, axis=0))
            
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, batch_size = x.shape
        one_hot = init.one_hot(self.num_embeddings, x, device=self.device, dtype=self.dtype)
        return (one_hot.reshape((seq_len * batch_size, self.num_embeddings)) @ self.weight).reshape((seq_len, batch_size, self.embedding_dim))
        ### END YOUR SOLUTION
