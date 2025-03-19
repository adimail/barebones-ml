import numpy as np


class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.i_weight = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.h_weight = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        self.h_bias = np.zeros((1, hidden_size))
        self.o_weight = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.o_bias = np.zeros((1, output_size))

    def forward(self, x, h_prev=None):
        seq_len = x.shape[0]
        if h_prev is None:
            h_prev = np.zeros((1, self.hidden_size))
        hiddens = np.zeros((seq_len, self.hidden_size))
        outputs = np.zeros((seq_len, self.output_size))
        h = h_prev
        for t in range(seq_len):
            xi = x[t].reshape(1, -1) @ self.i_weight
            xh = xi + h @ self.h_weight + self.h_bias
            h = np.tanh(xh)
            hiddens[t] = h[0]
            xo = h @ self.o_weight + self.o_bias
            outputs[t] = xo[0]
        return outputs, hiddens

    def backward(self, d_outputs, hiddens, x):
        seq_len = d_outputs.shape[0]
        d_i_weight = np.zeros_like(self.i_weight)
        d_h_weight = np.zeros_like(self.h_weight)
        d_h_bias = np.zeros_like(self.h_bias)
        d_o_weight = np.zeros_like(self.o_weight)
        d_o_bias = np.zeros_like(self.o_bias)
        next_d_h = np.zeros((1, self.hidden_size))
        for t in range(seq_len - 1, -1, -1):
            d_xo = d_outputs[t].reshape(1, -1)
            d_o_weight += hiddens[t].reshape(-1, 1) @ d_xo
            d_o_bias += d_xo
            d_h = d_xo @ self.o_weight.T
            d_h += next_d_h
            d_xh = d_h * (1 - hiddens[t] ** 2).reshape(1, -1)
            next_d_h = d_xh @ self.h_weight.T
            d_i_weight += x[t].reshape(-1, 1) @ d_xh
            if t > 0:
                d_h_weight += hiddens[t - 1].reshape(-1, 1) @ d_xh
            d_h_bias += d_xh
        return (d_i_weight, d_h_weight, d_h_bias, d_o_weight, d_o_bias)

    def update(self, gradients, learning_rate):
        d_i_weight, d_h_weight, d_h_bias, d_o_weight, d_o_bias = gradients
        self.i_weight -= learning_rate * d_i_weight
        self.h_weight -= learning_rate * d_h_weight
        self.h_bias -= learning_rate * d_h_bias
        self.o_weight -= learning_rate * d_o_weight
        self.o_bias -= learning_rate * d_o_bias


def create_sequences(x, y, seq_len):
    sequences = []
    targets = []
    for i in range(len(x) - seq_len):
        seq = x[i : i + seq_len]
        target = y[i + 1 : i + seq_len + 1]
        sequences.append(seq)
        targets.append(target)
    return sequences, targets


def mse(actual, predicted):
    return np.mean((actual - predicted) ** 2)


def mse_grad(actual, predicted):
    return predicted - actual
