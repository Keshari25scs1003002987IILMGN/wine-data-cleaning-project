"""
Reference CNN implementation in pure NumPy.
Used to actually execute and validate the network architecture described in
`cnn_tensorflow.py` inside an offline environment (no internet / no GPU).
Architecture: Conv2D(8,3x3) -> ReLU -> MaxPool(2x2) -> Flatten -> Dense(64) -> ReLU -> Dense(10) -> Softmax
"""
import numpy as np

np.random.seed(42)


def conv2d_forward(X, W, b):
    # X: (N, H, W), W: (F, kh, kw), b: (F,)
    N, H, Wd = X.shape
    F, kh, kw = W.shape
    oh, ow = H - kh + 1, Wd - kw + 1
    out = np.zeros((N, F, oh, ow))
    for f in range(F):
        for i in range(oh):
            for j in range(ow):
                patch = X[:, i:i + kh, j:j + kw]  # (N, kh, kw)
                out[:, f, i, j] = np.sum(patch * W[f], axis=(1, 2)) + b[f]
    return out


def conv2d_backward(dOut, X, W):
    N, H, Wd = X.shape
    F, kh, kw = W.shape
    oh, ow = dOut.shape[2], dOut.shape[3]
    dW = np.zeros_like(W)
    db = np.zeros(F)
    dX = np.zeros_like(X)
    for f in range(F):
        db[f] = np.sum(dOut[:, f, :, :])
        for i in range(oh):
            for j in range(ow):
                patch = X[:, i:i + kh, j:j + kw]  # (N, kh, kw)
                dW[f] += np.sum(patch * dOut[:, f, i, j][:, None, None], axis=0)
                dX[:, i:i + kh, j:j + kw] += W[f] * dOut[:, f, i, j][:, None, None]
    return dX, dW, db


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(float)


def maxpool_forward(X, size=2):
    N, F, H, W = X.shape
    oh, ow = H // size, W // size
    out = np.zeros((N, F, oh, ow))
    argmax = np.zeros((N, F, oh, ow, 2), dtype=int)
    for i in range(oh):
        for j in range(ow):
            window = X[:, :, i * size:i * size + size, j * size:j * size + size]
            flat = window.reshape(N, F, -1)
            idx = np.argmax(flat, axis=2)
            out[:, :, i, j] = np.max(flat, axis=2)
            argmax[:, :, i, j, 0] = idx // size
            argmax[:, :, i, j, 1] = idx % size
    return out, argmax


def maxpool_backward(dOut, X_shape, argmax, size=2):
    dX = np.zeros(X_shape)
    N, F, oh, ow = dOut.shape
    for i in range(oh):
        for j in range(ow):
            for n in range(N):
                for f in range(F):
                    di, dj = argmax[n, f, i, j]
                    dX[n, f, i * size + di, j * size + dj] += dOut[n, f, i, j]
    return dX


def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


class SimpleCNN:
    def __init__(self, n_filters=8, k=3, dense_hidden=64, n_classes=10, in_size=8):
        self.W_conv = np.random.randn(n_filters, k, k) * np.sqrt(2.0 / (k * k))
        self.b_conv = np.zeros(n_filters)
        conv_out = in_size - k + 1
        pool_out = conv_out // 2
        flat_dim = n_filters * pool_out * pool_out
        self.W1 = np.random.randn(flat_dim, dense_hidden) * np.sqrt(2.0 / flat_dim)
        self.b1 = np.zeros(dense_hidden)
        self.W2 = np.random.randn(dense_hidden, n_classes) * np.sqrt(2.0 / dense_hidden)
        self.b2 = np.zeros(n_classes)

    def forward(self, X):
        conv_out = conv2d_forward(X, self.W_conv, self.b_conv)
        relu_out = relu(conv_out)
        pool_out, argmax = maxpool_forward(relu_out, size=2)
        N = X.shape[0]
        flat = pool_out.reshape(N, -1)
        z1 = flat @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        probs = softmax(z2)
        cache = (X, conv_out, relu_out, pool_out, argmax, flat, z1, a1, z2)
        return probs, cache

    def backward(self, probs, y_onehot, cache, lr=0.01):
        X, conv_out, relu_out, pool_out, argmax, flat, z1, a1, z2 = cache
        N = X.shape[0]
        dz2 = (probs - y_onehot) / N
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_grad(z1)
        dW1 = flat.T @ dz1
        db1 = np.sum(dz1, axis=0)
        dflat = dz1 @ self.W1.T
        dpool = dflat.reshape(pool_out.shape)
        drelu_out = maxpool_backward(dpool, relu_out.shape, argmax, size=2)
        dconv_out = drelu_out * relu_grad(conv_out)
        _, dW_conv, db_conv = conv2d_backward(dconv_out, X, self.W_conv)

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W_conv -= lr * dW_conv
        self.b_conv -= lr * db_conv
