import numpy as np
import torch
import torch.nn as nn


def pca_np(x: np.ndarray, k=2):
    '''
    :param x: tensor of NxM
    :param k: k dim to preserve
    :return:
    '''
    shift_x = x - x.mean(0)
    U, S, V = np.linalg.svd(shift_x)
    C = U[:, :k] * S[:k]
    return C


class PCA(nn.Module):
    def __init__(self, n_components=2):
        super(PCA, self).__init__()
        self.V = None
        self.k = n_components
        self.V_inv = None
        self.mean = None

    def fit(self, x):
        self.mean = x.mean(0)
        shift_x = x - self.mean
        U, S, V = torch.svd(shift_x)  # U是
        self.V = V[:, :self.k]
        self.V_inv = self.V.pinverse()

    def fit_transform(self, x):
        self.fit(x)
        C = self.transform(x)
        return C

    def forward(self, x):
        return self.decompose(x)

    def transform(self, x):
        return torch.mm(x - self.mean, self.V[:, :self.k])

    def inverse_transform(self, c):
        return torch.mm(c, self.V_inv) + self.mean


if __name__ == "__main__":
    x = torch.randn(4, 3).double()
    k = 2
    pca = PCA(n_components=k)
    newX = pca.fit_transform(x)
    j = pca.inverse_transform(newX)
    print((j - x).abs().max().item())

    from sklearn.decomposition import PCA as PCA_SK

    x = x.numpy().astype(np.float)
    pca_sk = PCA_SK(n_components=k)
    newX = pca_sk.fit_transform(x)
    j = pca_sk.inverse_transform(newX)
    print(np.abs((j - x).max()))
