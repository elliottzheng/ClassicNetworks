import itertools
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor(),
                                ])
data_train = datasets.MNIST(root="../data/mnist",
                            transform=transform,
                            train=True,
                            download=True)

mb_size = 64  # batch size
Z_dim = 100  #
X_dim = 28 * 28
y_dim = 28 * 28
h_dim = 128
c = 0
lr = 1e-3


# =============================== Q(z|X) ======================================


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(X_dim, h_dim),
            nn.ReLU()
        )
        self.fc_mean = nn.Linear(h_dim, Z_dim)
        self.fc_var = nn.Linear(h_dim, Z_dim)  # 直接预测的是 logvar

    def forward(self, x):
        feat = self.fc1(x)
        z_mu = self.fc_mean(feat)
        z_var = self.fc_var(feat)
        return z_mu, z_var


def sample_z(mu, log_var):
    eps = torch.randn(len(mu), Z_dim)  # eps 是从标准正态分布采样的，然后再做一下缩放 和平移
    return mu + torch.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(Z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, X_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc1(x)
        return x


encoder = Encoder()  # 预测 mean和 var
decoder = Decoder()  # 将采样的向量恢复成X
# =============================== TRAINING ====================================

optimizer = optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=lr)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=mb_size,
                                                shuffle=True)
total_it = 0
for epoch in range(10):
    for i, (X, _) in enumerate(data_loader_train):
        X = X.view(-1, 28 * 28)
        # X.requires_grad_(True)

        # Forward
        z_mu, z_var = encoder(X)
        z = sample_z(z_mu, z_var)
        X_sample = decoder(z)

        # Loss
        recon_loss = F.binary_cross_entropy(X_sample, X, size_average=False) / len(X)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var, 1))
        loss = recon_loss + kl_loss

        # Backward
        loss.backward()

        # Update
        optimizer.step()

        # Housekeeping
        optimizer.zero_grad()
        total_it += 1
        # Print and plot every now and then
        print('Iter-{}; Loss: {:.4}'.format(total_it, loss.item()))

        if (total_it + 1) % 100 == 0:
            samples = decoder(z).detach().numpy()[:16]

            fig = plt.figure(figsize=(4, 4))
            gs = gridspec.GridSpec(4, 4)
            gs.update(wspace=0.05, hspace=0.05)

            for j, sample in enumerate(samples):
                ax = plt.subplot(gs[j])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

            if not os.path.exists('out/'):
                os.makedirs('out/')

            plt.savefig('out/{}.png'.format(str(c).zfill(3)), bbox_inches='tight')
            c += 1
            plt.close(fig)
