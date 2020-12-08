import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Net(nn.Module):
    """
    L2-Net: Deep Learning of Discriminative Patch Descriptor in Euclidean Space
    """

    def __init__(self, out_dim=128, binary=False, dropout_rate=0.1):
        super().__init__()
        self._binary = binary

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(128, out_dim, kernel_size=8, bias=False),
            nn.BatchNorm2d(out_dim, affine=False),
        )

        if self._binary:
            self.binarizer = nn.Tanh()

        self.features.apply(weights_init)

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (
            x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        ) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        input = self.input_norm(input)
        x = self.features(input)
        x = x.view(x.size(0), -1)
        if self._binary:
            return self.binarizer(x)
        else:
            return F.normalize(x, p=2, dim=1)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, gain=0.6)
        try:
            nn.init.constant_(m.bias.data, 0.01)
        except:
            pass
    return
