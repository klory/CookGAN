import torch
import torch.nn as nn


def compute_distance_matrix_unit_l2(a, b, eps=1e-6):
    """
    computes pairwise Euclidean distance and return a N x N matrix
    """

    dmat = torch.matmul(a, torch.transpose(b, 0, 1))
    dmat = ((1.0 - dmat + eps) * 2.0).pow(0.5)
    return dmat


def compute_distance_matrix_hamming(a, b):
    """
    computes pairwise Hamming distance and return a N x N matrix
    """

    dims = a.size(1)
    dmat = torch.matmul(a, torch.transpose(b, 0, 1))
    dmat = (dims - dmat) * 0.5
    return dmat


def find_hard_negatives(dmat, output_index=True, empirical_thresh=0.0):
    """
    a = A * P'
    A: N * ndim
    P: N * ndim

    a1p1 a1p2 a1p3 a1p4 ...
    a2p1 a2p2 a2p3 a2p4 ...
    a3p1 a3p2 a3p3 a3p4 ...
    a4p1 a4p2 a4p3 a4p4 ...
    ...  ...  ...  ...
    """

    cnt = dmat.size(0)

    if not output_index:
        pos = dmat.diag()

    dmat = dmat + torch.eye(cnt).to(dmat.device) * 99999  # filter diagonal
    # dmat[dmat < empirical_thresh] = 99999  # filter outliers in brown dataset
    min_a, min_a_idx = torch.min(dmat, dim=0)
    min_p, min_p_idx = torch.min(dmat, dim=1)

    if not output_index:
        neg = torch.min(min_a, min_p)
        # import pdb; pdb.set_trace()
        return pos, neg

    mask = min_a < min_p
    a_idx = torch.cat(
        (mask.nonzero().view(-1) + cnt, (~mask).nonzero().view(-1))
    )  # use p as anchor
    p_idx = torch.cat(
        (mask.nonzero().view(-1), (~mask).nonzero().view(-1) + cnt)
    )  # use a as anchor
    n_idx = torch.cat((min_a_idx[mask], min_p_idx[~mask] + cnt))
    return a_idx, p_idx, n_idx


def approx_hamming_distance(a, p):
    return (1.0 - a * p).sum(dim=1) * 0.5


class HardNetLoss(nn.Module):
    def __init__(self, margin, is_binary):
        super().__init__()
        self._margin = margin
        self._is_binary = is_binary

    def _forward_float(self, x):
        cnt = x.size(0) // 2
        a = x[:cnt, :]
        p = x[cnt:, :]

        dmat = compute_distance_matrix_unit_l2(a, p)
        pos, neg = find_hard_negatives(dmat, output_index=False, empirical_thresh=0.008)
        return (self._margin - neg + pos).clamp(0).mean()

    def _forward_binary(self, x):
        cnt = x.size(0) // 2
        ndim = x.size(1)
        a = x[:cnt, :]
        p = x[cnt:, :]

        dmat = compute_distance_matrix_hamming(
            (a > 0).float() * 2.0 - 1.0, (p > 0).float() * 2.0 - 1.0
        )
        a_idx, p_idx, n_idx = find_hard_negatives(
            dmat, output_index=True, empirical_thresh=2
        )

        a = x[a_idx, :]
        p = x[p_idx, :]
        n = x[n_idx, :]

        pos_dist = approx_hamming_distance(a, p)
        neg_dist = approx_hamming_distance(a, n)

        pos_dist = pos_dist / ndim
        neg_dist = neg_dist / ndim

        return (self._margin - neg_dist + pos_dist).clamp(0).mean()

    def forward(self, x):
        if self._is_binary:
            return self._forward_binary(x)
        return self._forward_float(x)
