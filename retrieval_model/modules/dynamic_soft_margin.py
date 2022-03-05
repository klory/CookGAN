import torch
import torch.nn as nn
from .hardnet_loss import (
    compute_distance_matrix_hamming,
    compute_distance_matrix_unit_l2,
    find_hard_negatives,
)


class DynamicSoftMarginLoss(nn.Module):
    def __init__(self, is_binary=False, momentum=0.01, max_dist=None, nbins=512):
        """
        is_binary: true if learning binary descriptor
        momentum: weight assigned to the histogram computed from the current batch
        max_dist: maximum possible distance in the feature space
        nbins: number of bins to discretize the PDF
        """
        super(DynamicSoftMarginLoss, self).__init__()
        self._is_binary = is_binary

        if max_dist is None:
            # max_dist = 256 if self._is_binary else 2.0
            max_dist = 2.0

        self._momentum = momentum
        self._max_val = max_dist
        self._min_val = -max_dist
        self.register_buffer("histogram", torch.ones(nbins))

        self._stats_initialized = False
        self.current_step = None

    def _compute_distances(self, x):
        if self._is_binary:
            return self._compute_hamming_distances(x)
        else:
            return self._compute_l2_distances(x)

    def _compute_l2_distances(self, x):
        cnt = x.size(0) // 2
        a = x[:cnt, :]
        p = x[cnt:, :]
        # import pdb; pdb.set_trace()
        dmat = compute_distance_matrix_unit_l2(a, p)
        return find_hard_negatives(dmat, output_index=False, empirical_thresh=0.008)

    def _compute_hamming_distances(self, x):
        cnt = x.size(0) // 2
        ndims = x.size(1)
        a = x[:cnt, :]
        p = x[cnt:, :]

        dmat = compute_distance_matrix_hamming(
            (a > 0).float() * 2.0 - 1.0, (p > 0).float() * 2.0 - 1.0
        )
        a_idx, p_idx, n_idx = find_hard_negatives(
            dmat, output_index=True, empirical_thresh=2
        )

        # differentiable Hamming distance
        a = x[a_idx, :]
        p = x[p_idx, :]
        n = x[n_idx, :]

        pos_dist = (1.0 - a * p).sum(dim=1) / ndims
        neg_dist = (1.0 - a * n).sum(dim=1) / ndims

        # non-differentiable Hamming distance
        a_b = (a > 0).float() * 2.0 - 1.0
        p_b = (p > 0).float() * 2.0 - 1.0
        n_b = (n > 0).float() * 2.0 - 1.0

        pos_dist_b = (1.0 - a_b * p_b).sum(dim=1) / ndims
        neg_dist_b = (1.0 - a_b * n_b).sum(dim=1) / ndims

        return pos_dist, neg_dist, pos_dist_b, neg_dist_b

    def _compute_histogram(self, x, momentum):
        """
        update the histogram using the current batch
        """
        num_bins = self.histogram.size(0)
        x_detached = x.detach()
        self.bin_width = (self._max_val - self._min_val) / (num_bins - 1)
        lo = torch.floor((x_detached - self._min_val) / self.bin_width).long()
        hi = (lo + 1).clamp(min=0, max=num_bins - 1)
        hist = x.new_zeros(num_bins)
        alpha = (
            1.0
            - (x_detached - self._min_val - lo.float() * self.bin_width)
            / self.bin_width
        )
        hist.index_add_(0, lo, alpha)
        hist.index_add_(0, hi, 1.0 - alpha)
        hist = hist / (hist.sum() + 1e-6)
        self.histogram = (1.0 - momentum) * self.histogram + momentum * hist

    def _compute_stats(self, pos_dist, neg_dist):
        hist_val = pos_dist - neg_dist
        if self._stats_initialized:
            self._compute_histogram(hist_val, self._momentum)
        else:
            self._compute_histogram(hist_val, 1.0)
            self._stats_initialized = True

    def forward(self, x):
        distances = self._compute_distances(x)
        if not self._is_binary:
            pos_dist, neg_dist = distances
            self._compute_stats(pos_dist, neg_dist)
            hist_var = pos_dist - neg_dist
        else:
            pos_dist, neg_dist, pos_dist_b, neg_dist_b = distances
            self._compute_stats(pos_dist_b, neg_dist_b)
            hist_var = pos_dist_b - neg_dist_b

        PDF = self.histogram / self.histogram.sum()
        CDF = PDF.cumsum(0)

        # lookup weight from the CDF
        bin_idx = torch.floor((hist_var - self._min_val) / self.bin_width).long()
        weight = CDF[bin_idx]

        loss = -(neg_dist * weight).mean() + (pos_dist * weight).mean()
        return loss
