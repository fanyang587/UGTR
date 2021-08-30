"""
This is the implementation of following paper:
https://arxiv.org/pdf/1802.05591.pdf
This implementation is based on following code:
https://github.com/Wizaron/instance-segmentation-pytorch
"""
from torch.nn.modules.loss import  _Loss
from torch.autograd import Variable
import torch


class DiscriminativeLoss(_Loss):

    def __init__(self, delta_var=0.5, delta_dist=1.5,
                 norm=2, alpha=1.0, beta=1.0, gamma=0.001,
                 usegpu=True, size_average=True):
        super(DiscriminativeLoss, self).__init__(size_average)
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.usegpu = usegpu
        assert self.norm in [1, 2]

    def forward(self, input):
        #_assert_no_grad(target)
        return self._discriminative_loss(input)

    def _discriminative_loss(self, input):
        n_clusters, bs, n_features = input.size()
        #max_n_clusters = target.size(1)

        input = input.contiguous().permute(1, 2, 0) #.view(bs, n_features, height * width)
        #target = target.contiguous().view(bs, max_n_clusters, height * width)

        c_means = input# self._cluster_means(input, target, n_clusters)
        #l_var = self._variance_term(input, target, c_means, n_clusters)
        l_dist = self._distance_term(c_means)
        l_reg = self._regularization_term(c_means)

        loss = self.beta * l_dist + self.gamma * l_reg

        return loss


    def _distance_term(self, c_means):
        bs, n_features, max_n_clusters = c_means.size()

        n_clusters = [max_n_clusters]*bs

        dist_term = 0
        for i in range(bs):
            #if n_clusters[i] <= 1:
            #    continue

            # n_features, n_clusters
            mean_sample = c_means[i, :, :n_clusters[i]]

            # n_features, n_clusters, n_clusters
            means_a = mean_sample.unsqueeze(2).expand(n_features, n_clusters[i], n_clusters[i])
            means_b = means_a.permute(0, 2, 1)
            diff = means_a - means_b

            margin = 2 * self.delta_dist * (1.0 - torch.eye(n_clusters[i]))
            margin = Variable(margin)
            if self.usegpu:
                margin = margin.cuda()
            c_dist = torch.sum(torch.clamp(margin - torch.norm(diff, self.norm, 0), min=0) ** 2)
            dist_term += c_dist / (2 * n_clusters[i] * (n_clusters[i] - 1))
        dist_term /= bs

        return dist_term

    def _regularization_term(self, c_means):
        bs, n_features, max_n_clusters = c_means.size()

        reg_term = 0
        for i in range(bs):
            # n_features, n_clusters
            mean_sample = c_means[i, :, :]
            reg_term += torch.mean(torch.norm(mean_sample, self.norm, 0))
        reg_term /= bs

        return reg_term
