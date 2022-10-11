import numpy as np
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import torch

def _parallel_compute_distance(X, cluster):
    n_samples = X.shape[0]
    dis_mat = torch.zeros((n_samples, 1))
    for i in range(n_samples):
        dis_mat[i] += torch.sqrt(torch.sum((X[i] - cluster) ** 2, axis=0))
    return dis_mat


class batch_KMeans(object):

    def __init__(self, args):
        self.args = args
        self.n_features = args.ae_latent_dim
        self.n_clusters = args.n_clusters
        self.device = args.device
        self.clusters = torch.zeros((self.n_clusters, self.n_features)).to(self.device)
        self.count = 100 * torch.ones((self.n_clusters)).to(self.device)  # serve as learning rate
        self.n_jobs = args.n_jobs

    def _compute_dist(self, X):
        # dis_mat = Parallel(n_jobs=self.n_jobs)(
        #     delayed(_parallel_compute_distance)(X, self.clusters[i])
        #     for i in range(self.n_clusters))
        # dis_mat = torch.hstack(dis_mat)

        dis_mat = ((X[:, None, :] - self.clusters[None, :, :])**2).sum(-1)
        return dis_mat

    def init_cluster(self, X, indices=None):
        """ Generate initial clusters using sklearn.Kmeans """
        model = KMeans(n_clusters=self.n_clusters,
                       n_init=20)
        model.fit(X)
        self.clusters = torch.Tensor(model.cluster_centers_).to(self.device)  # copy clusters

    def update_cluster(self, X, cluster_idx):
        """ Update clusters in Kmeans on a batch of data """
        n_samples = X.shape[0]
        for i in range(n_samples):
            self.count[cluster_idx] += 1
            eta = 1.0 / self.count[cluster_idx]
            updated_cluster = ((1 - eta) * self.clusters[cluster_idx] +
                               eta * X[i])
            self.clusters[cluster_idx] = updated_cluster

    def update_assign(self, X):
        """ Assign samples in `X` to clusters """
        dis_mat = self._compute_dist(X)

        return torch.argmin(dis_mat, dim=1)
