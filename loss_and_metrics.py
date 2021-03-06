import torch
import numpy as np
from torch.autograd import Variable
from functools import partial
import torch.nn.functional as F


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask_k = torch.sum(mask, dim=0, keepdim=True)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    diffs = preds - labels

    loss = diffs * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    loss = loss ** 2

    mse_loss = loss / mask_k.expand_as(loss)
    mse_loss = torch.sum(mse_loss) / mask.shape[1]

    return mse_loss


def masked_simse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask_k = torch.sum(mask, dim=0, keepdim=True)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    diffs = preds - labels

    loss = diffs * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    loss = loss ** 2

    mse_loss = loss / mask_k.expand_as(loss)
    mse_loss = torch.sum(mse_loss) / mask.shape[1]

    diffs = diffs * mask
    diffs = torch.where(torch.isnan(diffs), torch.zeros_like(diffs), diffs)
    penalty = torch.sum(torch.square(torch.sum(diffs, dim=0, keepdim=True)) / torch.square(mask_k)) / mask.shape[1]

    return mse_loss - 0.1 * penalty


def cov(m, rowvar=False):
    """Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    """
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


def pairwise_distance(x, y):
    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')

    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)

    return output


def gaussian_kernel_matrix(x, y, sigmas):
    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / (2. * sigmas)
    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)
    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))

    return cost


def mmd_loss(source_features, target_features, device):
    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    gaussian_kernel = partial(
        gaussian_kernel_matrix, sigmas=Variable(torch.Tensor(sigmas), requires_grad=False).to(device)
    )

    loss_value = maximum_mean_discrepancy(source_features, target_features, kernel=gaussian_kernel)
    loss_value = loss_value

    return loss_value


def compute_cosine_distances_matrix(x, y):
    normalize_x = F.normalize(x, p=2, dim=1)
    normalize_y = F.normalize(y, p=2, dim=1)
    sim_matrix = torch.matmul(normalize_x, normalize_y.transpose(0, 1))
    #sim_matrix[torch.isnan(sim_matrix)] = 0.0
    #sim_matrix = torch.nan_to_num(sim_matrix, nan=0.0)
    return sim_matrix


def contrastive_loss(y_true, y_pred, device):
    sim_matrix = compute_cosine_distances_matrix(y_true, y_pred)
    denominator = torch.sum(torch.mul(torch.exp(sim_matrix), -1 * (
            torch.eye(n=sim_matrix.shape[0], dtype=torch.float32).to(device) - 1)), dim=0) + torch.sum(torch.mul(torch.exp(sim_matrix), -1 * (
            torch.eye(n=sim_matrix.shape[0], dtype=torch.float32).to(device) - 1)), dim=1)
    nominator = torch.sum(
        torch.mul(torch.exp(sim_matrix), torch.eye(n=sim_matrix.shape[0], dtype=torch.float32).to(device)), dim=0)
    # nominator = torch.nan_to_num(nominator, nan=0.0)
    # # denominator = torch.nan_to_num(denominator, nan=0.0)
    # if torch.isnan(denominator).any():
    #     print(torch.isnan(y_true).any())
    #     print(torch.isnan(y_pred).any())
    #     print(torch.isnan(sim_matrix).any())
    #     print(denominator)
    #     print(nominator)
    return -torch.mean(torch.log(nominator) - torch.log(denominator))
