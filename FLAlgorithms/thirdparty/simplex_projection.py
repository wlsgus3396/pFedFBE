""" Module to compute projections on the positive simplex or the L1-ball
 
A positive simplex is a set X = { \mathbf{x} | \sum_i x_i = s, x_i \geq 0 }
 
The (unit) L1-ball is the set X = { \mathbf{x} | || x ||_1 \leq 1 }
 
Adrien Gaidon - INRIA - 2011
URL: https://github.com/dongwookim-ml/python-topic-model/blob/450c901bc08842894a65931aed94523690f9fc90/ptm/simplex_projection.py
"""

import numpy as np
import torch


def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
 
    Solves the optimisation problem (using the algorithm from [1]):
 
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
 
    s: int, optional, default: 1,
       radius of the simplex
 
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
 
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
 
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    # if tf.reduce_sum(v) == s and tf.reduce_all(v >= 0):
    #     # best projection: itself!
    #     return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = torch.sort(v, descending=True)[0]
    cssv = torch.cumsum(u, dim=0)
    # get the number of > 0 components of the optimal solution
    tmp = torch.where(u * torch.arange(1, n + 1) > (cssv - s), True, False)
    rho = torch.nonzero(tmp)[-1][0]
    # compute the Lagrange multiplier associated to the simplex constraint
    rhotmp = rho.type(torch.float32)
    theta = (cssv[rho] - s) / (rhotmp + 1.0)
    # compute the projection by thresholding v using theta
    w = torch.clamp(v-theta, min=0, max=np.inf)
    return w


def euclidean_proj_l1ball(v, s=1):
    """ Compute the Euclidean projection on a L1-ball
 
    Solves the optimisation problem (using the algorithm from [1]):
 
        min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
 
    s: int, optional, default: 1,
       radius of the L1-ball
 
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s
 
    Notes
    -----
    Solves the problem by a reduction to the positive simplex case
 
    See also
    --------
    euclidean_proj_simplex
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = torch.abs(v)
    # check if v is already a solution

    w = torch.where(torch.sum(u) <= s, v, euclidean_proj_simplex(u, s=s) * torch.sign(v))
    
    return w