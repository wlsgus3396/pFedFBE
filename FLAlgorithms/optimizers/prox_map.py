import torch
from FLAlgorithms.thirdparty.simplex_projection import euclidean_proj_l1ball
import numpy


def RLprox(regularizer, w, lamdaC0, learning_rate,l1const,l2const):
    if regularizer == "l1-constraint":
        
        proj_w = euclidean_proj_l1ball(w, l1const)

        w = proj_w










    if regularizer == "l2-constraint":
        scalar = max(torch.linalg.norm(w,2)/l2const, 1)
        w = w / scalar






    if regularizer == "l1":
        tmp = torch.abs(w) - lamdaC0*learning_rate
        w = (tmp > 0) * tmp * torch.sign(w)


    if regularizer == "l2":
        scalar = 1 - lamdaC0*learning_rate / torch.linalg.norm(w,2)
        w = (scalar > 0) * w

    if regularizer == "nuclear":
        ww = torch.reshape(w, (32, 32))

        u, s, v, = torch.linalg.svd(ww)
        tmp = torch.abs(s) - lamdaC0*learning_rate
        ss = (tmp > 0) * tmp * torch.sign(s)
        ww = u @ torch.diag(ss) @ torch.transpose(v, 0, 1)

        w = torch.reshape(ww, (32 * 32,1))
        w=numpy.squeeze(w)

        
    return w