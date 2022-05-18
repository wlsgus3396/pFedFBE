from torch.optim import Optimizer
import torch
from FLAlgorithms.optimizers.prox_map import RLprox
import copy
class MySGD(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(MySGD, self).__init__(params, defaults)

    def step(self, closure=None, beta = 0):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            # print(group)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if(beta != 0):
                    p.data.add_(-beta, d_p)
                else:     
                    p.data.add_(-group['lr'], d_p)
        return loss

class ManualSGD(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(ManualSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            # print(group)
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data -= group['lr'] * p.grad.data
        return group['params'], loss


class FEDLOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, server_grads=None, pre_grads=None, eta=0.1):
        self.server_grads = server_grads
        self.pre_grads = pre_grads
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, eta=eta)
        super(FEDLOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        for group in self.param_groups:
            i = 0
            for p in group['params']:
                p.data = p.data - group['lr'] * \
                         (p.grad.data + group['eta'] * self.server_grads[i] - self.pre_grads[i])
                # p.data.add_(-group['lr'], p.grad.data)
                i += 1
        return loss

class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1 , mu = 0.001):
        #self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu = mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)
    
    def step(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip( group['params'], weight_update):
                p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu']*p.data)
        return  group['params'], loss
    
    def update_param(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip( group['params'], weight_update):
                p.data = localweight.data
        #return  p.data
        return  group['params']





class pFedFBEOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1 , mu = 0.001):
        #self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu = mu)
        super(pFedFBEOptimizer, self).__init__(params, defaults)
        self.params=params
    def step(self, persionalized_model, regularizer, lamda, learning_rate, lamdaCO, modeltype='DNN',l1const=0,l2const=0, closure=None):
        loss = None
        if closure is not None:
            loss = closure

        
        for group in self.param_groups:
            for p,q in zip(group['params'], persionalized_model):
                q.data =p.data - 1/lamda*p.grad.data


        Q=[]
        for q in persionalized_model:
            Q.append(torch.flatten(q.data))
        
        if modeltype!="Lasso" and modeltype!="Matrix":
            for ii in range(1,len(Q)):
                Q[0]=torch.cat((Q[0],Q[ii]))

        P=RLprox(regularizer, Q[0], lamdaCO, 1/lamda, l1const,l2const)

        ii=0
        for q in persionalized_model:
            if modeltype=="Lasso" or modeltype=="Matrix": 
                if len(torch.flatten(q.data))==1024:
                    q.data=P[ii:ii+len(torch.flatten(q.data))].reshape(q.data.size())
                    ii+=len(torch.flatten(q.data))
            else:
                q.data=P[ii:ii+len(torch.flatten(q.data))].reshape(q.data.size())    
                ii+=len(torch.flatten(q.data))

        for group in self.param_groups:
            for p,q in zip(group['params'], persionalized_model):
                d = lamda*(p.data - q.data)
                p.data = p.data - learning_rate * d
        return persionalized_model, group['params']
    
    def update_param(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip( group['params'], weight_update):
                p.data = localweight.data
        #return  p.data
        return  group['params']








class FedmirrorOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        # self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        
        super(FedmirrorOptimizer, self).__init__(params, defaults)
        
    def step(self, regularizer, lamdaCO=1, modeltype='DNN',l1const=0,l2const=0, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        # weight_update = local_weight_updated.copy()
        Q=[]
        for group in self.param_groups:
            for p in group['params']:
                p.data = p.data - group['lr'] * p.grad.data
                Q.append(torch.flatten(p.data))
        
        if modeltype!="Lasso" and modeltype!="Matrix":
            for ii in range(1,len(Q)):
                Q[0]=torch.cat((Q[0],Q[ii]))

        P=RLprox(regularizer, Q[0], lamdaCO, group['lr'], l1const,l2const)




        ii=0
        for group in self.param_groups:
            for p in group['params']:
                if modeltype=="Lasso" or modeltype=="Matrix": 
                    if len(torch.flatten(p.data))==1024:
                        p.data=P[ii:ii+len(torch.flatten(p.data))].reshape(p.data.size())
                        ii+=len(torch.flatten(p.data))
                else:
                    p.data=P[ii:ii+len(torch.flatten(p.data))].reshape(p.data.size())    
                    ii+=len(torch.flatten(p.data))
        return group['params'], loss

    def update_param(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                p.data = localweight.data
        # return  p.data
        return group['params']


class FedDualOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        # self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(FedDualOptimizer, self).__init__(params, defaults)
        self.params=params
    def step(self, regularizer, local_weight_dual, epoch, local_epochs, glob_iter, lamdaCO=1,  modeltype='DNN',l1const=0,l2const=0, closure=None):
        loss = None

        if closure is not None:
            loss = closure
        for group in self.param_groups:
            for p, p_dual in zip(group['params'], local_weight_dual):
                p_dual.data = p_dual.data - group['lr'] * p.grad.data


        return local_weight_dual, loss



    def update_param(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                p.data = localweight.data
        # return  p.data
        return group['params']





class pFeddittoOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1 , mu = 0.001):
        #self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu = mu)
        super(pFeddittoOptimizer, self).__init__(params, defaults)
    
    def step(self, global_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = global_weight_updated.copy()
        for group in self.param_groups:
            for p, globalweight in zip( group['params'], weight_update):
                p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (p.data - globalweight.data))
        return  group['params'], loss
    
    def update_param(self, global_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = global_weight_updated.copy()
        for group in self.param_groups:
            for p, globalweight in zip( group['params'], weight_update):
                p.data = globalweight.data
        #return  p.data
        return  group['params']



class FedproxOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1 , mu = 0.001):
        #self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu = mu)
        super(FedproxOptimizer, self).__init__(params, defaults)
    
    def step(self, global_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = global_weight_updated.copy()
        for group in self.param_groups:
            for p, globalweight in zip( group['params'], weight_update):
                p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (p.data - globalweight.data))
        return  group['params'], loss
    
    def update_param(self, global_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = global_weight_updated.copy()
        for group in self.param_groups:
            for p, globalweight in zip( group['params'], weight_update):
                p.data = globalweight.data
        #return  p.data
        return  group['params']



class SCAFFOLD(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1 , mu = 0.001):
        #self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu = mu)
        super(SCAFFOLD, self).__init__(params, defaults)
    
    def step(self, server_controls, client_controls, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        for group in self.param_groups:
            for p, c, ci  in zip( group['params'], server_controls, client_controls):
                if p.grad is None:
                    continue
                p.data = p.data - group['lr'] * (p.grad.data + c.data-ci.data)
        return  group['params'], loss
    
    def update_param(self, global_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = global_weight_updated.copy()
        for group in self.param_groups:
            for p, globalweight in zip( group['params'], weight_update):
                p.data = globalweight.data
        #return  p.data
        return  group['params']


class FedDyn(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1 , mu = 0.001):
        #self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu = mu)
        super(FedDyn, self).__init__(params, defaults)
    
    def step(self, local_model, client_controls, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        for group in self.param_groups:
            for p, c, ci  in zip( group['params'], local_model, client_controls):
                if p.grad is None:
                    continue
                p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (p.data - c.data)-ci.data)
        return  group['params'], loss
    
    def update_param(self, global_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = global_weight_updated.copy()
        for group in self.param_groups:
            for p, globalweight in zip( group['params'], weight_update):
                p.data = globalweight.data
        #return  p.data
        return  group['params']



        
class APFLOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(APFLOptimizer, self).__init__(params, defaults)

    def step(self, closure=None, beta = 1, n_k = 1):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            # print(group)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = beta  * n_k * p.grad.data
                p.data.add_(-group['lr'], d_p)
        return loss
