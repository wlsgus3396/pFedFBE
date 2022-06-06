import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.optimizers.fedoptimizer import FedDualOptimizer
from FLAlgorithms.users.userbase import User
from FLAlgorithms.optimizers.prox_map import RLprox
import copy

# Implementation for pFeMe clients

class UserFedDual(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate,beta,lamda,
                 local_epochs, optimizer, K, personal_learning_rate,regularizer,lamdaCO,modeltype,l1const,l2const):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, lamda,
                         local_epochs,regularizer,lamdaCO,modeltype,l1const,l2const)
        if(self.modeltype=='Lasso' or self.modeltype=='Matrix'):
            self.loss=nn.MSELoss()
        else:
            if(model[1] == "Mclr_CrossEntropy"):
                self.loss = nn.CrossEntropyLoss()
            else:
                self.loss = nn.NLLLoss()


        self.K = K
        self.personal_learning_rate = personal_learning_rate
        self.optimizer = FedDualOptimizer(self.model.parameters(), lr=self.personal_learning_rate, lamda=self.lamda)
        self.local_weight_dual = copy.deepcopy(list(self.model.parameters()))



    def get_local_weight_dual(self):
        for param in self.local_weight_dual:
            param.detach()
        return self.local_weight_dual



    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]






    def set_parameters_FedDual(self, model, local_weight_dual):#################################################### It works?
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()
        for old_param_dual, new_param_dual in zip(self.local_weight_dual, local_weight_dual):
            old_param_dual.data = new_param_dual.data.clone()



    def set_parameters_dual_personal(self, model):
        for old_param, new_param in zip(self.persionalized_model_bar, model.parameters()):
            old_param.data = new_param.data.clone()
        #self.local_weight_updated = copy.deepcopy(self.optimizer.param_groups[0]['params'])




    def train(self, epochs, glob_iter):
        LOSS = 0
        self.model.train()
        persionalized_model = copy.deepcopy(list(self.model.parameters()))
        # self.clone_model_paramenter(self.model.parameters(), persionalized_model)
        for epoch in range(1, self.local_epochs + 1):  # local update
            
            self.model.train()
            X, y = self.get_next_train_batch()

            Q=[]
            for q in self.local_weight_dual:
                Q.append(torch.flatten(q.data))
            
            if self.modeltype!="Lasso" and self.modeltype!="Matrix":
                for ii in range(1,len(Q)):
                    Q[0]=torch.cat((Q[0],Q[ii]))

            
            eta_tilde = self.learning_rate * (glob_iter + 1) * self.local_epochs + self.learning_rate * (epoch-1)

            P=RLprox(self.regularizer, Q[0], self.lamdaCO, eta_tilde, self.l1const,self.l2const)

            ii=0
            for q in self.model.parameters():
                if self.modeltype=="Lasso" or self.modeltype=="Matrix": 
                    if len(torch.flatten(q.data))==1024:
                        q.data=P[ii:ii+len(torch.flatten(q.data))].reshape(q.data.size())
                        ii+=len(torch.flatten(q.data))
                else:
                    q.data=P[ii:ii+len(torch.flatten(q.data))].reshape(q.data.size())    
                    ii+=len(torch.flatten(q.data))



            # K = 30 # K is number of personalized steps
            self.optimizer.zero_grad()
            output = self.model(X)
            if self.modeltype=='Lasso' or self.modeltype=='Matrix':
                output=torch.squeeze(output)
                y = y.float()
            loss = self.loss(output, y)
            loss.backward()
            local_weight_dual, _ = self.optimizer.step(self.regularizer, self.local_weight_dual, epoch, self.local_epochs, glob_iter,self.lamdaCO,self.modeltype,self.l1const,self.l2const)
            self.clone_model_paramenter(local_weight_dual, self.local_weight_dual)


        return LOSS