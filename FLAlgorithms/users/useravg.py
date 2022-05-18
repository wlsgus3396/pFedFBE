import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User
from FLAlgorithms.optimizers.fedoptimizer import ManualSGD

# Implementation for FedAvg clients

class UserAVG(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, lamda,
                 local_epochs, optimizer,regularizer,lamdaCO,modeltype,l1const,l2const):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, lamda,
                         local_epochs,regularizer,lamdaCO,modeltype,l1const,l2const)
        if(self.modeltype=='Lasso' or self.modeltype=='Matrix'):
            self.loss=nn.MSELoss()
        else:
            if(model[1] == "Mclr_CrossEntropy"):
                self.loss = nn.CrossEntropyLoss()
            else:
                self.loss = nn.NLLLoss()


        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        #self.optimizer = ManualSGD(self.model.parameters(), lr=self.learning_rate)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]



    def set_parameters_avg_personal(self, model):
        for old_param, new_param in zip(self.persionalized_model_bar, model.parameters()):
            old_param.data = new_param.data.clone()
        #self.local_weight_updated = copy.deepcopy(self.optimizer.param_groups[0]['params'])
    def train(self, epochs):
        
        LOSS = 0
        self.model.train()
        
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()

            output = self.model(X)
            if self.modeltype=='Lasso' or self.modeltype=='Matrix':
                output=torch.squeeze(output)
                y=y.float()
            loss =self.loss(output, y)+self.lamdaCO*self.RL(self.model)


            
            loss.backward()
            self.optimizer.step()

        self.clone_model_paramenter(self.model.parameters(), self.local_model)
        self.clone_model_paramenter(self.local_model, self.persionalized_model_bar)
        return LOSS

