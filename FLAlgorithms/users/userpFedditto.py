import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
import json
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User
from FLAlgorithms.optimizers.fedoptimizer import pFeddittoOptimizer
# Implementation for FedAvg clients

class Userditto(User):
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
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.poptimizer = pFeddittoOptimizer(self.persionalized_model.parameters(), lr=self.learning_rate, lamda=self.lamda)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):

        self.persionalized_model.train()
        for epochs in range(1,self.K+1):
            X, y = self.get_next_train_batch()
            self.poptimizer.zero_grad()
            output = self.persionalized_model(X)
            if self.modeltype=='Lasso' or self.modeltype=='Matrix':
                output=torch.squeeze(output)
                y=y.float()
            loss = self.loss(output, y)+self.lamdaCO*self.RL(self.persionalized_model)
            loss.backward()
            self.persionalized_model_bar, _ = self.poptimizer.step(copy.deepcopy(list(self.model.parameters())))

        self.clone_model_paramenter(self.persionalized_model_bar,self.persionalized_model.parameters())

        
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
            loss = self.loss(output, y)+self.lamdaCO*self.RL(self.model)
            loss.backward()
            self.optimizer.step()
        return LOSS

