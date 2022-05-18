import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import copy
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User
from FLAlgorithms.optimizers.fedoptimizer import SCAFFOLD
import math
# Implementation for FedAvg clients

class UserSCAFFOLD(User):
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
        
        
        
        self.optimizer = SCAFFOLD(self.model.parameters(), lr=self.learning_rate, lamda=self.lamda)
        self.controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.server_controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.delta_controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.csi = None




    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]




    def set_parameters(self, model):
        for old_param, new_param, local_param, server_param in zip(self.model.parameters(), model.parameters(),   self.local_model,   self.server_model):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()
            server_param.data = new_param.data.clone()


    def get_delta_controls(self):
        for param in self.delta_controls:
            param.detach()
        return self.delta_controls




    def train(self, epochs):


        LOSS = 0
        self.model.train()
        grads = self.get_grads()
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
            self.optimizer.step(self.server_controls,self.controls)
            




        self.clone_model_paramenter(self.model.parameters(), self.persionalized_model_bar)
        self.clone_model_paramenter(self.model.parameters(), self.local_model)


        for local, server, delta in zip(self.model.parameters(), self.server_model, self.delta_model):
            delta.data = local.data.detach() - server.data.detach()

        # get client new controls
        new_controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        opt = 2
        if opt == 1:
            for new_control, grad in zip(new_controls, grads):
                new_control.data = grad.grad
        if opt == 2:
            for server_control, control, new_control, delta in zip(self.server_controls, self.controls, new_controls,
                                                                   self.delta_model):
                a = 1 / (math.ceil(self.train_samples / self.batch_size) * self.learning_rate)
                new_control.data = control.data - server_control.data - delta.data * a

        # get controls differences
        for control, new_control, delta in zip(self.controls, new_controls, self.delta_controls):
            delta.data = new_control.data - control.data
            control.data = new_control.data

        return LOSS

