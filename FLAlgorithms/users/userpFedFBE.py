import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.optimizers.fedoptimizer import pFedFBEOptimizer
from FLAlgorithms.users.userbase import User
import copy

# Implementation for pFeMe clients

class UserpFedFBE(User):
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
        self.local_grad= [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.optimizer = pFedFBEOptimizer(self.model.parameters(), lr=self.personal_learning_rate, lamda=self.lamda)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        LOSS = 0
        self.model.train()
        persionalized_model = copy.deepcopy(list(self.persionalized_model.parameters()))######################################################################Use personalized model
        for epoch in range(1, self.local_epochs + 1):  # local update
            
            self.model.train()
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            output = self.model(X)
            if self.modeltype=='Lasso' or self.modeltype=='Matrix':
                output=torch.squeeze(output)
                y = y.float()
            loss = self.loss(output, y)
            loss.backward()
            persionalized_model, _ = self.optimizer.step(persionalized_model, self.regularizer, self.lamda, self.personal_learning_rate, self.lamdaCO,self.modeltype,self.l1const,self.l2const)
            
        self.clone_model_paramenter(persionalized_model, self.persionalized_model_bar)
        self.clone_model_paramenter(persionalized_model, self.persionalized_model.parameters())
        ###self.clone_model_paramenter(self.model.parameters(),self.local_model)################################################Why deleted?
        return LOSS