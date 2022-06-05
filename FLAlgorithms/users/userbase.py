from concurrent.futures import thread
from sklearn.metrics import accuracy_score, precision_recall_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy
from collections import defaultdict, abc as container_abcs
from copy import deepcopy
from itertools import chain
import warnings
import functools
import math

class User:
    """
    Base class for users in federated learning.
    """
    def __init__(self, device, id, train_data, test_data, model, batch_size = 0, learning_rate = 0, beta = 0 , lamda = 0, local_epochs = 0, regularizer='l1',lamdaCO=0,modeltype='DNN',l1const=0,l2const=0):

        self.device = device
        self.model = copy.deepcopy(model)
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.lamda = lamda
        self.local_epochs = local_epochs
        self.trainloader = DataLoader(train_data, self.batch_size)
        self.testloader =  DataLoader(test_data, self.batch_size)
        self.testloaderfull = DataLoader(test_data, self.test_samples)
        self.trainloaderfull = DataLoader(train_data, self.train_samples)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)
        self.regularizer=regularizer
        # those parameters are for persionalized federated learing.
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.persionalized_model = copy.deepcopy(model)
        self.persionalized_model_bar = copy.deepcopy(list(self.model.parameters()))
        self.lamdaCO=lamdaCO
        self.modeltype=modeltype

        self.delta_model = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.server_model = copy.deepcopy(list(self.model.parameters()))
        self.l1const=l1const
        self.l2const=l2const
        
        
        #if(model[1] == "Mclr_CrossEntropy"):
        #    self.loss = nn.CrossEntropyLoss()
        #else:
        
        

    def set_parameters(self, model):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()
        #self.local_weight_updated = copy.deepcopy(self.optimizer.param_groups[0]['params'])

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()
    
    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param
    
    def get_updated_parameters(self):
        return self.local_weight_updated
    
    def update_parameters(self, new_params):
        for param , new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def test(self):
        if self.modeltype=='Lasso':
            self.model.eval()
            TP=0
            TN=0
            FP=0
            FN=0
            count=0

            accuracy=0
            precision=0
            F1=0
            density=0
            loss=0

            W=[]
            W.append(np.concatenate((np.ones(32),np.zeros(1024-32)), axis=None))
            thr=0.01
            test_model =copy.deepcopy(self.model)
            test_model=test_model.linear.weight[0]
            W=torch.tensor(W)
            P=abs(test_model[torch.nonzero(W[0])])
            N=abs(test_model[W[0]==0])
            TP=sum(P>thr)
            FP=sum(P<thr)
            TN=sum(N<thr)
            FN=sum(N>thr)
            count=TP+FN
            accuracy=(TN+TP)/(TN+FP+TP+FN)
            precision=TP/(TP+FP)
            recall=TP/(TP+FN)
            if precision+recall!=0:
                F1=2*(precision*recall)/(precision+recall)
            else:
                F1=0
            density=count/1024




            for x, y, w in self.testloaderfull:
                x, y,w = x.to(self.device), y.to(self.device),w.to(self.device)
                output = self.model(x)
                output=torch.squeeze(output)
                y=y.float()
                loss += self.loss(output, y)+self.lamdaCO*self.RL(self.model)
            







            return accuracy.item(),precision.item(),recall.item(),F1.item(),density.item(),y.shape[0]
        
        elif self.modeltype=='Matrix':
            self.model.eval()
            
            mse=0
            loss=0
            rerank=0
            reerror=0

            thr=0.01
            test_model =copy.deepcopy(self.model)
            test_model=test_model.linear.weight[0]
            test_model=torch.reshape(test_model,(32,32))
            
            W=torch.zeros((32,32))
            for i in range(16):
                W[i][i]=1
            W=W.to(self.device)
            reerror=torch.linalg.norm(test_model-W,'fro')
            rerank=torch.linalg.matrix_rank(test_model,0.01)
            for x, y,w in self.testloaderfull:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                output=torch.squeeze(output)
                loss += self.loss(output, y)+self.lamdaCO*self.RL(self.model)
                mse += self.loss(output, y)
            return loss.item(),mse.item(),rerank.item(),reerror.item(),y.shape[0]







        else:

            self.model.eval()
            test_acc = 0
            for x, y in self.testloaderfull:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                #@loss += self.loss(output, y)
                #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
                #print(self.id + ", Test Loss:", loss)
            return test_acc, y.shape[0]

    




    def train_error_and_loss(self):

        if self.modeltype=='Lasso':
            self.model.eval()
            TP=0
            TN=0
            FP=0
            FN=0
            count=0

            accuracy=0
            precision=0
            F1=0
            density=0
            loss=0


            W=[]
            W.append(np.concatenate((np.ones(32),np.zeros(1024-32)), axis=None))
            W=torch.tensor(W)
            thr=0.01
            test_model =copy.deepcopy(self.model)
            test_model=test_model.linear.weight[0]
            P=abs(test_model[torch.nonzero(W[0])])
            N=abs(test_model[W[0]==0])
            TP=sum(P>thr)
            FP=sum(P<thr)
            TN=sum(N<thr)
            FN=sum(N>thr)
            count=TP+FN
            accuracy=(TN+TP)/(TN+FP+TP+FN)
            precision=TP/(TP+FP)
            recall=TP/(TP+FN)
            if precision+recall!=0:
                F1=2*(precision*recall)/(precision+recall)
            else:
                F1=0
            density=count/1024





            for x, y, w in self.trainloaderfull:
                x, y,w = x.to(self.device), y.to(self.device),w.to(self.device)
                output = self.model(x)
                output=torch.squeeze(output)
                y=y.float()
                loss += self.loss(output, y)+self.lamdaCO*self.RL(self.model)
            
            return accuracy.item(),precision.item(),recall.item(),F1.item(),density.item(),loss.item(),self.train_samples





        elif self.modeltype=='Matrix':
            self.model.eval()
            
            mse=0
            loss=0
            rerank=0
            reerror=0

            thr=0.01
            test_model =copy.deepcopy(self.model)
            test_model=test_model.linear.weight[0]
            test_model=torch.reshape(test_model,(32,32))
            
            W=torch.zeros((32,32))
            for i in range(16):
                W[i][i]=1
            W=W.to(self.device)
            reerror=torch.linalg.norm(test_model-W,'fro')
            rerank=torch.linalg.matrix_rank(test_model,0.01)
            for x, y,w in self.trainloaderfull:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                output=torch.squeeze(output)
                y=y.float()
                loss += self.loss(output, y)+self.lamdaCO*self.RL(self.model)
                mse += self.loss(output, y)
            return loss.item(),mse.item(),rerank.item(),reerror.item(),self.train_samples


        else:
            self.model.eval()
            train_acc = 0
            loss = 0
            for x, y in self.trainloaderfull:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                loss += self.loss(output, y)+self.lamdaCO*self.RL(self.model)
                #print(self.id + ", Train Accuracy:", train_acc)
                #print(self.id + ", Train Loss:", loss)
            return train_acc, loss.item() , self.train_samples
    

        
        
    def test_persionalized_model(self):
        Local = copy.deepcopy(list(self.model.parameters()))
        self.update_parameters(self.persionalized_model_bar)
        if self.modeltype=='Lasso':
            self.model.eval()
            
            loss=0


            for x, y, w in self.testloaderfull:
                x, y,w = x.to(self.device), y.to(self.device),w.to(self.device)
                output = self.model(x)
                output=torch.squeeze(output)
                y=y.float()
                loss += self.loss(output, y)+self.lamdaCO*self.RL(self.model)
                W=w



            TP=0
            TN=0
            FP=0
            FN=0
            count=0

            accuracy=0
            precision=0
            F1=0
            density=0
            

            thr=0.01
            test_model =copy.deepcopy(self.model)
            test_model=test_model.linear.weight[0]
            P=abs(test_model[torch.nonzero(W[0])])
            N=abs(test_model[W[0]==0])
            TP=sum(P>thr)
            FP=sum(P<thr)
            TN=sum(N<thr)
            FN=sum(N>thr)
            count=TP+FN
            accuracy=(TN+TP)/(TN+FP+TP+FN)
            precision=TP/(TP+FP)
            recall=TP/(TP+FN)
            if precision+recall!=0:
                F1=2*(precision*recall)/(precision+recall)
            else:
                F1=0
            density=count/1024


            self.update_parameters(Local)
            return accuracy.item(),precision.item(),recall.item(),F1.item(),density.item(),y.shape[0]
        
        elif self.modeltype=='Matrix':
            self.model.eval()
            
            mse=0
            loss=0

            for x, y,W in self.testloaderfull:
                x, y,W = x.to(self.device), y.to(self.device), W.to(self.device)
                output = self.model(x)
                output=torch.squeeze(output)
                loss += self.loss(output, y)+self.lamdaCO*self.RL(self.model)
                mse += self.loss(output, y)


            rerank=0
            reerror=0

            thr=0.01
            test_model =copy.deepcopy(self.model)
            test_model=test_model.linear.weight[0]
            test_model=torch.reshape(test_model,(32,32))
            
            
            reerror=torch.linalg.norm(test_model-W[0],'fro')
            rerank=torch.linalg.matrix_rank(test_model,0.01)
            
            self.update_parameters(Local)
            return loss.item(),mse.item(),rerank.item(),reerror.item(),y.shape[0]


        
        else:

            self.model.eval()
            test_acc = 0
            for x, y in self.testloaderfull:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                #@loss += self.loss(output, y)
                #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
                #print(self.id + ", Test Loss:", loss)
            self.update_parameters(Local)
            return test_acc, y.shape[0]





    def train_error_and_loss_persionalized_model(self):
        Local = copy.deepcopy(list(self.model.parameters()))
        self.update_parameters(self.persionalized_model_bar)

        if self.modeltype=='Lasso':
            self.model.eval()

            loss=0
            for x, y,w in self.trainloaderfull:
                x, y,w = x.to(self.device), y.to(self.device),w.to(self.device)
                output = self.model(x)
                output=torch.squeeze(output)
                y=y.float()
                loss += self.loss(output, y)+self.lamdaCO*self.RL(self.model)
                W=w

            
            TP=0
            TN=0
            FP=0
            FN=0
            count=0

            accuracy=0
            precision=0
            F1=0
            density=0
            

            thr=0.01
            test_model =copy.deepcopy(self.model)
            test_model=test_model.linear.weight[0]

            P=abs(test_model[torch.nonzero(W[0])])
            N=abs(test_model[W[0]==0])

            TP=sum(P>thr)
            FP=sum(P<thr)
            TN=sum(N<thr)
            FN=sum(N>thr)

            
            count=TP+FN
            accuracy=(TN+TP)/(TN+FP+TP+FN)
            precision=TP/(TP+FP)
            recall=TP/(TP+FN)
            if precision+recall!=0:
                F1=2*(precision*recall)/(precision+recall)
            else:
                F1=0
            density=count/1024



            self.update_parameters(Local)
            return accuracy.item(),precision.item(),recall.item(),F1.item(),density.item(),loss.item(),self.train_samples



        elif self.modeltype=='Matrix':
            self.model.eval()
            loss=0
            mse=0

            for x, y,W in self.trainloaderfull:
                x, y,W = x.to(self.device), y.to(self.device), W.to(self.device)
                output = self.model(x)
                output=torch.squeeze(output)
                y=y.float()
                loss += self.loss(output, y)+self.lamdaCO*self.RL(self.model)
                mse += self.loss(output, y)


            
            rerank=0
            reerror=0

            thr=0.01
            test_model =copy.deepcopy(self.model)
            test_model=test_model.linear.weight[0]
            test_model=torch.reshape(test_model,(32,32))
            
            
            reerror=torch.linalg.norm(test_model-W[0],'fro')
            rerank=torch.linalg.matrix_rank(test_model,0.01)
            
            self.update_parameters(Local)
            return loss.item(),mse.item(),rerank.item(),reerror.item(),self.train_samples


        else:
            self.model.eval()
            train_acc = 0
            loss = 0
            for x, y in self.trainloaderfull:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                loss += self.loss(output, y)+self.lamdaCO*self.RL(self.model)
                #print(self.id + ", Train Accuracy:", train_acc)
                #print(self.id + ", Train Loss:", loss)
            self.update_parameters(Local)
            return train_acc, loss.item() , self.train_samples











            
    
    def get_next_train_batch(self):
        if self.modeltype=="Lasso" or self.modeltype=="Matrix":
            try:
                # Samples a new batch for persionalizing
                (X, y,_) = next(self.iter_trainloader)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                self.iter_trainloader = iter(self.trainloader)
                (X, y,_) = next(self.iter_trainloader)
            return (X.to(self.device), y.to(self.device))
        else:
            try:
                # Samples a new batch for persionalizing
                (X, y) = next(self.iter_trainloader)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                self.iter_trainloader = iter(self.trainloader)
                (X, y) = next(self.iter_trainloader)
            return (X.to(self.device), y.to(self.device))
    
    def get_next_test_batch(self):
        if self.modeltype=="Lasso" or self.modeltype=="Matrix":
            try:
                # Samples a new batch for persionalizing
                (X, y,_) = next(self.iter_testloader)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                self.iter_testloader = iter(self.testloader)
                (X, y,_) = next(self.iter_testloader)
            return (X.to(self.device), y.to(self.device))
        else:
            try:
                # Samples a new batch for persionalizing
                (X, y) = next(self.iter_testloader)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                self.iter_testloader = iter(self.testloader)
                (X, y) = next(self.iter_testloader)
            return (X.to(self.device), y.to(self.device))
    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))
    
    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))


    def RL(self,w):
        if self.regularizer=="none":
            return 0
        if self.regularizer=="l1-constraint":
            if self.modeltype=="Lasso":
                l1=torch.linalg.norm(w.linear.weight[0],1)
            else:
                l1=sum([torch.abs(p).sum() for p in w.parameters()])


            lc=l1>1500     ###DNN:1500 
            lc=lc*(10**10)
            #if l1>1500:
            #    lc=torch.nan
            #else:
            #    lc=0
            
            return lc

        if self.regularizer=="l2-constraint":
            if self.modeltype=="Lasso":
                l2=torch.linalg.norm(w.linear.weight[0],2)
            else:
                l2=torch.sqrt(sum([(p**2).sum() for p in w.parameters()]))


            l2=torch.sqrt(sum([(p**2).sum() for p in w.parameters()]))
            lc=l2>6.5               ###DNN:6.5
            lc=lc*(10**10)
            return lc





        if self.regularizer=="l1":
            #l1_parameters = []
            #for parameter in w.parameters():
            #    l1_parameters.append(parameter.view(-1))
            if self.modeltype=="Lasso":
                l1=torch.linalg.norm(w.linear.weight[0],1)
            else:
                l1=sum([torch.abs(p).sum() for p in w.parameters()])
            return l1

        if self.regularizer=="l2":
            if self.modeltype=="Lasso":
                l2=torch.linalg.norm(w.linear.weight[0],2)
            else:
                l2=torch.sqrt(sum([(p**2).sum() for p in w.parameters()]))
            return l2




        if self.regularizer=="nuclear":
            w =copy.deepcopy(w)
            w=w.linear.weight[0]
            w=torch.reshape(w,(32,32))
            nuc=torch.linalg.norm(w,'nuc')
            return nuc

