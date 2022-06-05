import torch
import os
import copy

from FLAlgorithms.users.userFedDual import UserFedDual
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
from FLAlgorithms.optimizers.prox_map import RLprox
import numpy as np
 
# Implementation for pFedMe Server

class FedDual(Server):
    def __init__(self, device,  dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, K, personal_learning_rate, times,regularizer,lamdaCO,modeltype,l1const,l2const):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times,regularizer,lamdaCO,modeltype,l1const,l2const)


        self.local_weight_dual = copy.deepcopy(list(self.model.parameters()))

        # Initialize data for all  users
        data = read_data(dataset)
        total_users = len(data[0])
        self.K = K
        self.personal_learning_rate = personal_learning_rate
        for i in range(total_users):
            id, train , test = read_user_data(i, data, dataset)
            user = UserFedDual(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, K, personal_learning_rate,regularizer,lamdaCO,modeltype,l1const,l2const)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating pFedMe server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)
    def send_parameters_FedDual(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters_FedDual(self.model, self.local_weight_dual)





    def add_parameters_FedDual(self, user, ratio):
        for server_param, user_param in zip(self.local_weight_dual, user.get_local_weight_dual()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters_FedDual(self, glob_iter):
        assert (self.users is not None and len(self.users) > 0)
        for param_dual in self.local_weight_dual:
            param_dual.data = torch.zeros_like(param_dual.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters_FedDual(user, user.train_samples / total_train)



        Q=[]
        for q in self.local_weight_dual:
            Q.append(torch.flatten(q.data))
        
        if self.modeltype!="Lasso" and self.modeltype!="Matrix":
            for ii in range(1,len(Q)):
                Q[0]=torch.cat((Q[0],Q[ii]))
        P=RLprox(self.regularizer, Q[0], self.lamdaCO, self.local_epochs * (glob_iter +1) * self.learning_rate, self.l1const,self.l2const)
        ii=0
        for server_param in self.model.parameters():
            if self.modeltype=="Lasso" or self.modeltype=="Matrix": 
                if len(torch.flatten(server_param.data))==1024:
                    server_param.data=P[ii:ii+len(torch.flatten(server_param.data))].reshape(server_param.data.size())
                    ii+=len(torch.flatten(server_param.data))
            else:
                server_param.data=P[ii:ii+len(torch.flatten(server_param.data))].reshape(server_param.data.size())    
                ii+=len(torch.flatten(server_param.data))


    
    def send_parameters_dual(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters_dual_personal(self.model)



    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ", glob_iter, " -------------")
            # send all parameter for users 
            self.send_parameters_FedDual()

            # Evaluate gloal model on user for each interation
            print("Evaluate global model")
            print("")
            self.selected_users = self.select_users(glob_iter,self.num_users)
            self.evaluate()

            # do update for all users not only selected users
            for user in self.users:
                user.train(self.local_epochs, glob_iter) #* user.train_samples
            
            # choose several users to send back upated model to server
            # self.personalized_evaluate()
            

            
            self.aggregate_parameters_FedDual(glob_iter)
            self.send_parameters_dual
            self.evaluate_personalized_model()
        #print(loss)
        self.save_results()
        self.save_model()
    
  
