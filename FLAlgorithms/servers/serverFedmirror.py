import torch
import os
import copy
from FLAlgorithms.users.userFedmirror import UserFedmirror
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
from FLAlgorithms.optimizers.prox_map import RLprox
import numpy as np
 
# Implementation for pFedMe Server

class Fedmirror(Server):
    def __init__(self, device,  dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, K, personal_learning_rate, times,regularizer,lamdaCO,modeltype,l1const,l2const):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times,regularizer,lamdaCO,modeltype,l1const,l2const)

        # Initialize data for all  users
        data = read_data(dataset)
        total_users = len(data[0])
        self.K = K
        self.personal_learning_rate = personal_learning_rate
        for i in range(total_users):
            id, train , test = read_user_data(i, data, dataset)
            user = UserFedmirror(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, K, personal_learning_rate,regularizer,lamdaCO,modeltype,l1const,l2const)
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



    def aggregate_parameters_mirror(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)

        Q=[]
        for q in self.model.parameters():
            Q.append(torch.flatten(q.data))
        
        if self.modeltype!="Lasso" and self.modeltype!="Matrix":
            for ii in range(1,len(Q)):
                Q[0]=torch.cat((Q[0],Q[ii]))
                
        P=RLprox(self.regularizer, Q[0], self.lamdaCO, self.local_epochs* self.learning_rate, self.l1const,self.l2const)
        #P=Q[0]
        ii=0
        for server_param in self.model.parameters():
            if self.modeltype=="Lasso" or self.modeltype=="Matrix": 
                if len(torch.flatten(server_param.data))==1024:
                    server_param.data=P[ii:ii+len(torch.flatten(server_param.data))].reshape(server_param.data.size())
                    ii+=len(torch.flatten(server_param.data))
            else:
                server_param.data=P[ii:ii+len(torch.flatten(server_param.data))].reshape(server_param.data.size())    
                ii+=len(torch.flatten(server_param.data))




    def send_parameters_mirror(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters_mirror_personal(self.model)




    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            # send all parameter for users 
            self.send_parameters()

            # Evaluate gloal model on user for each interation
            print("Evaluate global model")
            print("")
            self.selected_users = self.select_users(glob_iter,self.num_users)
            self.evaluate()

            # do update for all users not only selected users
            for user in self.users:
                user.train(self.local_epochs) #* user.train_samples
            
            # choose several users to send back upated model to server
            # self.personalized_evaluate()
            



            self.aggregate_parameters_mirror()############################################################################################################# Changed



            self.send_parameters_mirror()

            # Evaluate gloal model on user for each interation
            #print("Evaluate persionalized model")
            #print("")
            self.evaluate_personalized_model()############################################## Why deleted? (deleted in the Ziang's code)
            #self.aggregate_parameters()


        #print(loss)
        self.save_results()
        self.save_model()
    
  
