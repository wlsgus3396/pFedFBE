import torch
import os

from FLAlgorithms.users.userperavg import UserPerAvg
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data

# Implementation for per-FedAvg Server

class PerAvg(Server):
    def __init__(self,device, dataset,algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users,times,regularizer,lamdaCO,modeltype,l1const,l2const):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times,regularizer,lamdaCO,modeltype,l1const,l2const)

        # Initialize data for all  users
        data = read_data(dataset)
        total_users = len(data[0])
        for i in range(total_users):
            id, train , test = read_user_data(i, data, dataset)
            user = UserPerAvg(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer ,total_users , num_users,regularizer,lamdaCO,modeltype,l1const,l2const)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating Local Per-Avg.")

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

    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            # send all parameter for users 
            self.send_parameters()
            self.selected_users = self.select_users(glob_iter,self.num_users)
            self.evaluate()
            # Evaluate gloal model on user for each interation
            print("Evaluate global model with one step update")
            print("")
            

            # choose several users to send back upated model to server
            
            for user in self.selected_users:
                user.train(self.local_epochs) #* user.train_samples

            self.evaluate_one_step()    
            self.aggregate_parameters()

        self.save_results()
        self.save_model()
