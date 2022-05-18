import torch
import os
import numpy as np
import h5py
from utils.model_utils import Metrics
import copy

class Server:
    def __init__(self, device, dataset,algorithm, model, batch_size, learning_rate ,beta, lamda,
                 num_glob_iters, local_epochs, optimizer,num_users, times,regularizer,lamdaCO,modeltype,l1const,l2const):


        # Set up the main attributes
        self.device = device
        self.dataset = dataset
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.model = copy.deepcopy(model)
        self.users = []
        self.selected_users = []
        self.num_users = num_users
        self.beta = beta
        self.lamda = lamda
        self.algorithm = algorithm
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc,self.rs_train_acc_per, self.rs_train_loss_per, self.rs_glob_acc_per = [], [], [], [], [], []
        self.rs_train_accuracy,self.rs_train_precision,self.rs_train_recall,self.rs_train_F1,self.rs_train_density =[], [], [], [], []
        self.rs_glob_accuracy,self.rs_glob_precision,self.rs_glob_recall,self.rs_glob_F1,self.rs_glob_density= [], [], [], [], []
        self.rs_train_accuracy_per,self.rs_train_precision_per,self.rs_train_recall_per,self.rs_train_F1_per,self.rs_train_density_per =[], [], [], [], []
        self.rs_glob_accuracy_per,self.rs_glob_precision_per,self.rs_glob_recall_per,self.rs_glob_F1_per,self.rs_glob_density_per= [], [], [], [], []
        
        self.rs_train_mse, self.rs_train_rerank, self.rs_train_reerror=[],[],[]
        self.rs_glob_mse, self.rs_glob_rerank, self.rs_glob_reerror=[],[],[]
        self.rs_train_mse_per, self.rs_train_rerank_per, self.rs_train_reerror_per=[],[],[]
        self.rs_glob_mse_per, self.rs_glob_rerank_per, self.rs_glob_reerror_per=[],[],[]
        
        
        self.times = times
        self.regularizer=regularizer
        self.lamdaCO=lamdaCO
        self.modeltype=modeltype
        self.l1const=l1const
        self.l2const=l2const

        # Initialize the server's grads to zeros
        #for param in self.model.parameters():
        #    param.data = torch.zeros_like(param.data)
        #    param.grad = torch.zeros_like(param.data)
        #self.send_parameters()
        
    def aggregate_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)
            
    def add_parameters(self, user, ratio):
        model = self.model.parameters()
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))
    
    def select_users(self, round, num_users):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''
        if(num_users == len(self.users)):
            print("All users are selected")
            return self.users

        num_users = min(num_users, len(self.users))
        #np.random.seed(round)
        return np.random.choice(self.users, num_users, replace=False) #, p=pk)

    # define function for persionalized agegatation.
    def persionalized_update_parameters(self,user, ratio):
        # only argegate the local_weight_update
        for server_param, user_param in zip(self.model.parameters(), user.local_weight_updated):
            server_param.data = server_param.data + user_param.data.clone() * ratio


    def persionalized_aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)

        # store previous parameters
        previous_param = copy.deepcopy(list(self.model.parameters()))
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples

        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)
            #self.add_parameters(user, 1 / len(self.selected_users))

        # aaggregate avergage model with previous model using parameter beta 
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data
            
    # Save loss, accurancy to h5 fiel
    def save_results(self):
        if self.modeltype=="Lasso":
            alg = self.dataset + "_" + self.algorithm
            alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_"  + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs)
            alg = alg + "_" + str(self.times)
            if (len(self.rs_glob_accuracy) != 0 &  len(self.rs_train_accuracy) & len(self.rs_train_loss)) :
                with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                    hf.create_dataset('rs_glob_accuracy', data=self.rs_glob_accuracy)
                    hf.create_dataset('rs_glob_precision', data=self.rs_glob_precision)
                    hf.create_dataset('rs_glob_recall', data=self.rs_glob_recall)
                    hf.create_dataset('rs_glob_F1', data=self.rs_glob_F1)
                    hf.create_dataset('rs_glob_density', data=self.rs_glob_density)
                    
                    hf.create_dataset('rs_train_accuracy', data=self.rs_train_accuracy)
                    hf.create_dataset('rs_train_precision', data=self.rs_train_precision)
                    hf.create_dataset('rs_train_recall', data=self.rs_train_recall)
                    hf.create_dataset('rs_train_F1', data=self.rs_train_F1)
                    hf.create_dataset('rs_train_density', data=self.rs_train_density)
                                        
                    hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                    
                    hf.close()
            
            # store persionalized value
            alg = self.dataset + "_" + self.algorithm + "_p"
            alg = alg  + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b"+ "_" + str(self.local_epochs)
            alg = alg + "_" + str(self.times)
            if (len(self.rs_glob_accuracy_per) != 0 &  len(self.rs_train_accuracy_per) & len(self.rs_train_loss_per)) :
                with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                    hf.create_dataset('rs_glob_accuracy', data=self.rs_glob_accuracy_per)
                    hf.create_dataset('rs_glob_precision', data=self.rs_glob_precision_per)
                    hf.create_dataset('rs_glob_recall', data=self.rs_glob_recall_per)
                    hf.create_dataset('rs_glob_F1', data=self.rs_glob_F1_per)
                    hf.create_dataset('rs_glob_density', data=self.rs_glob_density_per)
                    
                    hf.create_dataset('rs_train_accuracy', data=self.rs_train_accuracy_per)
                    hf.create_dataset('rs_train_precision', data=self.rs_train_precision_per)
                    hf.create_dataset('rs_train_recall', data=self.rs_train_recall_per)
                    hf.create_dataset('rs_train_F1', data=self.rs_train_F1_per)
                    hf.create_dataset('rs_train_density', data=self.rs_train_density_per)
                                        
                    hf.create_dataset('rs_train_loss', data=self.rs_train_loss_per)
                    hf.close() 


        elif self.modeltype=="Matrix":
            alg = self.dataset + "_" + self.algorithm
            alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta)  + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs)
            alg = alg + "_" + str(self.times)
            if (len(self.rs_glob_mse) != 0 &  len(self.rs_train_mse) & len(self.rs_train_loss)) :
                with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                    hf.create_dataset('rs_glob_mse', data=self.rs_glob_mse)
                    hf.create_dataset('rs_glob_rerank', data=self.rs_glob_rerank)
                    hf.create_dataset('rs_glob_reerror', data=self.rs_glob_reerror)
                    
                    hf.create_dataset('rs_train_mse', data=self.rs_train_mse)
                    hf.create_dataset('rs_train_rerank', data=self.rs_train_rerank)
                    hf.create_dataset('rs_train_reerror', data=self.rs_train_reerror)

                                        
                    hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                    
                    hf.close()
            
            # store persionalized value
            alg = self.dataset + "_" + self.algorithm + "_p"
            alg = alg  + "_" + str(self.learning_rate) + "_" + str(self.beta)  + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b"+ "_" + str(self.local_epochs)
            alg = alg + "_" + str(self.times)
            if (len(self.rs_glob_mse_per) != 0 &  len(self.rs_train_mse_per) & len(self.rs_train_loss_per)) :
                with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                    hf.create_dataset('rs_glob_mse', data=self.rs_glob_mse_per)
                    hf.create_dataset('rs_glob_rerank', data=self.rs_glob_rerank_per)
                    hf.create_dataset('rs_glob_reerror', data=self.rs_glob_reerror_per)
                    
                    hf.create_dataset('rs_train_mse', data=self.rs_train_mse_per)
                    hf.create_dataset('rs_train_rerank', data=self.rs_train_rerank_per)
                    hf.create_dataset('rs_train_reerror', data=self.rs_train_reerror_per)

                                        
                    hf.create_dataset('rs_train_loss', data=self.rs_train_loss_per)
                    
                    hf.close() 


        else:
            alg = self.dataset + "_" + self.algorithm
            alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta)  + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs)
            alg = alg + "_" + str(self.times)
            if (len(self.rs_glob_acc) != 0 &  len(self.rs_train_acc) & len(self.rs_train_loss)) :
                with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                    hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
                    hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                    hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                    hf.close()
            
            # store persionalized value
            alg = self.dataset + "_" + self.algorithm + "_p"
            alg = alg  + "_" + str(self.learning_rate) + "_" + str(self.beta)  + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b"+ "_" + str(self.local_epochs)
            alg = alg + "_" + str(self.times)
            if (len(self.rs_glob_acc_per) != 0 &  len(self.rs_train_acc_per) & len(self.rs_train_loss_per)) :
                with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                    hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc_per)
                    hf.create_dataset('rs_train_acc', data=self.rs_train_acc_per)
                    hf.create_dataset('rs_train_loss', data=self.rs_train_loss_per)
                    hf.close()

    def test(self):
        '''tests self.latest_model on given clients
        '''
        if self.modeltype=="Lasso":
            tot_accuracy=[]
            tot_precision=[]
            tot_recall=[]
            tot_F1=[]
            tot_density=[]
            for c in self.users:
                accuracy,precision,recall,F1,density,_=c.test()
                tot_accuracy.append(accuracy)
                tot_precision.append(precision)
                tot_recall.append(recall)
                tot_F1.append(F1)
                tot_density.append(density)
            return tot_accuracy, tot_precision, tot_recall, tot_F1,tot_density
        
        
        
        elif self.modeltype=="Matrix":
            tot_mse=[]
            tot_rerank=[]
            tot_reerror=[]
            for c in self.users:
                _,mse,rerank,reerror,_=c.test()
                tot_mse.append(mse)
                tot_rerank.append(rerank)
                tot_reerror.append(reerror)
            return tot_mse,tot_rerank,tot_reerror



        else:
            num_samples = []
            tot_correct = []
            losses = []
            for c in self.users:
                ct, ns = c.test()
                tot_correct.append(ct*1.0)
                num_samples.append(ns)
            ids = [c.id for c in self.users]

            return ids, num_samples, tot_correct

    def train_error_and_loss(self):
        if self.modeltype=="Lasso":
            tot_accuracy=[]
            tot_precision=[]
            tot_recall=[]
            tot_F1=[]
            tot_density=[]
            losses=[]
            num_samples=[]
            for c in self.users:
                accuracy,precision,recall,F1,density,loss,num_sample=c.train_error_and_loss()
                tot_accuracy.append(accuracy)
                tot_precision.append(precision)
                tot_recall.append(recall)
                tot_F1.append(F1)
                tot_density.append(density)
                num_samples.append(num_sample)
                losses.append(loss)
            return tot_accuracy, tot_precision, tot_recall, tot_F1,tot_density,num_samples,losses

        elif self.modeltype=="Matrix":
            losses=[]
            num_samples=[]
            tot_mse=[]
            tot_rerank=[]
            tot_reerror=[]
            for c in self.users:
                loss,mse,rerank,reerror,num_sample=c.train_error_and_loss()
                tot_mse.append(mse)
                tot_rerank.append(rerank)
                tot_reerror.append(reerror)
                num_samples.append(num_sample)
                losses.append(loss)
            return tot_mse,tot_rerank,tot_reerror,num_samples,losses
        else:
            num_samples = []
            tot_correct = []
            losses = []
            for c in self.users:
                ct, cl, ns = c.train_error_and_loss() 
                tot_correct.append(ct*1.0)
                num_samples.append(ns)
                losses.append(cl*1.0)
            
            ids = [c.id for c in self.users]
            #groups = [c.group for c in self.clients]

            return ids, num_samples, tot_correct, losses

    def test_persionalized_model(self):
        '''tests self.latest_model on given clients
        '''
        if self.modeltype=="Lasso":
            tot_accuracy=[]
            tot_precision=[]
            tot_recall=[]
            tot_F1=[]
            tot_density=[]
            for c in self.users:
                accuracy,precision,recall,F1,density,_=c.test_persionalized_model()
                tot_accuracy.append(accuracy)
                tot_precision.append(precision)
                tot_recall.append(recall)
                tot_F1.append(F1)
                tot_density.append(density)
            return tot_accuracy, tot_precision, tot_recall, tot_F1,tot_density

        elif self.modeltype=="Matrix":
            tot_mse=[]
            tot_rerank=[]
            tot_reerror=[]
            for c in self.users:
                _,mse,rerank,reerror,_=c.test_persionalized_model()
                tot_mse.append(mse)
                tot_rerank.append(rerank)
                tot_reerror.append(reerror)
            return tot_mse,tot_rerank,tot_reerror
        else:
            num_samples = []
            tot_correct = []
            losses = []
            for c in self.users:
                ct, ns = c.test_persionalized_model()
                tot_correct.append(ct*1.0)
                num_samples.append(ns)
            ids = [c.id for c in self.users]

            return ids, num_samples, tot_correct
    def train_error_and_loss_persionalized_model(self):
        if self.modeltype=="Lasso":
            tot_accuracy=[]
            tot_precision=[]
            tot_recall=[]
            tot_F1=[]
            tot_density=[]
            losses=[]
            num_samples=[]
            for c in self.users:
                accuracy,precision,recall,F1,density,loss,num_sample=c.train_error_and_loss_persionalized_model()
                tot_accuracy.append(accuracy)
                tot_precision.append(precision)
                tot_recall.append(recall)
                tot_F1.append(F1)
                tot_density.append(density)
                num_samples.append(num_sample)
                losses.append(loss)
            return tot_accuracy, tot_precision, tot_recall, tot_F1,tot_density,num_samples,losses
        elif self.modeltype=="Matrix":
            losses=[]
            num_samples=[]
            tot_mse=[]
            tot_rerank=[]
            tot_reerror=[]
            for c in self.users:
                loss,mse,rerank,reerror,num_sample=c.train_error_and_loss_persionalized_model()
                tot_mse.append(mse)
                tot_rerank.append(rerank)
                tot_reerror.append(reerror)
                num_samples.append(num_sample)
                losses.append(loss)
            return tot_mse,tot_rerank,tot_reerror,num_samples,losses
        else:
            num_samples = []
            tot_correct = []
            losses = []
            for c in self.users:
                ct, cl, ns = c.train_error_and_loss_persionalized_model() 
                tot_correct.append(ct*1.0)
                num_samples.append(ns)
                losses.append(cl*1.0)
            
            ids = [c.id for c in self.users]
            #groups = [c.group for c in self.clients]

            return ids, num_samples, tot_correct, losses






    def evaluate(self):
        if self.modeltype=="Lasso":
            stats = self.test()  
            stats_train = self.train_error_and_loss()
            glob_accuracy=sum(stats[0])/len(self.users)
            glob_precision=sum(stats[1])/len(self.users)
            glob_recall=sum(stats[2])/len(self.users)
            glob_F1=sum(stats[3])/len(self.users)
            glob_density=sum(stats[4])/len(self.users)

            train_accuracy=sum(stats_train[0])/len(self.users)
            train_precision=sum(stats_train[1])/len(self.users)
            train_recall=sum(stats_train[2])/len(self.users)
            train_F1=sum(stats_train[3])/len(self.users)
            train_density=sum(stats_train[4])/len(self.users)
            train_loss=sum(stats_train[6])/sum(stats_train[5])

            self.rs_glob_accuracy.append(glob_accuracy)
            self.rs_glob_precision.append(glob_precision)
            self.rs_glob_recall.append(glob_recall)
            self.rs_glob_F1.append(glob_F1)
            self.rs_glob_density.append(glob_density)
            
            self.rs_train_accuracy.append(train_accuracy)
            self.rs_train_precision.append(train_precision)
            self.rs_train_recall.append(train_recall)
            self.rs_train_F1.append(train_F1)
            self.rs_train_density.append(train_density)
            self.rs_train_loss.append(train_loss)
            print("Average Global Accurancy: ", glob_accuracy)
            print("Average Global Trainning Accurancy: ", train_accuracy)
            print("Average Global Trainning Loss: ",train_loss)
        elif self.modeltype=="Matrix":
            stats = self.test()  
            stats_train = self.train_error_and_loss()
            glob_mse=sum(stats[0])/len(self.users)
            glob_rerank=sum(stats[1])/len(self.users)
            glob_reerror=sum(stats[2])/len(self.users)

            train_mse=sum(stats_train[0])/len(self.users)
            train_rerank=sum(stats_train[1])/len(self.users)
            train_reerror=sum(stats_train[2])/len(self.users)
            train_loss=sum(stats_train[4])/sum(stats_train[3])

            self.rs_glob_mse.append(glob_mse)
            self.rs_glob_rerank.append(glob_rerank)
            self.rs_glob_reerror.append(glob_reerror)

            self.rs_train_mse.append(train_mse)
            self.rs_train_rerank.append(train_rerank)
            self.rs_train_reerror.append(train_reerror)
            self.rs_train_loss.append(train_loss)

            print("Average Global reerror: ", glob_reerror)
            print("Average Global Trainning reerror: ", train_reerror)
            print("Average Global Trainning Loss: ",train_loss)
        else:
            stats = self.test()  
            stats_train = self.train_error_and_loss()
            glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
            train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
            # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
            train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]) / np.sum(stats_train[1])
            self.rs_glob_acc.append(glob_acc)
            self.rs_train_acc.append(train_acc)
            self.rs_train_loss.append(train_loss)
            #print("stats_train[1]",stats_train[3][0])
            print("Average Global Accurancy: ", glob_acc)
            print("Average Global Trainning Accurancy: ", train_acc)
            print("Average Global Trainning Loss: ",train_loss)

    def evaluate_personalized_model(self):
        if self.modeltype=="Lasso":
            stats = self.test_persionalized_model()    
            stats_train = self.train_error_and_loss_persionalized_model()  
            glob_accuracy=sum(stats[0])/len(self.users)
            glob_precision=sum(stats[1])/len(self.users)
            glob_recall=sum(stats[2])/len(self.users)
            glob_F1=sum(stats[3])/len(self.users)
            glob_density=sum(stats[4])/len(self.users)

            train_accuracy=sum(stats_train[0])/len(self.users)
            train_precision=sum(stats_train[1])/len(self.users)
            train_recall=sum(stats_train[2])/len(self.users)
            train_F1=sum(stats_train[3])/len(self.users)
            train_density=sum(stats_train[4])/len(self.users)
            train_loss=sum(stats_train[6])/sum(stats_train[5])

            self.rs_glob_accuracy_per.append(glob_accuracy)
            self.rs_glob_precision_per.append(glob_precision)
            self.rs_glob_recall_per.append(glob_recall)
            self.rs_glob_F1_per.append(glob_F1)
            self.rs_glob_density_per.append(glob_density)
            
            self.rs_train_accuracy_per.append(train_accuracy)
            self.rs_train_precision_per.append(train_precision)
            self.rs_train_recall_per.append(train_recall)
            self.rs_train_F1_per.append(train_F1)
            self.rs_train_density_per.append(train_density)
            self.rs_train_loss_per.append(train_loss)
            print("Average Personal Accurancy: ", glob_accuracy)
            print("Average Personal Trainning Accurancy: ", train_accuracy)
            print("Average Personal Trainning Loss: ",train_loss)

        elif self.modeltype=="Matrix":
            stats = self.test_persionalized_model()    
            stats_train = self.train_error_and_loss_persionalized_model()  
            glob_mse=sum(stats[0])/len(self.users)
            glob_rerank=sum(stats[1])/len(self.users)
            glob_reerror=sum(stats[2])/len(self.users)

            train_mse=sum(stats_train[0])/len(self.users)
            train_rerank=sum(stats_train[1])/len(self.users)
            train_reerror=sum(stats_train[2])/len(self.users)
            train_loss=sum(stats_train[4])/sum(stats_train[3])

            self.rs_glob_mse_per.append(glob_mse)
            self.rs_glob_rerank_per.append(glob_rerank)
            self.rs_glob_reerror_per.append(glob_reerror)

            self.rs_train_mse_per.append(train_mse)
            self.rs_train_rerank_per.append(train_rerank)
            self.rs_train_reerror_per.append(train_reerror)
            self.rs_train_loss_per.append(train_loss)

            print("Average Personal reerror: ", glob_reerror)
            print("Average Personal Trainning reerror: ", train_reerror)
            print("Average Personal Trainning Loss: ",train_loss)
        else:
            stats = self.test_persionalized_model()  
            stats_train = self.train_error_and_loss_persionalized_model()
            glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
            train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
            # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
            train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]) / np.sum(stats_train[1])
            self.rs_glob_acc_per.append(glob_acc)
            self.rs_train_acc_per.append(train_acc)
            self.rs_train_loss_per.append(train_loss)
            #print("stats_train[1]",stats_train[3][0])
            print("Average Personal Accurancy: ", glob_acc)
            print("Average Personal Trainning Accurancy: ", train_acc)
            print("Average Personal Trainning Loss: ",train_loss)




    def evaluate_one_step(self):
        for c in self.users:
            c.train_one_step()

        stats = self.test()  
        stats_train = self.train_error_and_loss()

        # set local model back to client for training process.
        for c in self.users:
            c.update_parameters(c.local_model)

        if self.modeltype=="Lasso":
            glob_accuracy=sum(stats[0])/len(self.users)
            glob_precision=sum(stats[1])/len(self.users)
            glob_recall=sum(stats[2])/len(self.users)
            glob_F1=sum(stats[3])/len(self.users)
            glob_density=sum(stats[4])/len(self.users)

            train_accuracy=sum(stats_train[0])/len(self.users)
            train_precision=sum(stats_train[1])/len(self.users)
            train_recall=sum(stats_train[2])/len(self.users)
            train_F1=sum(stats_train[3])/len(self.users)
            train_density=sum(stats_train[4])/len(self.users)
            train_loss=sum(stats_train[6])/sum(stats_train[5])

            self.rs_glob_accuracy_per.append(glob_accuracy)
            self.rs_glob_precision_per.append(glob_precision)
            self.rs_glob_recall_per.append(glob_recall)
            self.rs_glob_F1_per.append(glob_F1)
            self.rs_glob_density_per.append(glob_density)
            
            self.rs_train_accuracy_per.append(train_accuracy)
            self.rs_train_precision_per.append(train_precision)
            self.rs_train_recall_per.append(train_recall)
            self.rs_train_F1_per.append(train_F1)
            self.rs_train_density_per.append(train_density)
            self.rs_train_loss_per.append(train_loss)
            print("Average Personal Accurancy: ", glob_accuracy)
            print("Average Personal Trainning Accurancy: ", train_accuracy)
            print("Average Personal Trainning Loss: ",train_loss)

        elif self.modeltype=="Matrix":
            glob_mse=sum(stats[0])/len(self.users)
            glob_rerank=sum(stats[1])/len(self.users)
            glob_reerror=sum(stats[2])/len(self.users)

            train_mse=sum(stats_train[0])/len(self.users)
            train_rerank=sum(stats_train[1])/len(self.users)
            train_reerror=sum(stats_train[2])/len(self.users)
            train_loss=sum(stats_train[4])/sum(stats_train[3])

            self.rs_glob_mse_per.append(glob_mse)
            self.rs_glob_rerank_per.append(glob_rerank)
            self.rs_glob_reerror_per.append(glob_reerror)

            self.rs_train_mse_per.append(train_mse)
            self.rs_train_rerank_per.append(train_rerank)
            self.rs_train_reerror_per.append(train_reerror)
            self.rs_train_loss_per.append(train_loss)

            print("Average Personal reerror: ", glob_reerror)
            print("Average Personal Trainning reerror: ", train_reerror)
            print("Average Personal Trainning Loss: ",train_loss)
        else:
            glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
            train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
            # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
            train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]) / np.sum(stats_train[1])
            self.rs_glob_acc_per.append(glob_acc)
            self.rs_train_acc_per.append(train_acc)
            self.rs_train_loss_per.append(train_loss)
            #print("stats_train[1]",stats_train[3][0])
            print("Average Personal Accurancy: ", glob_acc)
            print("Average Personal Trainning Accurancy: ", train_acc)
            print("Average Personal Trainning Loss: ",train_loss)
