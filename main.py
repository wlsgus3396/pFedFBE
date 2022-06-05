#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
import copy
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverpFedMe import pFedMe
from FLAlgorithms.servers.serverperavg import PerAvg
from FLAlgorithms.servers.serverpFedditto import pFedditto
from FLAlgorithms.servers.serverpFedprox import pFedprox
from FLAlgorithms.servers.serverSCAFFOLD import SCAFFOLD
from FLAlgorithms.servers.serverFedDyn import FedDyn
from FLAlgorithms.servers.serverpFedFBE import pFedFBE
from FLAlgorithms.servers.serverFedmirror import Fedmirror
from FLAlgorithms.servers.serverFedDual import FedDual
from FLAlgorithms.trainmodel.models import *
from utils.plot_utils import *
import torch
torch.manual_seed(0)

def main(dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
         local_epochs, optimizer, numusers, K, personal_learning_rate, times, gpu, regularizer,lamdaCO,l1const,l2const):

    # Get device status: Check GPU or CPU
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    for i in range(times):
        print("---------------Running time:------------",i)
        # Generate model
        modeltype=copy.deepcopy(model)
        if(model == "mclr"):
            if(dataset == "Mnist"):
                model = Mclr_Logistic().to(device), model
            else:
                model = Mclr_Logistic(60,10).to(device), model
                
        if(model == "cnn"):
            if(dataset == "Mnist"):
                model = Net().to(device), model
            elif(dataset == "Cifar10"):
                model = CNNCifar(10).to(device), model
            
        if(model == "dnn"):
            if(dataset == "Mnist"):
                model = DNN().to(device), model
            else: 
                model = DNN(60,20,10).to(device), model

        if(model=="Lasso"):
            if(dataset=="Lasso"):
                model=Lasso().to(device), model

        if(model=="Matrix"):
            if(dataset=="Matrix"):
                model=Matrix().to(device), model


        # select algorithm
        if(algorithm == "FedAvg"):
            server = FedAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, i,regularizer,lamdaCO,modeltype,l1const,l2const)
        
        if(algorithm == "pFedMe"):
            server = pFedMe(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i,regularizer,lamdaCO,modeltype,l1const,l2const)

        if(algorithm == "pPerAvg"):
            server = PerAvg(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, i,regularizer,lamdaCO,modeltype,l1const,l2const)
        if(algorithm == "pFedditto"):
            server = pFedditto(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i,regularizer,lamdaCO,modeltype,l1const,l2const)
        if(algorithm == "pFedprox"):
            server = pFedprox(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i,regularizer,lamdaCO,modeltype,l1const,l2const)
        if(algorithm == "pSCAFFOLD"):
            server = SCAFFOLD(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i,regularizer,lamdaCO,modeltype,l1const,l2const)
        if(algorithm == "pFedDyn"):
            server = FedDyn(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i,regularizer,lamdaCO,modeltype,l1const,l2const)
        if(algorithm == "pFedFBE"):
            server = pFedFBE(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i,regularizer,lamdaCO,modeltype,l1const,l2const)
        if(algorithm == "Fedmirror"):
            server = Fedmirror(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i,regularizer,lamdaCO,modeltype,l1const,l2const)
        if(algorithm == "FedDual"):
            server = FedDual(device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i,regularizer,lamdaCO,modeltype,l1const,l2const)

        server.train()
        server.test()

    # Average data 
    if(algorithm == "pPerAvg"):
        average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,learning_rate=learning_rate, beta = beta, algorithms="pPerAvg_p", batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times,modeltype=modeltype)
    if(algorithm == "FedAvg"):
        average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,learning_rate=learning_rate, beta = beta, algorithms="FedAvg_p", batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times,modeltype=modeltype)
    if(algorithm == "pFedMe"):
        average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,learning_rate=learning_rate, beta = beta, algorithms="pFedMe_p", batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times,modeltype=modeltype)
    if(algorithm == "pFedditto"):
        average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,learning_rate=learning_rate, beta = beta, algorithms="pFedditto_p", batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times,modeltype=modeltype)
    if(algorithm == "pFedprox"):
        average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,learning_rate=learning_rate, beta = beta, algorithms="pFedprox_p", batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times,modeltype=modeltype)
    if(algorithm == "pSCAFFOLD"):
        average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,learning_rate=learning_rate, beta = beta, algorithms="pSCAFFOLD_p", batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times,modeltype=modeltype)
    if(algorithm == "pFedDyn"):
        average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,learning_rate=learning_rate, beta = beta, algorithms="pFedDyn_p", batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times,modeltype=modeltype)
    if(algorithm == "pFedFBE"):
        average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,learning_rate=learning_rate, beta = beta, algorithms="pFedFBE_p", batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times,modeltype=modeltype)
    if(algorithm == "Fedmirror"):
        average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,learning_rate=learning_rate, beta = beta, algorithms="Fedmirror_p", batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times,modeltype=modeltype)
    if(algorithm == "FedDual"):
        average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,learning_rate=learning_rate, beta = beta, algorithms="FedDual_p", batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times,modeltype=modeltype)
    
    average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,learning_rate=learning_rate, beta = beta, algorithms=algorithm, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times,modeltype=modeltype)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Lasso", choices=["Mnist", "Synthetic", "Cifar10", "Lasso", "Matrix"])
    parser.add_argument("--model", type=str, default="Lasso", choices=["dnn", "mclr", "cnn","Lasso","Matrix"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Local learning rate,0.005, Lasso,Matrix:0.0005")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe (1), or Second learning rate of Per-FedAvg (0.001, 0.0005?)")
    parser.add_argument("--lamda", type=int, default=1000, help="Regularization term for personalization term")
    parser.add_argument("--num_global_iters", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="FedAvg",choices=["pFedMe", "pPerAvg", "FedAvg","pFedditto","pFedprox","pSCAFFOLD","pFedDyn","pFedFBE","Fedmirror","FedDual"]) 
    parser.add_argument("--numusers", type=int, default=10, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=10, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.0005, help="Persionalized learning rate to caculate theta aproximately using K steps, 0.005 (MNIST) , ")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments, -1 mean CPU, 0,1,2,3 for GPU")
    parser.add_argument("--regularizer", type=str, default="l1", help="Which regularizer in the training,none, l1-constraint, l2-constraint, l1-norm, l2-norm, nuclear")
    parser.add_argument("--lamdaCO", type=float, default=0.1, help="Regularization term for Composite optimization, l1:0.01, l2:1")
    parser.add_argument("--l1const", type=int, default=1500, help="Constaint of l1 norm")
    parser.add_argument("--l2const", type=int, default=7, help="Constaint of l2 norm")
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.numusers))
    print("lamda      : {}".format(args.lamda))
    print("lamda for CO      : {}".format(args.lamdaCO))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("=" * 80)

    main(
        dataset=args.dataset,
        algorithm = args.algorithm,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta = args.beta, 
        lamda = args.lamda,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer= args.optimizer,
        numusers = args.numusers,
        K=args.K,
        personal_learning_rate=args.personal_learning_rate,
        times = args.times,
        gpu=args.gpu,
        regularizer=args.regularizer,
        lamdaCO=args.lamdaCO,
        l1const=args.l1const,
        l2const=args.l2const
        )
