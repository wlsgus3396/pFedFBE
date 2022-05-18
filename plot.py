import matplotlib.pyplot as plt
import h5py
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.ticker import StrMethodFormatter
from FLAlgorithms.servers.serverpFedprox import pFedprox
from utils.plot_utils import *
import os
import argparse
plt.rcParams.update({'font.size': 14})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Lasso", choices=["Mnist", "Synthetic", "Cifar10", "Lasso", "Matrix"])
    parser.add_argument("--model", type=str, default="Lasso", choices=["dnn", "mclr", "cnn","Lasso","Matrix"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Local learning rate,0.005, Lasso,Matrix:0.0005")
    parser.add_argument("--beta", type=float, default=1, help="Average moving parameter for pFedMe (1), or Second learning rate of Per-FedAvg (0.001, 0.0005?)")
    parser.add_argument("--lamda", type=int, default=200, help="Regularization term for personalization term")
    parser.add_argument("--num_global_iters", type=int, default=2)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="pFedFBE",choices=["pFedMe", "pPerAvg", "FedAvg","pFedditto","pFedprox","pSCAFFOLD","pFedDyn","pFedFBE","Fedmirror","FedDual"])
    parser.add_argument("--numusers", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=10, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.0005, help="Persionalized learning rate to caculate theta aproximately using K steps, 0.005 (MNIST) , ")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments, -1 mean CPU, 0,1,2,3 for GPU")
    parser.add_argument("--regularizer", type=str, default="l2", help="Which regularizer in the training,none, l1-constraint, l2-constraint, l1-norm, l2-norm, nuclear")
    parser.add_argument("--lamdaCO", type=int, default=1, help="Regularization term for Composite optimization, l1:0.01, l2:1")
    parser.add_argument("--l1const", type=int, default=1500, help="Constaint of l1 norm")
    parser.add_argument("--l2const", type=int, default=7, help="Constaint of l2 norm")
    args = parser.parse_args()

    num_users = 10
    loc_ep1 = 20
    Numb_Glob_Iters = 100
    lamb = 200
    lamdaCO=0.3
    learning_rate = [0.0005,0.0005,0.0005,0.0005]
    beta = 1
    #algorithms_list = ['pFedFBE', 'FedDual','Fedmirror', 'pFedprox', 'pSCAFFOLD', 'pFedditto','pFedMe', 'FedAvg' ]
    #algorithms_list = ['pFedFBE', 'Fedmirror', 'FedAvg' ]
    algorithms_list = ['FedAvg', 'pFedFBE', 'Fedmirror', 'FedDual' ]
    
    batch_size = 20
    dataset = "Lasso"
    k = []
    personal_learning_rate = 0.0005
    modeltype = "Lasso"

    plot_summary_one_figure_Lasso(num_users,loc_ep1,Numb_Glob_Iters,lamb,learning_rate,beta,algorithms_list,batch_size,dataset,k,personal_learning_rate,modeltype)
