import json
import math
import numpy as np
import os
import sys
import random
from tqdm import trange
import math


NUM_USER = 64
np.random.seed(0)
def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex


def generate_Matrix_niid(d1, d0):

    samples_per_user = np.random.choice(np.arange(32,256), NUM_USER)
    
    for i in range(NUM_USER):
        samples_per_user[i]=150
    
    print(samples_per_user)
    num_samples = np.sum(samples_per_user)

    X_split = [[] for _ in range(NUM_USER)]
    y_split = [[] for _ in range(NUM_USER)]
    W_split = [[] for _ in range(NUM_USER)]

    #### define some eprior ####
    #W=np.zeros((1024,1024))
    #for i in range(d0):
    # W[i][i]=1
    
    b=np.random.normal(0, 1, 1)
    
    
    
    
    
    
    mean_x = np.random.normal(0,1,(NUM_USER, 32,32))

        

    for i in range(NUM_USER):
        xx = np.random.normal(0,1,(samples_per_user[i],32,32))

        for j in range(samples_per_user[i]):

            xx[j] += mean_x[i]
            
        w=np.zeros(32-d1)
        w[np.random.choice(32-d1,d0,replace=False)]=1
        ww= np.diag(np.concatenate((np.ones(d1),np.zeros(32-d1)), axis=None)+np.concatenate((np.zeros(d1),w), axis=None))
        yy = np.zeros(samples_per_user[i])
        W_mul=[ww for _ in range(samples_per_user[i])]
        W_mul=np.array(W_mul)
        for j in range(samples_per_user[i]):
            tmp = np.sum(xx[j]*ww) + b
            yy[j] = tmp+np.random.normal(0, 1, 1)

        X_split[i] = xx.tolist()
        y_split[i] = yy.tolist()
        W_split[i]= W_mul.tolist()
        print("{}-th users has {} exampls".format(i, len(y_split[i])))


    return X_split, y_split, W_split



def main():


    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

    train_path = "./data/train/mytrain.json"
    test_path = "./data/test/mytest.json"    
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    #X, y = generate_synthetic(alpha=0, beta=0, iid=0)     # synthetiv (0,0)
    X, y,W = generate_Matrix_niid(2,2) # synthetic (0.5, 0.5)
    #X, y = generate_synthetic(alpha=1, beta=1, iid=0)     # synthetic (1,1)
    #X, y = generate_synthetic(alpha=0, beta=0, iid=1)      # synthetic_IID


    # Create data structure
    train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    
    for i in trange(NUM_USER, ncols=120):

        uname = 'f_{0:05d}'.format(i)        
        combined = list(zip(X[i], y[i], W[i]))
        random.shuffle(combined)
        X[i][:], y[i][:], W[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.9 * num_samples)
        test_len = num_samples - train_len
        
        train_data['users'].append(uname) 
        train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len], 'W': W[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:], 'W': W[i][:train_len]}
        test_data['num_samples'].append(test_len)
    

    with open(train_path,'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)


if __name__ == "__main__":
    main()

