import matplotlib.pyplot as plt
import h5py
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.ticker import StrMethodFormatter
import os
plt.rcParams.update({'font.size': 14})

def simple_read_data(alg,modeltype):
    print(alg)
    if modeltype=="Lasso":
        hf = h5py.File("./results/"+'{}.h5'.format(alg), 'r')
        rs_glob_accuracy = np.array(hf.get('rs_glob_accuracy')[:])
        rs_glob_precision = np.array(hf.get('rs_glob_precision')[:])
        rs_glob_recall = np.array(hf.get('rs_glob_recall')[:])
        rs_glob_F1 = np.array(hf.get('rs_glob_F1')[:])
        rs_glob_density = np.array(hf.get('rs_glob_density')[:])
        rs_train_accuracy = np.array(hf.get('rs_train_accuracy')[:])
        rs_train_precision = np.array(hf.get('rs_train_precision')[:])
        rs_train_recall = np.array(hf.get('rs_train_recall')[:])
        rs_train_F1 = np.array(hf.get('rs_train_F1')[:])
        rs_train_density = np.array(hf.get('rs_train_density')[:])



        rs_train_loss = np.array(hf.get('rs_train_loss')[:])
        return rs_glob_accuracy,rs_glob_precision,rs_glob_recall,rs_glob_F1,rs_glob_density,rs_train_accuracy,rs_train_precision,rs_train_recall,rs_train_F1,rs_train_density,rs_train_loss
    elif modeltype=="Matrix":
        hf = h5py.File("./results/"+'{}.h5'.format(alg), 'r')

        rs_glob_mse = np.array(hf.get('rs_glob_mse')[:])
        rs_glob_rerank = np.array(hf.get('rs_glob_rerank')[:])
        rs_glob_reerror = np.array(hf.get('rs_glob_reerror')[:])
        rs_train_mse = np.array(hf.get('rs_train_mse')[:])
        rs_train_rerank = np.array(hf.get('rs_train_rerank')[:])
        rs_train_reerror = np.array(hf.get('rs_train_reerror')[:])


        rs_train_loss = np.array(hf.get('rs_train_loss')[:])
        return rs_glob_mse,rs_glob_rerank,rs_glob_reerror, rs_train_mse, rs_train_rerank ,rs_train_reerror ,rs_train_loss
    else:
        hf = h5py.File("./results/"+'{}.h5'.format(alg), 'r')
        rs_glob_acc = np.array(hf.get('rs_glob_acc')[:])
        rs_train_acc = np.array(hf.get('rs_train_acc')[:])
        rs_train_loss = np.array(hf.get('rs_train_loss')[:])
        return rs_train_acc, rs_train_loss, rs_glob_acc

def get_training_data_value(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[],beta=[],algorithms_list=[], batch_size=[], dataset="", k= [] , personal_learning_rate = [],modeltype="DNN"):
    Numb_Algs = len(algorithms_list)
    if type(learning_rate[0]) == float or type(learning_rate[0]) == int :
        X=[]
        for i in range(Numb_Algs):
            X.append(learning_rate[i])
        learning_rate=X
    if type(lamb) == float or type(lamb) == int:
        X=[]
        for _ in range(Numb_Algs):
            X.append(lamb)
        lamb=X
    if type(beta) == float or type(beta) == int:
        X=[]
        for _ in range(Numb_Algs):
            X.append(beta)
        beta=X
    if type(batch_size) == float or type(batch_size) == int:
        X=[]
        for _ in range(Numb_Algs):
            X.append(batch_size)
        batch_size=X
    if type(loc_ep1) == float or type(loc_ep1) == int:
        X=[]
        for _ in range(Numb_Algs):
            X.append(loc_ep1)
        loc_ep1=X



    if modeltype=="Lasso":
        Numb_Algs = len(algorithms_list)
        glob_accuracy = np.zeros((Numb_Algs, Numb_Glob_Iters))
        glob_precision = np.zeros((Numb_Algs, Numb_Glob_Iters))
        glob_recall = np.zeros((Numb_Algs, Numb_Glob_Iters))
        glob_F1 = np.zeros((Numb_Algs, Numb_Glob_Iters))
        glob_density = np.zeros((Numb_Algs, Numb_Glob_Iters))
        train_accuracy = np.zeros((Numb_Algs, Numb_Glob_Iters))
        train_precision = np.zeros((Numb_Algs, Numb_Glob_Iters))
        train_recall = np.zeros((Numb_Algs, Numb_Glob_Iters))
        train_F1 = np.zeros((Numb_Algs, Numb_Glob_Iters))
        train_density = np.zeros((Numb_Algs, Numb_Glob_Iters))
        train_loss = np.zeros((Numb_Algs, Numb_Glob_Iters))

        algs_lbl = algorithms_list.copy()
        for i in range(Numb_Algs):
            string_learning_rate = str(learning_rate[i])
            string_learning_rate = string_learning_rate + "_" +str(float(beta[i])) 
            #if(algorithms_list[i] == "pFedMe" or algorithms_list[i] == "pFedMe_p"):
            #    algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(batch_size[i]) + "b" + "_" +str(loc_ep1[i]) + "_"+ str(k[i])  + "_"+ str(personal_learning_rate[i])
            #else:
            algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(batch_size[i]) + "b"  "_" +str(loc_ep1[i])

            glob_accuracy[i, :], glob_precision[i, :], glob_recall[i, :], glob_F1[i, :], glob_density[i, :],train_accuracy[i, :], train_precision[i, :], train_recall[i, :], train_F1[i, :], train_density[i, :], train_loss[i, :] = np.array(
                simple_read_data(dataset +"_"+ algorithms_list[i] + "_avg",modeltype))[:, :Numb_Glob_Iters]
            algs_lbl[i] = algs_lbl[i]
        #return glob_accuracy, train_accuracy, train_loss
        return glob_accuracy,glob_precision,glob_recall,glob_F1,glob_density,train_accuracy,train_precision,train_recall,train_F1,train_density,train_loss



    elif modeltype=="Matrix":
        Numb_Algs = len(algorithms_list)
        glob_mse = np.zeros((Numb_Algs, Numb_Glob_Iters))
        glob_rerank = np.zeros((Numb_Algs, Numb_Glob_Iters))
        glob_reerror = np.zeros((Numb_Algs, Numb_Glob_Iters))

        train_mse = np.zeros((Numb_Algs, Numb_Glob_Iters))
        train_rerank = np.zeros((Numb_Algs, Numb_Glob_Iters))
        train_reerror = np.zeros((Numb_Algs, Numb_Glob_Iters))

        train_loss = np.zeros((Numb_Algs, Numb_Glob_Iters))

        algs_lbl = algorithms_list.copy()
        for i in range(Numb_Algs):
            string_learning_rate = str(learning_rate[i])
            string_learning_rate = string_learning_rate + "_" +str(beta[i]) 
            #if(algorithms_list[i] == "pFedMe" or algorithms_list[i] == "pFedMe_p"):
            #    algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(batch_size[i]) + "b" + "_" +str(loc_ep1[i]) + "_"+ str(k[i])  + "_"+ str(personal_learning_rate[i])
            #else:
            algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(batch_size[i]) + "b"  "_" +str(loc_ep1[i])

            glob_mse[i, :], glob_rerank[i, :], glob_reerror[i, :],train_mse[i, :], train_rerank[i, :], train_reerror[i, :], train_loss[i, :] = np.array(
                simple_read_data(dataset +"_"+ algorithms_list[i] + "_avg",modeltype))[:, :Numb_Glob_Iters]
            algs_lbl[i] = algs_lbl[i]
        return  glob_mse,glob_rerank,glob_reerror,train_mse,train_rerank,train_reerror,train_loss



    else:
        Numb_Algs = len(algorithms_list)
        train_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
        train_loss = np.zeros((Numb_Algs, Numb_Glob_Iters))
        glob_acc = np.zeros((Numb_Algs, Numb_Glob_Iters))
        algs_lbl = algorithms_list.copy()
        for i in range(Numb_Algs):
            string_learning_rate = str(learning_rate[i])
            string_learning_rate = string_learning_rate + "_" +str(float(beta[i])) 
            #if(algorithms_list[i] == "pFedMe" or algorithms_list[i] == "pFedMe_p"):
            #    algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(batch_size[i]) + "b" + "_" +str(loc_ep1[i]) + "_"+ str(k[i])  + "_"+ str(personal_learning_rate[i])
            #else:
            algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(batch_size[i]) + "b"  "_" +str(loc_ep1[i])

            train_acc[i, :], train_loss[i, :], glob_acc[i, :] = np.array(
                simple_read_data(dataset +"_"+ algorithms_list[i] + "_avg",modeltype))[:, :Numb_Glob_Iters]
            algs_lbl[i] = algs_lbl[i]
        return glob_acc, train_acc, train_loss

def get_all_training_data_value(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=0, learning_rate=0,beta=0,algorithms="", batch_size=0, dataset="", k= 0 , personal_learning_rate =0 ,times = 5,modeltype ="DNN"):
    if type(learning_rate) == float or type(learning_rate) == int :
        X=[]
        for _ in range(times):
            X.append(learning_rate)
        learning_rate=X
    if type(lamb) == float or type(lamb) == int:
        X=[]
        for _ in range(times):
            X.append(lamb)
        lamb=X
    if type(beta) == float or type(beta) == int:
        X=[]
        for _ in range(times):
            X.append(beta)
        beta=X
    if type(batch_size) == float or type(batch_size) == int:
        X=[]
        for _ in range(times):
            X.append(batch_size)
        batch_size=X
    if type(loc_ep1) == float or type(loc_ep1) == int:
        X=[]
        for _ in range(times):
            X.append(loc_ep1)
        loc_ep1=X


    if modeltype=="Lasso":

        glob_accuracy = np.zeros((times, Numb_Glob_Iters))
        glob_precision = np.zeros((times, Numb_Glob_Iters))
        glob_recall = np.zeros((times, Numb_Glob_Iters))
        glob_F1 = np.zeros((times, Numb_Glob_Iters))
        glob_density = np.zeros((times, Numb_Glob_Iters))
        train_accuracy = np.zeros((times, Numb_Glob_Iters))
        train_precision = np.zeros((times, Numb_Glob_Iters))
        train_recall = np.zeros((times, Numb_Glob_Iters))
        train_F1 = np.zeros((times, Numb_Glob_Iters))
        train_density = np.zeros((times, Numb_Glob_Iters))
        train_loss = np.zeros((times, Numb_Glob_Iters))

        algorithms_list  = [algorithms] * times
        for i in range(times):
            string_learning_rate = str(learning_rate[i])
            string_learning_rate = string_learning_rate + "_" +str(beta[i]) 
            #if(algorithms_list[i] == "pFedMe" or algorithms_list[i] == "pFedMe_p"):
            #    algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(batch_size[i]) + "b" + "_" +str(loc_ep1[i]) + "_"+ str(k[i])  + "_"+ str(personal_learning_rate[i])
            #else:
            algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(batch_size[i]) + "b"  "_" +str(loc_ep1[i])+  "_" +str(i)

            glob_accuracy[i, :], glob_precision[i, :], glob_recall[i, :], glob_F1[i, :], glob_density[i, :],train_accuracy[i, :], train_precision[i, :], train_recall[i, :], train_F1[i, :], train_density[i, :], train_loss[i, :] = np.array(
                simple_read_data(dataset +"_"+ algorithms_list[i],modeltype))[:, :Numb_Glob_Iters]

        return  glob_accuracy,glob_precision,glob_recall,glob_F1,glob_density,train_accuracy,train_precision,train_recall,train_F1,train_density,train_loss



    elif modeltype=="Matrix":

        glob_mse = np.zeros((times, Numb_Glob_Iters))
        glob_rerank = np.zeros((times, Numb_Glob_Iters))
        glob_reerror = np.zeros((times, Numb_Glob_Iters))

        train_mse = np.zeros((times, Numb_Glob_Iters))
        train_rerank = np.zeros((times, Numb_Glob_Iters))
        train_reerror = np.zeros((times, Numb_Glob_Iters))

        train_loss = np.zeros((times, Numb_Glob_Iters))

        algorithms_list  = [algorithms] * times
        for i in range(times):
            string_learning_rate = str(learning_rate[i])
            string_learning_rate = string_learning_rate + "_" +str(beta[i]) 
            #if(algorithms_list[i] == "pFedMe" or algorithms_list[i] == "pFedMe_p"):
            #    algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(batch_size[i]) + "b" + "_" +str(loc_ep1[i]) + "_"+ str(k[i])  + "_"+ str(personal_learning_rate[i])
            #else:
            algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(batch_size[i]) + "b"  "_" +str(loc_ep1[i])+  "_" +str(i)

            glob_mse[i, :], glob_rerank[i, :], glob_reerror[i, :],train_mse[i, :], train_rerank[i, :], train_reerror[i, :], train_loss[i, :] = np.array(
                simple_read_data(dataset +"_"+ algorithms_list[i],modeltype))[:, :Numb_Glob_Iters]

        return  glob_mse,glob_rerank,glob_reerror,train_mse,train_rerank,train_reerror,train_loss



    else:

        train_acc = np.zeros((times, Numb_Glob_Iters))
        train_loss = np.zeros((times, Numb_Glob_Iters))
        glob_acc = np.zeros((times, Numb_Glob_Iters))
        algorithms_list  = [algorithms] * times
        for i in range(times):
            string_learning_rate = str(learning_rate[i])
            string_learning_rate = string_learning_rate + "_" +str(beta[i]) 
            #if(algorithms_list[i] == "pFedMe" or algorithms_list[i] == "pFedMe_p"):
            #    algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(batch_size[i]) + "b" + "_" +str(loc_ep1[i]) + "_"+ str(k[i])  + "_"+ str(personal_learning_rate[i])
            #else:
            algorithms_list[i] = algorithms_list[i] + "_" + string_learning_rate + "_" + str(num_users) + "u" + "_" + str(batch_size[i]) + "b"  "_" +str(loc_ep1[i])+  "_" +str(i)

            train_acc[i, :], train_loss[i, :], glob_acc[i, :] = np.array(
                simple_read_data(dataset +"_"+ algorithms_list[i],modeltype))[:, :Numb_Glob_Iters]

        return glob_acc, train_acc, train_loss




def get_data_label_style(input_data = [], linestyles= [], algs_lbl = [], lamb = [], loc_ep1 = 0, batch_size =0):
    data, lstyles, labels = [], [], []
    if type(learning_rate) == float or type(learning_rate) == int :
        X=[]
        for _ in range(len(algs_lbl)):
            X.append(learning_rate)
        learning_rate=X
    if type(lamb) == float or type(lamb) == int:
        X=[]
        for _ in range(len(algs_lbl)):
            X.append(lamb)
        lamb=X
    if type(beta) == float or type(beta) == int:
        X=[]
        for _ in range(len(algs_lbl)):
            X.append(beta)
        beta=X
    if type(batch_size) == float or type(batch_size) == int:
        X=[]
        for _ in range(len(algs_lbl)):
            X.append(batch_size)
        batch_size=X
    if type(loc_ep1) == float or type(loc_ep1) == int:
        X=[]
        for _ in range(len(algs_lbl)):
            X.append(loc_ep1)
        loc_ep1=X

    for i in range(len(algs_lbl)):
        data.append(input_data[i, ::])
        lstyles.append(linestyles[i])
        labels.append(algs_lbl[i]+str(lamb[i])+"_" +
                      str(loc_ep1[i])+"e" + "_" + str(batch_size[i]) + "b")

    return data, lstyles, labels

def average_data(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb="", learning_rate="", beta="", algorithms="", batch_size=0, dataset = "", k = "", personal_learning_rate = "", times = 5, modeltype="DNN"):
    if modeltype=="Lasso":
        glob_accuracy,glob_precision,glob_recall,glob_F1,glob_density,train_accuracy,train_precision,train_recall,train_F1,train_density,train_loss = get_all_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms, batch_size, dataset, k, personal_learning_rate,times,modeltype)
        glob_accuracy_data = np.average(glob_accuracy, axis=0)
        glob_precision_data = np.average(glob_precision, axis=0)
        glob_recall_data = np.average(glob_recall, axis=0)
        glob_F1_data = np.average(glob_F1, axis=0)
        glob_density_data = np.average(glob_density, axis=0)

        train_accuracy_data = np.average(train_accuracy, axis=0)
        train_precision_data = np.average(train_precision, axis=0)
        train_recall_data = np.average(train_recall, axis=0)
        train_F1_data = np.average(train_F1, axis=0)
        train_density_data = np.average(train_density, axis=0)
        train_loss_data = np.average(train_loss, axis=0)
        # store average value to h5 file
        max_accuracy = []
        for i in range(times):
            max_accuracy.append(glob_accuracy[i].max())

        print("std:", np.std(max_accuracy))
        print("Mean:", np.mean(max_accuracy))

        alg = dataset + "_" + algorithms
        alg = alg + "_" + str(learning_rate) + "_" + str(float(beta))  + "_" + str(num_users) + "u" + "_" + str(batch_size) + "b" + "_" + str(loc_ep1)
        #if(algorithms == "pFedMe" or algorithms == "pFedMe_p"):
        #    alg = alg + "_" + str(k) + "_" + str(personal_learning_rate)
        alg = alg + "_" + "avg"
        if (len(glob_accuracy) != 0 &  len(train_accuracy) & len(train_loss)) :
            with h5py.File("./results/"+'{}.h5'.format(alg,loc_ep1), 'w') as hf:
                hf.create_dataset('rs_glob_accuracy', data=glob_accuracy_data)
                hf.create_dataset('rs_glob_precision', data=glob_precision_data)
                hf.create_dataset('rs_glob_recall', data=glob_recall_data)
                hf.create_dataset('rs_glob_F1', data=glob_F1_data)
                hf.create_dataset('rs_glob_density', data=glob_density_data)

                hf.create_dataset('rs_train_accuracy', data=train_accuracy_data)
                hf.create_dataset('rs_train_precision', data=train_precision_data)
                hf.create_dataset('rs_train_recall', data=train_recall_data)
                hf.create_dataset('rs_train_F1', data=train_F1_data)
                hf.create_dataset('rs_train_density', data=train_density_data)

                hf.create_dataset('rs_train_loss', data=train_loss_data)

                hf.close()



    elif modeltype=="Matrix":
        glob_mse,glob_rerank,glob_reerror,train_mse,train_rerank,train_reerror,train_loss = get_all_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms, batch_size, dataset, k, personal_learning_rate,times,modeltype)
        glob_mse_data = np.average(glob_mse, axis=0)
        glob_rerank_data = np.average(glob_rerank, axis=0)
        glob_reerror_data = np.average(glob_reerror, axis=0)


        train_mse_data = np.average(train_mse, axis=0)
        train_rerank_data = np.average(train_rerank, axis=0)
        train_reerror_data = np.average(train_reerror, axis=0)


        train_loss_data = np.average(train_loss, axis=0)
        # store average value to h5 file
        max_mse = []
        for i in range(times):
            max_mse.append(glob_mse[i].max())

        print("std:", np.std(max_mse))
        print("Mean:", np.mean(max_mse))

        alg = dataset + "_" + algorithms
        alg = alg + "_" + str(learning_rate) + "_" + str(beta)  + "_" + str(num_users) + "u" + "_" + str(batch_size) + "b" + "_" + str(loc_ep1)
        #if(algorithms == "pFedMe" or algorithms == "pFedMe_p"):
        #    alg = alg + "_" + str(k) + "_" + str(personal_learning_rate)
        alg = alg + "_" + "avg"
        if (len(glob_mse) != 0 &  len(train_mse) & len(train_loss)) :
            with h5py.File("./results/"+'{}.h5'.format(alg,loc_ep1), 'w') as hf:
                hf.create_dataset('rs_glob_mse', data=glob_mse_data)
                hf.create_dataset('rs_glob_rerank', data=glob_rerank_data)
                hf.create_dataset('rs_glob_reerror', data=glob_reerror_data)


                hf.create_dataset('rs_train_mse', data=train_mse_data)
                hf.create_dataset('rs_train_rerank', data=train_rerank_data)
                hf.create_dataset('rs_train_reerror', data=train_reerror_data)


                hf.create_dataset('rs_train_loss', data=train_loss_data)

                hf.close()
    else:
        glob_acc, train_acc, train_loss = get_all_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms, batch_size, dataset, k, personal_learning_rate,times,modeltype)
        glob_acc_data = np.average(glob_acc, axis=0)
        train_acc_data = np.average(train_acc, axis=0)
        train_loss_data = np.average(train_loss, axis=0)
        # store average value to h5 file
        max_accurancy = []
        for i in range(times):
            max_accurancy.append(glob_acc[i].max())

        print("std:", np.std(max_accurancy))
        print("Mean:", np.mean(max_accurancy))

        alg = dataset + "_" + algorithms
        alg = alg + "_" + str(learning_rate) + "_" + str(beta)  + "_" + str(num_users) + "u" + "_" + str(batch_size) + "b" + "_" + str(loc_ep1)
        #if(algorithms == "pFedMe" or algorithms == "pFedMe_p"):
        #    alg = alg + "_" + str(k) + "_" + str(personal_learning_rate)
        alg = alg + "_" + "avg"
        if (len(glob_acc) != 0 &  len(train_acc) & len(train_loss)) :
            with h5py.File("./results/"+'{}.h5'.format(alg,loc_ep1), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=glob_acc_data)
                hf.create_dataset('rs_train_acc', data=train_acc_data)
                hf.create_dataset('rs_train_loss', data=train_loss_data)
                hf.close()

def plot_summary_one_figure(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], beta=[], algorithms_list=[], batch_size=0, dataset = "", k = [], personal_learning_rate = [], modeltype="DNN"):
    Numb_Algs = len(algorithms_list)
    algorithms_list_p = [i + "_p" for i in algorithms_list]
    dataset = dataset
    glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate,modeltype)

    glob_acc =  average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')

    # load personal performance
    dataset = dataset
    glob_acc_p, train_acc_p, train_loss_p = get_training_data_value(num_users, loc_ep1, Numb_Glob_Iters, lamb,
                                                                 learning_rate, beta, algorithms_list_p, batch_size,
                                                                 dataset, k, personal_learning_rate, modeltype)

    glob_accp = average_smooth(glob_acc_p, window='flat')
    train_lossp = average_smooth(train_loss_p, window='flat')
    train_accp = average_smooth(train_acc_p, window='flat')

    #glob_acc, train_acc, train_loss = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate )
    print("max value of test accurancy",glob_acc.max())
    plt.figure(1,figsize=(5, 5))
    plt.grid(True)
    MIN = train_loss.min() - 0.001
    start = 0
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    for i in range(Numb_Algs):
        plt.plot(train_accp[i, 1:], linestyle=linestyles[i], label=algorithms_list_p[i] )
    #plt.plot(train_accp[0, 1:], linestyle=linestyles[0], label=algorithms_list_p[0])##################################
    #for i in range(Numb_Algs):
    #    plt.plot(train_accp[i, 1:], linestyle=linestyles[i], label=algorithms_list_p[i] )
    plt.legend(loc='lower right')
    plt.ylabel('Training Accuracy')
    plt.xlabel('Global rounds ')
    plt.title(dataset.upper())
    #plt.ylim([0.8, glob_acc.max()])
    plt.savefig(dataset.upper() + str(loc_ep1) + 'train_acc(p).png', bbox_inches="tight")
    #plt.savefig(dataset + str(loc_ep1[1]) + 'train_acc.pdf')



    plt.figure(2,figsize=(5, 5))
    plt.grid(True)
    MIN = train_loss.min() - 0.001
    start = 0
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    for i in range(Numb_Algs):
        plt.plot(train_acc[i, 1:], linestyle=linestyles[i], label=algorithms_list[i] )
    #plt.plot(train_accp[0, 1:], linestyle=linestyles[0], label=algorithms_list_p[0])##################################
    #for i in range(Numb_Algs):
    #    plt.plot(train_accp[i, 1:], linestyle=linestyles[i], label=algorithms_list_p[i] )
    plt.legend(loc='lower right')
    plt.ylabel('Training Accuracy')
    plt.xlabel('Global rounds ')
    plt.title(dataset.upper())
    #plt.ylim([0.8, glob_acc.max()])
    plt.savefig(dataset.upper() + str(loc_ep1) + 'train_acc.png', bbox_inches="tight")
    #plt.savefig(dataset + str(loc_ep1[1]) + 'train_acc.pdf')



    plt.figure(3,figsize=(5, 5))
    plt.grid(True)
    MIN = train_loss.min() - 0.001
    start = 0
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    for i in range(Numb_Algs):
        plt.plot(glob_accp[i, 1:], linestyle=linestyles[i], label=algorithms_list_p[i] )
    #plt.plot(train_accp[0, 1:], linestyle=linestyles[0], label=algorithms_list_p[0])##################################
    #for i in range(Numb_Algs):
    #    plt.plot(train_accp[i, 1:], linestyle=linestyles[i], label=algorithms_list_p[i] )
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds ')
    plt.title(dataset.upper())
    #plt.ylim([0.8, glob_acc.max()])
    plt.savefig(dataset.upper() + str(loc_ep1) + 'test_acc(p).png', bbox_inches="tight")
    #plt.savefig(dataset + str(loc_ep1[1]) + 'train_acc.pdf')



    plt.figure(4,figsize=(5, 5))
    plt.grid(True)
    MIN = train_loss.min() - 0.001
    start = 0
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    for i in range(Numb_Algs):
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=algorithms_list[i] )
    #plt.plot(train_accp[0, 1:], linestyle=linestyles[0], label=algorithms_list_p[0])##################################
    #for i in range(Numb_Algs):
    #    plt.plot(train_accp[i, 1:], linestyle=linestyles[i], label=algorithms_list_p[i] )
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds ')
    plt.title(dataset.upper())
    #plt.ylim([0.8, glob_acc.max()])
    plt.savefig(dataset.upper() + str(loc_ep1) + 'test_acc.png', bbox_inches="tight")
    #plt.savefig(dataset + str(loc_ep1[1]) + 'train_acc.pdf')





    plt.figure(5,figsize=(5, 5))
    plt.grid(True)
    for i in range(Numb_Algs):
        plt.plot(train_loss[i, start:], linestyle=linestyles[i], label=algorithms_list[i] )
        #plt.plot(train_loss1[i, 1:], label=algs_lbl1[i])
    #plt.plot(train_lossp[0, start:], linestyle=linestyles[0], label=algorithms_list_p[0])
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    plt.title(dataset.upper())
    #plt.ylim([train_loss.min(), 0.5])
    plt.savefig(dataset.upper() + str(loc_ep1) + 'train_loss.png', bbox_inches="tight")
    #plt.savefig(dataset + str(loc_ep1[1]) + 'train_loss.pdf')




    plt.figure(6,figsize=(5, 5))
    plt.grid(True)
    for i in range(Numb_Algs):
        plt.plot(train_lossp[i, start:], linestyle=linestyles[i], label=algorithms_list_p[i] )
        #plt.plot(train_loss1[i, 1:], label=algs_lbl1[i])
    #plt.plot(train_lossp[0, start:], linestyle=linestyles[0], label=algorithms_list_p[0])
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    plt.title(dataset.upper())
    #plt.ylim([train_loss.min(), 0.5])
    plt.savefig(dataset.upper() + str(loc_ep1) + 'train_loss(p).png', bbox_inches="tight")
    #plt.savefig(dataset + str(loc_ep1[1]) + 'train_loss.pdf')



def plot_summary_one_figure_Lasso(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], beta=[], algorithms_list=[], batch_size=0, dataset = "", k = [], personal_learning_rate = [], modeltype="DNN"):
    Numb_Algs = len(algorithms_list)
    algorithms_list_p = [i + "_p" for i in algorithms_list]
    #dataset = dataset
    #glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate,modeltype)

    #glob_acc =  average_smooth(glob_acc_, window='flat')
    #train_loss = average_smooth(train_loss_, window='flat')
    #train_acc = average_smooth(train_acc_, window='flat')

    # load personal performance
    dataset = dataset
    glob_accuracy_p,glob_precision_p,glob_recall_p,glob_F1_p,glob_density_p,train_accuracy_p,train_precision_p,train_recall_p,train_F1_p,train_density_p,train_loss_p = get_training_data_value(num_users, loc_ep1, Numb_Glob_Iters, lamb,
                                                                 learning_rate, beta, algorithms_list_p, batch_size,
                                                                 dataset, k, personal_learning_rate, modeltype)

    glob_accuracy_p = average_smooth(glob_accuracy_p, window='flat')
    glob_preicision_p = average_smooth(glob_precision_p, window='flat')
    glob_recall_p = average_smooth(glob_recall_p, window='flat')
    glob_F1_p = average_smooth(glob_F1_p, window='flat')
    glob_density_p = average_smooth(glob_density_p, window='flat')


    train_accuracy_p = average_smooth(train_accuracy_p, window='flat')
    train_preicision_p = average_smooth(train_precision_p, window='flat')
    train_recall_p = average_smooth(train_recall_p, window='flat')
    train_F1_p = average_smooth(train_F1_p, window='flat')
    train_density_p = average_smooth(train_density_p, window='flat')


    train_loss_p = average_smooth(train_loss_p, window='flat')

    #glob_acc, train_acc, train_loss = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate )
    print("max value of test accurancy",glob_accuracy_p.max())
    plt.figure(1,figsize=(5, 5))
    plt.grid(True)
    MIN = train_loss_p.min() - 0.001
    start = 0
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    for i in range(Numb_Algs):
        plt.plot(glob_precision_p[i, 1:], linestyle=linestyles[i], label=algorithms_list_p[i] )
    #plt.plot(train_accp[0, 1:], linestyle=linestyles[0], label=algorithms_list_p[0])##################################
    #for i in range(Numb_Algs):
    #    plt.plot(train_accp[i, 1:], linestyle=linestyles[i], label=algorithms_list_p[i] )
    plt.legend(loc='lower right')
    plt.ylabel('Test Precision')
    plt.xlabel('Global rounds ')
    plt.title(dataset.upper())
    #plt.ylim([0.8, glob_acc.max()])
    plt.savefig(dataset.upper() + str(loc_ep1) + 'test_precision(p).png', bbox_inches="tight")
    #plt.savefig(dataset + str(loc_ep1[1]) + 'train_acc.pdf')

    plt.figure(2,figsize=(5, 5))
    plt.grid(True)
    MIN = train_loss_p.min() - 0.001
    start = 0
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    for i in range(Numb_Algs):
        plt.plot(glob_recall_p[i, 1:], linestyle=linestyles[i], label=algorithms_list_p[i] )
    #plt.plot(train_accp[0, 1:], linestyle=linestyles[0], label=algorithms_list_p[0])##################################
    #for i in range(Numb_Algs):
    #    plt.plot(train_accp[i, 1:], linestyle=linestyles[i], label=algorithms_list_p[i] )
    plt.legend(loc='lower right')
    plt.ylabel('Test Recall')
    plt.xlabel('Global rounds ')
    plt.title(dataset.upper())
    #plt.ylim([0.8, glob_acc.max()])
    plt.savefig(dataset.upper() + str(loc_ep1) + 'test_recall(p).png', bbox_inches="tight")
    #plt.savefig(dataset + str(loc_ep1[1]) + 'train_acc.pdf')

    plt.figure(3,figsize=(5, 5))
    plt.grid(True)
    MIN = train_loss_p.min() - 0.001
    start = 0
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    for i in range(Numb_Algs):
        plt.plot(glob_F1_p[i, 1:], linestyle=linestyles[i], label=algorithms_list_p[i] )
    #plt.plot(train_accp[0, 1:], linestyle=linestyles[0], label=algorithms_list_p[0])##################################
    #for i in range(Numb_Algs):
    #    plt.plot(train_accp[i, 1:], linestyle=linestyles[i], label=algorithms_list_p[i] )
    plt.legend(loc='lower right')
    plt.ylabel('Test F1')
    plt.xlabel('Global rounds ')
    plt.title(dataset.upper())
    #plt.ylim([0.8, glob_acc.max()])
    plt.savefig(dataset.upper() + str(loc_ep1) + 'test_F1(p).png', bbox_inches="tight")
    #plt.savefig(dataset + str(loc_ep1[1]) + 'train_acc.pdf')


    plt.figure(4,figsize=(5, 5))
    plt.grid(True)
    MIN = train_loss_p.min() - 0.001
    start = 0
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    for i in range(Numb_Algs):
        plt.plot(glob_density_p[i, 1:], linestyle=linestyles[i], label=algorithms_list_p[i] )
    #plt.plot(train_accp[0, 1:], linestyle=linestyles[0], label=algorithms_list_p[0])##################################
    #for i in range(Numb_Algs):
    #    plt.plot(train_accp[i, 1:], linestyle=linestyles[i], label=algorithms_list_p[i] )
    plt.legend(loc='lower right')
    plt.ylabel('Test Density')
    plt.xlabel('Global rounds ')
    plt.title(dataset.upper())
    #plt.ylim([0.8, glob_acc.max()])
    plt.savefig(dataset.upper() + str(loc_ep1) + 'test_density(p).png', bbox_inches="tight")
    #plt.savefig(dataset + str(loc_ep1[1]) + 'train_acc.pdf')


    plt.figure(5,figsize=(5, 5))
    plt.grid(True)
    MIN = train_loss_p.min() - 0.001
    start = 0
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    for i in range(Numb_Algs):
        plt.plot(train_loss_p[i, 1:], linestyle=linestyles[i], label=algorithms_list_p[i] )
    #plt.plot(train_accp[0, 1:], linestyle=linestyles[0], label=algorithms_list_p[0])##################################
    #for i in range(Numb_Algs):
    #    plt.plot(train_accp[i, 1:], linestyle=linestyles[i], label=algorithms_list_p[i] )
    plt.legend(loc='lower right')
    plt.ylabel('Train Loss')
    plt.xlabel('Global rounds ')
    plt.title(dataset.upper())
    #plt.ylim([0.8, glob_acc.max()])
    plt.savefig(dataset.upper() + str(loc_ep1) + 'train_loss(p).png', bbox_inches="tight")
    #plt.savefig(dataset + str(loc_ep1[1]) + 'train_acc.pdf')



    plt.figure(6,figsize=(5, 5))
    plt.grid(True)
    MIN = train_loss_p.min() - 0.001
    start = 0
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    for i in range(Numb_Algs):
        plt.plot(glob_accuracy_p[i, 1:], linestyle=linestyles[i], label=algorithms_list_p[i] )
    #plt.plot(train_accp[0, 1:], linestyle=linestyles[0], label=algorithms_list_p[0])##################################
    #for i in range(Numb_Algs):
    #    plt.plot(train_accp[i, 1:], linestyle=linestyles[i], label=algorithms_list_p[i] )
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds ')
    plt.title(dataset.upper())
    #plt.ylim([0.8, glob_acc.max()])
    plt.savefig(dataset.upper() + str(loc_ep1) + 'test_accuracy(p).png', bbox_inches="tight")
    #plt.savefig(dataset + str(loc_ep1[1]) + 'train_acc.pdf')


























def get_max_value_index(num_users=100, loc_ep1=5, Numb_Glob_Iters=10, lamb=[], learning_rate=[], algorithms_list=[], batch_size=0, dataset="",modeltype="DNN"):
    Numb_Algs = len(algorithms_list)
    glob_acc, train_acc, train_loss = get_training_data_value(
        num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, algorithms_list, batch_size, dataset,modeltype)
    for i in range(Numb_Algs):
        print("Algorithm: ", algorithms_list[i], "Max testing Accurancy: ", glob_acc[i].max(
        ), "Index: ", np.argmax(glob_acc[i]), "local update:", loc_ep1[i])

def get_label_name(name):
    if name.startswith("pFedMe"):
        if name.startswith("pFedMe_p"):
            return "pFedMe"
        else:
            return "pFedMe"
    if name.startswith("PerAvg"):
        if name.startswith("PerAvg_p"):
            return "pPerAvg"
        else:
            return "pPerAvg"
    if name.startswith("FedAvg"):
        if name.startswith("FedAvg_p"):
            return "FedAvg"
        else:
            return "FedAvg"
    if name.startswith("pFedditto"):
        if name.startswith("pFedditto_p"):
            return "pFedditto"
        else:
            return "pFedditto"
    if name.startswith("pFedprox"):
        if name.startswith("pFedprox_p"):
            return "pFedprox"
        else:
            return "pFedprox"
    if name.startswith("SCAFFOLD"):
        if name.startswith("SCAFFOLD_p"):
            return "pSCAFFOLD"
        else:
            return "pSCAFFOLD"
    if name.startswith("FedDyn"):
        if name.startswith("FedDyn_p"):
            return "pFedDyn"
        else:
            return "pFedDyn"
    if name.startswith("pFedFBE"):
        if name.startswith("pFedFBE_p"):
            return "pFedFBE"
        else:
            return "pFedFBE"
    if name.startswith("Fedmirror"):
        if name.startswith("Fedmirror_p"):
            return "Fedmirror"
        else:
            return "Fedmirror"
    if name.startswith("FedDual"):
        if name.startswith("FedDual_p"):
            return "FedDual"
        else:
            return "FedDual"


def average_smooth(data, window_len=20, window='hanning'):
    results = []
    if window_len<3:
        return data
    for i in range(len(data)):
        x = data[i]
        s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        #print(len(s))
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('numpy.'+window+'(window_len)')

        y=np.convolve(w/w.sum(),s,mode='valid')
        results.append(y[window_len-1:])
    return np.array(results)

def plot_summary_one_figure_synthetic_R(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)
    dataset = dataset
    glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate )

    glob_acc =  average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')


    linestyles = ['-','-','-','-.','-.','-.']
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'c', 'darkorange', 'tab:brown', 'w']
    markers = ["o","v","s","*","x","P"]
    plt.figure(1,figsize=(5, 5))
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], label=label + ": "
                 r'$R = $' +str(loc_ep1[i]), linewidth = 1, color=colors[i], marker = markers[i],markevery=0.05, markersize=5)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    plt.ylim([train_loss.min() - 0.01,  2])
    #plt.ylim([0.5,  1.8])
    plt.savefig(dataset.upper() + "Non_Convex_Syn_fixR.pdf", bbox_inches="tight")

    plt.figure(2,figsize=(5, 5))
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], label=label + ": "
                 r'$R = $' +str(loc_ep1[i]), linewidth = 1, color=colors[i], marker = markers[i],markevery=0.05, markersize=5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    plt.ylim([0.6,  0.86])
    #plt.ylim([0.89,  0.945])
    plt.savefig(dataset.upper() + "Non_Convex_Syn_fixR_test.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Convex_Syn_fixR.pdf", bbox_inches="tight")
    plt.close()

def plot_summary_one_figure_synthetic_K(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)
    dataset = dataset

    glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate )

    glob_acc =  average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')

    linestyles = ['-','-','-','-.','-.','-.']
    print(lamb)
    colors = ['tab:blue', 'tab:green','darkorange', 'r', 'c', 'tab:brown', 'w']
    markers = ["o","v","s","*","x","P"]
    plt.figure(1,figsize=(5, 5))
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label + ": "
                 r'$K = $' +str(k[i]), linewidth = 1, color=colors[i], marker = markers[i],markevery=0.05, markersize=5)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    #plt.ylim([0.5,  1.8])
    plt.ylim([train_loss.min() - 0.01,  2])
    #plt.savefig(dataset.upper() + "Non_Convex_Syn_fixK.pdf", bbox_inches="tight")
    plt.savefig(dataset.upper() + "Convex_Syn_fixK.pdf", bbox_inches="tight")
    plt.figure(2,figsize=(5, 5))
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label + ": "
                 r'$K = $' +str(k[i]), linewidth = 1, color=colors[i], marker = markers[i],markevery=0.05, markersize=5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    plt.ylim([0.6,  0.86])
    plt.savefig(dataset.upper() + "Convex_Syn_fixK_test.pdf", bbox_inches="tight")
    plt.close()

def plot_summary_one_figure_synthetic_L(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)
    dataset = dataset

    glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate )

    glob_acc =  average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')

    linestyles = ['-','-','-','-.','-.','-.']
    markers = ["o","v","s","*","x","P"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'c', 'darkorange', 'tab:brown', 'm']
    plt.figure(1,figsize=(5, 5))
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label + ": "
                 + r'$\lambda = $'+ str(lamb[i]), linewidth = 1, color=colors[i],marker = markers[i],markevery=0.05, markersize=5)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    plt.ylim([0.5,  1.8])
    #plt.ylim([train_loss.min() - 0.01,  2])
    #plt.savefig(dataset.upper() + "Non_Convex_Syn_fixL.pdf", bbox_inches="tight")
    plt.savefig(dataset.upper() + "Convex_Syn_fixL.pdf", bbox_inches="tight")
    plt.figure(2,figsize=(5, 5))
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label + ": "
                 + r'$\lambda = $'+ str(lamb[i]), linewidth = 1, color=colors[i],marker = markers[i],markevery=0.05, markersize=5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    plt.ylim([0.6,  0.86])
    plt.savefig(dataset.upper() + "Convex_Syn_fixL_test.pdf", bbox_inches="tight")
    plt.close()

def plot_summary_one_figure_synthetic_D(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)
    dataset = dataset

    glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate )

    glob_acc =  average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')

    linestyles = ['-','-','-','-.','-.','-.']
    markers = ["o","v","s","*","x","P"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'c', 'darkorange', 'tab:brown', 'm']
    plt.figure(1,figsize=(5, 5))
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label + ": "
                 + r'$\lambda = $'+ str(lamb[i]), linewidth = 1, color=colors[i],marker = markers[i],markevery=0.05, markersize=5)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    plt.ylim([0.5,  1.8])
    #plt.ylim([train_loss.min() - 0.01,  2])
    #plt.savefig(dataset.upper() + "Non_Convex_Syn_fixL.pdf", bbox_inches="tight")
    plt.savefig(dataset.upper() + "Convex_Syn_fixL.pdf", bbox_inches="tight")
    plt.figure(2,figsize=(5, 5))
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label + ": "
                 + r'$\lambda = $'+ str(lamb[i]), linewidth = 1, color=colors[i],marker = markers[i],markevery=0.05, markersize=5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    plt.ylim([0.6,  0.86])
    plt.savefig(dataset.upper() + "Convex_Syn_fixL_test.pdf", bbox_inches="tight")
    plt.close()

def plot_summary_one_figure_synthetic_Compare(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)
    dataset = dataset
    glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate )
    for i in range(Numb_Algs):
        print("max accurancy:", train_acc_[i].max())
    glob_acc =  average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')
    linestyles = ['-', '--', '-.','-', '--', '-.']
    linestyles = ['-','-','-','-','-','-','-']
    #linestyles = ['-','-','-','-','-','-','-']
    markers = ["o","v","s","*","x","P"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 'tab:brown', 'm']
    plt.figure(1,figsize=(5, 5))
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    plt.grid(True)
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label, linewidth = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    plt.ylim([0.5,  1.8]) # convex
    #plt.ylim([0.4,  1.8]) # non convex
    #plt.ylim([train_loss.min() - 0.01,  2])
    #plt.savefig(dataset.upper() + "Non_Convex_Syn_train_Com.pdf", bbox_inches="tight")
    plt.savefig(dataset.upper() + "Convex_Syn_train_Com.pdf", bbox_inches="tight")
    plt.figure(2,figsize=(5, 5))
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    plt.grid(True)
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i],label=label, linewidth = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    plt.ylim([0.5,  0.86]) # convex
    #plt.savefig(dataset.upper() + "Non_Convex_Syn_test_Com.pdf", bbox_inches="tight")
    plt.savefig(dataset.upper() + "Convex_Syn_test_Com.pdf", bbox_inches="tight")
    plt.close()


def plot_summary_one_figure_mnist_Compare(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)
    dataset = dataset

    glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate )
    for i in range(Numb_Algs):
        print("max accurancy:", glob_acc_[i].max())
    glob_acc =  average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')

    linestyles = ['-', '--', '-.','-', '--', '-.']
    linestyles = ['-','-','-','-','-','-','-']
    #linestyles = ['-','-','-','-','-','-','-']
    markers = ["o","v","s","*","x","P"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 'tab:brown', 'm']
    plt.figure(1,figsize=(5, 5))
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    plt.grid(True)
    # training loss
    marks = []
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label, linewidth = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    #plt.ylim([0.05,  0.6]) # non convex-case
    plt.ylim([0.19,  0.4]) # convex-case
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
    plt.savefig(dataset.upper() + "Convex_Mnist_train_Com.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_train_Com.pdf", bbox_inches="tight")
    plt.figure(2,figsize=(5, 5))
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    plt.grid(True)
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label, linewidth = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    #plt.ylim([0.84,  0.98]) # non convex-case
    plt.ylim([0.88,  0.95]) # Convex-case
    plt.savefig(dataset.upper() + "Convex_Mnist_test_Com.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_test_Com.pdf", bbox_inches="tight")
    plt.close()

def plot_summary_one_figure_mnist_K(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)
    dataset = dataset

    glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate )

    glob_acc =  average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')

    linestyles = ['-','-','-','-.','-.','-.']
    #linestyles = ['-','-','-','-','-','-','-']
    markers = ["o","v","s","*","x","P"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 'tab:brown', 'm']
    plt.figure(1,figsize=(5, 5))
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    plt.grid(True)
    # training loss
    marks = []
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label + r': $K = $'+ str(k[i]), linewidth  = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    #plt.ylim([0.05,  0.6]) # non convex-case
    plt.ylim([0.19,  0.5]) # convex-case
    plt.savefig(dataset.upper() + "Convex_Mnist_train_K.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_train_K.pdf", bbox_inches="tight")
    plt.figure(2,figsize=(5, 5))
    plt.grid(True)
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label + r': $K = $'+ str(k[i]), linewidth  = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    # plt.ylim([0.84,  0.98]) # non convex-case
    plt.ylim([0.86,  0.95]) # Convex-case
    plt.savefig(dataset.upper() + "Convex_Mnist_test_K.pdf", bbox_inches="tight")
   #plt.savefig(dataset.upper() + "Non_Convex_Mnist_test_K.pdf", bbox_inches="tight")
    plt.close()

def plot_summary_one_figure_mnist_R(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)
    dataset = dataset

    glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate )

    glob_acc =  average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')
    linestyles = ['-','-','-','-.','-.','-.']
    #linestyles = ['-','-','-','-','-','-','-']
    markers = ["o","v","s","*","x","P"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 'tab:brown', 'm']
    plt.figure(1,figsize=(5, 5))
    plt.grid(True)
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    # training loss
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label + r': $R = $'+ str(loc_ep1[i]), linewidth  = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    #plt.ylim([0.05,  0.6]) # non convex-case
    plt.ylim([0.17,  0.5]) # convex-case
    plt.savefig(dataset.upper() + "Convex_Mnist_train_R.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_train_R.pdf", bbox_inches="tight")
    plt.figure(2,figsize=(5, 5))
    plt.grid(True)
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label + r': $R = $'+ str(loc_ep1[i]), linewidth  = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    # plt.ylim([0.84,  0.985]) # non convex-case
    plt.ylim([0.86,  0.955]) # Convex-case
    plt.savefig(dataset.upper() + "Convex_Mnist_test_R.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_test_R.pdf", bbox_inches="tight")
    plt.close()

def plot_summary_one_figure_mnist_L(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)
    dataset = dataset

    glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate )

    glob_acc =  average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')
    linestyles = ['-','-','-','-.','-.','-.']
    #linestyles = ['-','-','-','-','-','-','-']
    markers = ["o","v","s","*","x","d"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 'tab:brown', 'm']
    plt.figure(1,figsize=(5, 5))
    plt.grid(True)
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    # training loss
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label + r': $\lambda = $'+ str(lamb[i]), linewidth  = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    #plt.ylim([0.05,  0.6]) # non convex-case
    plt.ylim([0.19,  0.5]) # convex-case
    plt.savefig(dataset.upper() + "Convex_Mnist_train_L.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_train_L.pdf", bbox_inches="tight")
    plt.figure(2,figsize=(5, 5))
    plt.grid(True)
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label + r': $\lambda = $'+ str(lamb[i]), linewidth  = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    #plt.ylim([0.84,  0.98]) # non convex-case
    plt.ylim([0.86,  0.95]) # Convex-case
    plt.savefig(dataset.upper() + "Convex_Mnist_test_L.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_test_L.pdf", bbox_inches="tight")
    plt.close()

def plot_summary_one_figure_mnist_D(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)
    dataset = dataset

    glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate )

    glob_acc =  average_smooth(glob_acc_, window='flat')
    train_loss = average_smooth(train_loss_, window='flat')
    train_acc = average_smooth(train_acc_, window='flat')
    linestyles = ['-','-','-','-.','-.','-.']
    markers = ["o","v","s","*","x","P"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 'tab:brown', 'm']
    plt.figure(1,figsize=(5, 5))
    plt.grid(True)
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    # training loss
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label + r': $|\mathcal{D}|=$'+ str(batch_size[i]), linewidth  = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    #plt.ylim([0.05,  0.6]) # non convex-case
    plt.ylim([0.19,  0.5]) # convex-case
    plt.savefig(dataset.upper() + "Convex_Mnist_train_D.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_train_D.pdf", bbox_inches="tight")
    plt.figure(2,figsize=(5, 5))
    plt.grid(True)
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label + r': $|\mathcal{D}|=$'+ str(batch_size[i]), linewidth  = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    # plt.ylim([0.84,  0.98]) # non convex-case
    plt.ylim([0.86,  0.95]) # Convex-case
    plt.savefig(dataset.upper() + "Convex_Mnist_test_D.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_test_D.pdf", bbox_inches="tight")
    plt.close()


def plot_summary_one_figure_mnist_Beta(num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate):
    Numb_Algs = len(algorithms_list)
    dataset = dataset

    glob_acc_, train_acc_, train_loss_ = get_training_data_value( num_users, loc_ep1, Numb_Glob_Iters, lamb, learning_rate, beta, algorithms_list, batch_size, dataset, k, personal_learning_rate )

    glob_acc =  average_smooth(glob_acc_,window_len=10, window='flat')
    train_loss = average_smooth(train_loss_,window_len=10, window='flat')
    train_acc = average_smooth(train_acc_,window_len=10, window='flat')

    linestyles = ['-','-','-','-.','-.','-.']
    markers = ["o","v","s","*","x","P"]
    print(lamb)
    colors = ['tab:blue', 'tab:green', 'r', 'darkorange', 'tab:brown', 'm']
    plt.figure(1,figsize=(5, 5))
    plt.grid(True)
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    # training loss
    marks = []
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(train_loss[i, 1:], linestyle=linestyles[i], label=label + r': $\beta = $'+ str(beta[i]), linewidth  = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='upper right')
    plt.ylabel('Training Loss')
    plt.xlabel('Global rounds')
    #plt.ylim([0.05,  0.6]) # non convex-case
    plt.ylim([0.18,  0.5]) # convex-case
    plt.savefig(dataset.upper() + "Convex_Mnist_train_Beta.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_train_Beta.pdf", bbox_inches="tight")
    plt.figure(2,figsize=(5, 5))
    plt.grid(True)
    plt.title("$\mu-$"+ "strongly convex")
    # plt.title("Nonconvex") # for non convex case
    # Global accurancy
    for i in range(Numb_Algs):
        label = get_label_name(algorithms_list[i])
        plt.plot(glob_acc[i, 1:], linestyle=linestyles[i], label=label + r': $\beta = $'+ str(beta[i]), linewidth  = 1, color=colors[i],marker = markers[i],markevery=0.2, markersize=5)
    plt.legend(loc='lower right')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Global rounds')
    # plt.ylim([0.84,  0.985]) # non convex-case
    plt.ylim([0.88,  0.946]) # Convex-case
    plt.savefig(dataset.upper() + "Convex_Mnist_test_Beta.pdf", bbox_inches="tight")
    #plt.savefig(dataset.upper() + "Non_Convex_Mnist_test_Beta.pdf", bbox_inches="tight")
    plt.close()