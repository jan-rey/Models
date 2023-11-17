#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rescorla Wagner model with epsilon-greedy decision making policy
Meta-learning of epsilon with policy-gradient method
Contexts:
*Stable: reward probabiliies stay the same during all trials
*Volatile: reward probabilities shuffle every couple of trials
*Reinforced variability: reward dependent on how variable options are chosen according to Hide and Seek game
First reward schedule test:
    stable: [0.7,0.7,0.7,0.3,0.3,0.3,0.3,0.3]
    volatile: [0.9,0.9,0.9,0.1,0.1,0.1,0.1,0.1]
    variable: least frequent 60% of sequences
                         
Initial values of epsilon are fixed (equal to ML_mean_int) as opposed to being sampled
@author: Janne Reynders; janne.reynders@ugent.be
"""
import numpy as np                  
import pandas as pd                 
import matplotlib.pyplot as plt     
import os
import random

#simulation of Rescorla-Wagner model with meta-learning of epsilon 
#meta-learning goes through parameter ML which is transformed to epsilon with a logit transformation
#rewards are baselined
def simulate_RW_MLeps_has8(Q_alpha, reward_alpha, ML_alpha_mean, ML_alpha_std, T, Q_int, ML_mean_int, ML_std_int, scale, update, percentile):
    seq_options = np.array([[a, b] for a in range(8) for b in range(8)])

    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    Freq = np.random.uniform(0.9,1.1,K_seq)
    #alpha          --->        learning rate
    #ML_alpha       --->        learning rate for meta-learning parameter (list of two LR's: one for mean ML and one for std ML)
    #T              --->        amount of trials for each simulation
    #K              --->        amount of choice options
    #Q_int          --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #eps_mean_int   --->        initial value for the mean of eps
    #eps_std_int    --->        initial value for the standard deviation of eps
    #scale          --->        scale with which ML-parameter gets transformed to epsilon in a logit transformation
    #threshold      --->        % of least frequently chosen options which will be rewarded
    #update         --->        this number equals the amount of trials after which the meta-learning parameter gets updated

    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made based on epsilon
    r = np.zeros((T), dtype=float) #vector of rewards
    av_reward = 0.5 #starting baseline reward


    ML_stored = np.zeros((T), dtype=float)
    ML_mean_stored = np.zeros((T), dtype=float)
    ML_std_stored = np.zeros((T), dtype=float)
    eps_var = np.zeros((T), dtype=float)
    eps_var_mean = np.zeros((T), dtype=float)
    eps_var_std = np.zeros((T), dtype=float)
    av_reward_stored = np.zeros((T), dtype=float)


    
    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    #initial values of the meta-learning (ML) parameter 
    #ML_mean_int = (1/scale)*np.log(eps_mean_int/(1-eps_mean_int))
    #ML_std_int = (1/scale)*np.log(eps_std_int/(1-eps_std_int))

    #ML_int = np.random.normal(loc = ML_mean_int, scale=ML_std_int)
    ML_int = ML_mean_int
    ML_mean = ML_mean_int
    ML_std = ML_std_int
    ML = ML_int   

    log_ML_mean = np.log(ML_mean) #the parameter that gets updated at the end of the loop
    log_ML_std = np.log(ML_std) #the parameter that gets updated at the end of the loop

    for t in range(T):
        #epsilon is calculated with a logit transformation of the ML
        eps = np.exp(scale*ML)/(1+np.exp(scale*ML))
        eps_mean = np.exp(scale*ML_mean)/(1+np.exp(scale*ML_mean))
        eps_std = np.exp(scale*ML_std)/(1+np.exp(scale*ML_std))
        eps_var[t] = eps
        eps_var_mean[t] = eps_mean
        eps_var_std[t] = eps_std
        # store values for Q and epsilon
        Q_k_stored[t,:] = Q_k
        ML_stored[t] = ML
        ML_mean_stored[t] = ML_mean
        ML_std_stored[t] = ML_std

      
        # make choice based on choice probababilities
        rand[t] = np.random.choice(2, p=[1-eps,eps])
        if rand[t] == 0:
         k[t] = np.argmax(Q_k)
        if rand[t] == 1:
         k[t] = np.random.choice(range(K))

        if t < 1:
            r[t] = 1
        else: 
            current_seq = k[t-1:t+1]
            current_index = np.where(np.all(seq_options==current_seq,axis=1))[0]
            if Freq[current_index] < np.percentile(Freq,percentile):
                r[t] = 1
            else:
                r[t] = 0
            Adding = np.ones(64, dtype=float)*(-1/63)
            Freq = np.add(Freq, Adding)
            Freq[current_index] = Freq[current_index] + 1 + (1/63)
            Freq = Freq*0.984
        

        # update Q values
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + Q_alpha * delta_k

        # baseline reward
        av_reward = av_reward + reward_alpha * (r[t] - av_reward)
        av_reward_stored[t] = av_reward #this is the weighted reward over trials

        if (t%update)==0 and t != 0: #update every x=update amount of trials (first update after 11 trials, last update lasts 9 trials, because index t begins at 0)
            if t == update:
                baseline_reward = r[t] - av_reward_stored[t]
            else: #the reward is baselined by subtracting the weighted average reward over trials 0 -> t-x (x = update) from the average reward in the last x trials
                begin = t-update
                R_mean = np.mean(r[begin+1:t+1])
                baseline_reward = R_mean-av_reward_stored[begin+1]

            ###################################################################################
            ###################################################################################
            #The next lines ensure an update of the meta-learning parameter
            #both the mean and the std can be updated
            #the first lines establish an update of mean and std in normal space
            #the next lines establish an update of mean and std in log space
            #lines that are not used need to be commented out
            dif = ML - ML_mean 
            dif2 = ((dif)**2) 
            ### UPDATE MEAN AND STD IN NORMAL SPACE
            ### MEAN:
            update_ML_mean = (ML_alpha_mean*baseline_reward*dif) / ((ML_std)**2)            
            ML_mean = ML_mean + update_ML_mean
            ### STD:
            #update_ML_std = ML_alpha_std*baseline_reward*((dif2 - (ML_std)**2)/(ML_std)**3)
            #ML_std = ML_std + update_ML_std


            ### UPDATE LOG(MEAN) AND LOG(STD) 
            ### LOG(MEAN) 
            #update_log_mean = (ML_alpha_mean*baseline_reward*dif*ML) / ((ML_std)**2) 
            #log_ML_mean = log_ML_mean + update_log_mean
            #ML_mean = np.exp(log_ML_mean)
            ### LOG(STD)
            update_log_std =  ML_alpha_std*baseline_reward*((dif2 /(ML_std)**2)-1) 
            log_ML_std = log_ML_std + update_log_std 
            ML_std = np.exp(log_ML_std)
            ###################################################################################
            ###################################################################################
            ML_mean = np.min([ML_mean,10])
            ML_mean = np.max([ML_mean, -10])
            ML_std = np.min([ML_std, 5])
            ML = np.random.normal(loc=ML_mean, scale=ML_std)
            
  
    return k, r, Q_k_stored, eps_var, eps_var_mean, eps_var_std, ML_stored, ML_mean_stored, ML_std_stored

def simulate_RW_MLeps_stable(Q_alpha, reward_alpha, ML_alpha_mean, ML_alpha_std, T, Q_int, ML_mean_int, ML_std_int, scale, update, reward_stable):
    reward_prob = reward_stable
    K=len(reward_prob) #the amount of choice options
    #alpha          --->        learning rate
    #ML_alpha       --->        learning rate for meta-learning parameter (list of two LR's: one for mean ML and one for std ML)
    #T              --->        amount of trials for each simulation
    #K              --->        amount of choice options
    #Q_int          --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #eps_mean_int   --->        initial value for the mean of eps
    #eps_std_int    --->        initial value for the standard deviation of eps
    #reward prob    --->        probabilites to recieve a reward, associated to each option
    #scale          --->        scale with which ML-parameter gets transformed to epsilon in a logit transformation
    #update         --->        this number equals the amount of trials after which the meta-learning parameter gets updated

   
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made based on epsilon
    r = np.zeros((T), dtype=float) #vector of rewards
    av_reward = 0.5 #starting baseline reward

    ML_stored = np.zeros((T), dtype=float)
    ML_mean_stored = np.zeros((T), dtype=float)
    ML_std_stored = np.zeros((T), dtype=float)
    eps_sta = np.zeros((T), dtype=float)
    eps_sta_mean = np.zeros((T), dtype=float)
    eps_sta_std = np.zeros((T), dtype=float)
    av_reward_stored = np.zeros((T), dtype=float)
    

    
    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    #initial values of the meta-learning (ML) parameter
    #ML_mean_int = (1/scale)*np.log(eps_mean_int/(1-eps_mean_int))
    #ML_std_int = (1/scale)*np.log(eps_std_int/(1-eps_std_int))
    
    #ML_int = np.random.normal(loc = ML_mean_int, scale=ML_std_int)
    ML_int = ML_mean_int
    ML_mean = ML_mean_int
    ML_std = ML_std_int
    ML = ML_int   

    log_ML_mean = np.log(ML_mean) #the parameter that gets updated at the end of the loop
    log_ML_std = np.log(ML_std) #the parameter that gets updated at the end of the loop

    for t in range(T):
        #epsilon is calculated with a logit transformation of the ML
        eps = np.exp(scale*ML)/(1+np.exp(scale*ML))
        eps_mean = np.exp(scale*ML_mean)/(1+np.exp(scale*ML_mean))
        eps_std = np.exp(scale*ML_std)/(1+np.exp(scale*ML_std))
        eps_sta[t] = eps
        eps_sta_mean[t] = eps_mean
        eps_sta_std[t] = eps_std
        # store values for Q and epsilon
        Q_k_stored[t,:] = Q_k
        ML_stored[t] = ML
        ML_mean_stored[t] = ML_mean
        ML_std_stored[t] = ML_std

      
        # make choice based on choice probababilities
        rand[t] = np.random.choice(2, p=[1-eps,eps])
        if rand[t] == 0:
         k[t] = np.argmax(Q_k)
        if rand[t] == 1:
         k[t] = np.random.choice(range(K))
           
        a1 = reward_prob[k[t]]
        a0 = 1-a1
        r[t] = np.random.choice([1, 0], p=[a1, a0])

        # update Q values
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + Q_alpha * delta_k

        # baseline reward
        av_reward = av_reward + reward_alpha * (r[t] - av_reward)
        av_reward_stored[t] = av_reward #this is the weighted reward over trials

        if (t%update)==0 and t != 0: #update every x=update amount of trials (first update after 11 trials, last update lasts 9 trials, because index t begins at 0)
            if t == update:
                baseline_reward = r[t] - av_reward_stored[t]
            else: #the reward is baselined by subtracting the weighted average reward over trials 0 -> t-x (x = update) from the average reward in the last x trials
                begin = t-update
                R_mean = np.mean(r[begin+1:t+1])
                baseline_reward = R_mean-av_reward_stored[begin+1]
            ###################################################################################
            ###################################################################################
            #The next lines ensure an update of the meta-learning parameter
            #both the mean and the std can be updated
            #the first lines establish an update of mean and std in normal space
            #the next lines establish an update of mean and std in log space
            #lines that are not used need to be commented out
            dif = ML - ML_mean 
            dif2 = ((dif)**2) 
            ### UPDATE MEAN AND STD IN NORMAL SPACE
            ### MEAN:
            update_ML_mean = (ML_alpha_mean*baseline_reward*dif) / ((ML_std)**2)            
            ML_mean = ML_mean + update_ML_mean
            ### STD:
            #update_ML_std = ML_alpha_std*baseline_reward*((dif2 - (ML_std)**2)/(ML_std)**3)
            #ML_std = ML_std + update_ML_std


            ### UPDATE LOG(MEAN) AND LOG(STD) 
            ### LOG(MEAN) 
            #update_log_mean = (ML_alpha_mean*baseline_reward*dif*ML) / ((ML_std)**2) 
            #log_ML_mean = log_ML_mean + update_log_mean
            #ML_mean = np.exp(log_ML_mean)
            ### LOG(STD)
            update_log_std =  ML_alpha_std*baseline_reward*((dif2 /(ML_std)**2)-1) 
            log_ML_std = log_ML_std + update_log_std 
            ML_std = np.exp(log_ML_std)
            ###################################################################################
            ###################################################################################
            ML_mean = np.min([ML_mean,10])
            ML_mean = np.max([ML_mean, -10])
            ML_std = np.min([ML_std, 5])
            ML = np.random.normal(loc=ML_mean, scale=ML_std)
         
  
    return k, r, Q_k_stored, eps_sta, eps_sta_mean, eps_sta_std, ML_stored, ML_mean_stored, ML_std_stored

def simulate_RW_MLeps_volatile(Q_alpha, reward_alpha, ML_alpha_mean, ML_alpha_std, T, Q_int, ML_mean_int, ML_std_int, rot, scale, update, reward_volatile):
    reward_prob = reward_volatile
    K=len(reward_prob) #the amount of choice options
    #alpha          --->        learning rate
    #ML_alpha       --->        learning rate for meta-learning parameter (list of two LR's: one for mean ML and one for std ML)
    #T              --->        amount of trials for each simulation
    #K              --->        amount of choice options
    #Q_int          --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #eps_mean_int   --->        initial value for the mean of eps
    #eps_std_int    --->        initial value for the standard deviation of eps
    #reward prob    --->        probabilites to recieve a reward, associated to each option
    #rot            --->        amount of trials after which mean reward values rotate among choice options
    #scale          --->        scale with which ML-parameter gets transformed to epsilon in a logit transformation
    #update         --->        this number equals the amount of trials after which the meta-learning parameter gets updated

   
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made based on epsilon
    r = np.zeros((T), dtype=float) #vector of rewards
    av_reward = 0.5 #starting baseline reward
    ML_stored = np.zeros((T), dtype=float)
    ML_mean_stored = np.zeros((T), dtype=float)
    ML_std_stored = np.zeros((T), dtype=float)
    eps_vol = np.zeros((T), dtype=float)
    eps_vol_mean = np.zeros((T), dtype=float)
    eps_vol_std = np.zeros((T), dtype=float)
    av_reward_stored = np.zeros((T), dtype=float)

    
    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    #initial values of the meta-learning (ML) parameter
    #ML_mean_int = (1/scale)*np.log(eps_mean_int/(1-eps_mean_int))
    #ML_std_int = eps_std_int #ML_std_int = (1/scale)*np.log(eps_std_int/(1-eps_std_int))
    
    #ML_int = np.random.normal(loc = ML_mean_int, scale=ML_std_int)
    ML_int = ML_mean_int
    ML_mean = ML_mean_int
    ML_std = ML_std_int
    ML = ML_int   

    log_ML_mean = np.log(ML_mean) #the parameter that gets updated at the end of the loop
    log_ML_std = np.log(ML_std) #the parameter that gets updated at the end of the loop

    first_update = 15
    for t in range(T):
        #epsilon is calculated with a logit transformation of the ML
        eps = np.exp(scale*ML)/(1+np.exp(scale*ML))
        eps_mean = np.exp(scale*ML_mean)/(1+np.exp(scale*ML_mean))
        eps_std = np.exp(scale*ML_std)/(1+np.exp(scale*ML_std))
        eps_vol[t] = eps
        eps_vol_mean[t] = eps_mean
        eps_vol_std[t] = eps_std
        # store values for Q and epsilon
        Q_k_stored[t,:] = Q_k
        ML_stored[t] = ML
        ML_mean_stored[t] = ML_mean
        ML_std_stored[t] = ML_std
      
        # make choice based on choice probababilities
        rand[t] = np.random.choice(2, p=[1-eps,eps])
        if rand[t] == 0:
         k[t] = np.argmax(Q_k)
        if rand[t] == 1:
         k[t] = np.random.choice(range(K))       

        # generate reward based on normal distribution linked to choice made
        if t == first_update:
            reward_orig = reward_prob.copy()
            while reward_orig == reward_prob:
                np.random.shuffle(reward_prob)
            p=[0.2,0.2,0.4,0.5,0.4,0.4,0.3,0.3,0.3,0.2,0.2,0.1,0.1,0.1]
            sum = np.sum(p)
            p=p/sum
            random_number = np.random.choice([12,13,14,15,16,17,18,19,20,21,22,23,24,25], p=p) #[7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            first_update = first_update + random_number
           
        a1 = reward_prob[k[t]]
        a0 = 1-a1
        r[t] = np.random.choice([1, 0], p=[a1, a0])

        # update Q values
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + Q_alpha * delta_k

        # baseline reward
        av_reward = av_reward + reward_alpha * (r[t] - av_reward)
        av_reward_stored[t] = av_reward #this is the weighted reward over trials

        if (t%update)==0 and t != 0: #update every x=update amount of trials (first update after 11 trials, last update lasts 9 trials, because index t begins at 0)
            if t == update:
                baseline_reward = r[t] - av_reward_stored[t]
            else: #the reward is baselined by subtracting the weighted average reward over trials 0 -> t-x (x = update) from the average reward in the last x trials
                begin = t-update
                R_mean = np.mean(r[begin+1:t+1])
                baseline_reward = R_mean-av_reward_stored[begin+1]
            ###################################################################################
            ###################################################################################
            #The next lines ensure an update of the meta-learning parameter
            #both the mean and the std can be updated
            #the first lines establish an update of mean and std in normal space
            #the next lines establish an update of mean and std in log space
            #lines that are not used need to be commented out
            dif = ML - ML_mean 
            dif2 = ((dif)**2)
            ### UPDATE MEAN AND STD IN NORMAL SPACE
            ### MEAN:
            update_ML_mean = (ML_alpha_mean*baseline_reward*dif) / ((ML_std)**2)            
            ML_mean = ML_mean + update_ML_mean
            ### STD:
            #update_ML_std = ML_alpha_std*baseline_reward*((dif2 - (ML_std)**2)/(ML_std)**3)
            #ML_std = ML_std + update_ML_std


            ### UPDATE LOG(MEAN) AND LOG(STD) 
            ### LOG(MEAN) 
            #update_log_mean = (ML_alpha_mean*baseline_reward*dif*ML) / ((ML_std)**2) 
            #log_ML_mean = log_ML_mean + update_log_mean
            #ML_mean = np.exp(log_ML_mean)
            ### LOG(STD)
            update_log_std =  ML_alpha_std*baseline_reward*((dif2 /(ML_std)**2)-1) 
            log_ML_std = log_ML_std + update_log_std 
            ML_std = np.exp(log_ML_std)
            ###################################################################################
            ###################################################################################
            ML_mean = np.min([ML_mean,10])
            ML_mean = np.max([ML_mean, -10])
            ML_std = np.min([ML_std, 5])
            ML = np.random.normal(loc=ML_mean, scale=ML_std)
            
  
    return k, r, Q_k_stored, eps_vol, eps_vol_mean, eps_vol_std, ML_stored, ML_mean_stored, ML_std_stored


sim_nr = '500trials'
reward_stable = [0.70,0.70,0.70,0.30,0.30,0.30,0.30,0.30]
reward_st = '3-5 70-30'
reward_volatile = [0.90,0.90,0.90,0.10,0.10,0.10,0.10,0.10]
reward_vl = '3-5 90-10'
percentile = 60
Q_alpha = 0.5
eps_mean_int = 0.5
ML_mean_int = np.log(eps_mean_int/(1-eps_mean_int))
ML_std_int = 1

update = 10
threshold = 0.5
percentage=50
T=500
Q_int = 1
reward_alpha = 0.25
amount_of_sim = 300
rot=10
K=8
ML_alpha_mean = 0.9
ML_alpha_std = 0.1 #LR std, LR pos and LR neg

window = 50





#################################################################
#SIMULATIONS
#################################################################

mean_start = 100 #9000 #trial from which mean epsilon is taken
mean_end = 100 #1000
# META-LEARNING OF EPSILON IN A VARIABLE CONTEXT (VARIABILITY IS REINFORCED)
#for barplots:
total_eps_var = np.zeros(amount_of_sim)
total_eps_var_mean = np.zeros(amount_of_sim)
total_eps_var_std = np.zeros(amount_of_sim)
reward_var = np.zeros(amount_of_sim)
reward_var2 = np.zeros(amount_of_sim)

#for time plots:
r_var_cumsum = np.zeros(T)
r_var = np.zeros(T)
e_var = np.zeros(T)
e_mean_var = np.zeros(T)
e_std_var = np.zeros(T)

#for time plots of untransformed ML
ML_var = np.zeros(T)
ML_mean_var = np.zeros(T)
ML_std_var = np.zeros(T)

#simulation:
for sim in range(amount_of_sim):
    k, r, Q_k_stored, eps_var, eps_var_mean, eps_var_std, ML_stored, ML_mean_stored, ML_std_stored = simulate_RW_MLeps_has8(Q_alpha=Q_alpha, reward_alpha= reward_alpha, ML_alpha_mean = ML_alpha_mean, ML_alpha_std=ML_alpha_std, T=T, Q_int=Q_int, ML_mean_int=ML_mean_int, ML_std_int=ML_std_int, scale=1, update = 10, percentile=percentile)
    
    #for bar plot:
    total_eps_var[sim] = np.mean(eps_var[mean_start:])
    total_eps_var_mean[sim] = np.mean(eps_var_mean[mean_start:])
    total_eps_var_std[sim] = np.mean(eps_var_std[mean_start:])
    reward_var[sim] = np.mean(r[mean_start:])
    reward_var2[sim] = np.mean(r[:mean_end])


    #for time plot:
    r_cumsum = np.cumsum(r)
    r_cumsum_av = np.zeros(len(r))
    for nr, i in enumerate(r):
        divide = nr+1
        r_cumsum_av[nr] = r_cumsum[nr]/divide

    r_var_cumsum = r_var_cumsum + r_cumsum_av
    r_var = r_var + r
    e_var = e_var + eps_var
    e_mean_var = e_mean_var + eps_var_mean
    e_std_var = e_std_var + eps_var_std

    #for time plots of untransformed ML
    ML_var = ML_var + ML_stored
    ML_mean_var = ML_mean_var + ML_mean_stored
    ML_std_var = ML_std_var + ML_std_stored

#for bar plot:
av_eps_var = np.mean(total_eps_var)
av_eps_var_mean = np.mean(total_eps_var_mean)
av_eps_var_std = np.mean(total_eps_var_std)
av_reward_var = np.mean(reward_var)
av_reward_var2 = np.mean(reward_var2)


std_eps_var = np.std(total_eps_var)
std_eps_var_mean = np.std(total_eps_var_mean)
std_eps_var_std = np.std(total_eps_var_std)
std_reward_var = np.std(reward_var)
std_reward_var2 = np.std(reward_var2)

#for time plot:
r_var_cumsum_end = np.divide(r_var_cumsum, amount_of_sim)
r_var_end = np.divide(r_var, amount_of_sim)
e_var_end= np.divide(e_var, amount_of_sim)
e_mean_var_end = np.divide(e_mean_var, amount_of_sim)
e_std_var_end = np.divide(e_std_var, amount_of_sim)

#for time plots of untransformed ML
ML_var_end = np.divide(ML_var, amount_of_sim)
ML_mean_var_end = np.divide(ML_mean_var, amount_of_sim)
ML_std_var_end = np.divide(ML_std_var, amount_of_sim)





# META-LEARNING OF EPSILON IN A STABLE CONTEXT (REWARD PROBABILITIES ARE CONSTANT)
#for bar plot:
total_eps_sta = np.zeros(amount_of_sim)
total_eps_sta_mean = np.zeros(amount_of_sim)
total_eps_sta_std = np.zeros(amount_of_sim)
reward_sta = np.zeros(amount_of_sim)
reward_sta2 = np.zeros(amount_of_sim)


#for time plots:
r_sta_cumsum = np.zeros(T)
r_sta = np.zeros(T)
e_sta = np.zeros(T)
e_mean_sta = np.zeros(T)
e_std_sta = np.zeros(T)

#for time plots of untransformed ML
ML_sta = np.zeros(T)
ML_mean_sta = np.zeros(T)
ML_std_sta = np.zeros(T)

#simulation:
for sim in range(amount_of_sim):
    k, r, Q_k_stored, eps_sta, eps_sta_mean, eps_sta_std, ML_stored, ML_mean_stored, ML_std_stored = simulate_RW_MLeps_stable(Q_alpha=Q_alpha, reward_alpha = reward_alpha, ML_alpha_mean = ML_alpha_mean, ML_alpha_std=ML_alpha_std, T=T, Q_int=Q_int, ML_mean_int=ML_mean_int, ML_std_int=ML_std_int, scale=1, update = 10,reward_stable=reward_stable)

    #for bar plot:
    total_eps_sta[sim] = np.mean(eps_sta[mean_start:])
    total_eps_sta_mean[sim] = np.mean(eps_sta_mean[mean_start:])
    total_eps_sta_std[sim] = np.mean(eps_sta_std[mean_start:])
    reward_sta[sim] = np.mean(r[mean_start:])
    reward_sta2[sim] = np.mean(r[:mean_end])

    #for time plots:
    r_cumsum = np.cumsum(r)
    r_cumsum_av = np.zeros(len(r))
    for nr, i in enumerate(r):
        divide = nr+1
        r_cumsum_av[nr] = r_cumsum[nr]/divide

    r_sta_cumsum = r_sta_cumsum + r_cumsum_av
    r_sta = r_sta + r
    e_sta = e_sta + eps_sta
    e_mean_sta = e_mean_sta + eps_sta_mean
    e_std_sta = e_std_sta + eps_sta_std

    #for time plots of untransformed ML
    ML_sta = ML_sta + ML_stored
    ML_mean_sta = ML_mean_sta + ML_mean_stored
    ML_std_sta = ML_std_sta + ML_std_stored

#for bar plot:
av_eps_sta = np.mean(total_eps_sta)
av_eps_sta_mean = np.mean(total_eps_sta_mean)
av_eps_sta_std = np.mean(total_eps_sta_std)
av_reward_sta = np.mean(reward_sta)
av_reward_sta2 = np.mean(reward_sta2)


std_eps_sta = np.std(total_eps_sta)
std_eps_sta_mean = np.std(total_eps_sta_mean)
std_eps_sta_std = np.std(total_eps_sta_std)
std_reward_sta = np.std(reward_sta)
std_reward_sta2 = np.std(reward_sta2)


#for time plots:
r_sta_cumsum_end = np.divide(r_sta_cumsum, amount_of_sim)
r_sta_end = np.divide(r_sta, amount_of_sim)
e_sta_end= np.divide(e_sta, amount_of_sim)
e_mean_sta_end = np.divide(e_mean_sta, amount_of_sim)
e_std_sta_end = np.divide(e_std_sta, amount_of_sim)

#for time plots of untransformed ML
ML_sta_end = np.divide(ML_sta, amount_of_sim)
ML_mean_sta_end = np.divide(ML_mean_sta, amount_of_sim)
ML_std_sta_end = np.divide(ML_std_sta, amount_of_sim)






# META-LEARNING OF EPSILON IN A VOLATILE CONTEXT (REWARD PROBABILITIES SHUFFLE)
#for bar plot:
total_eps_vol = np.zeros(amount_of_sim)
total_eps_vol_mean = np.zeros(amount_of_sim)
total_eps_vol_std = np.zeros(amount_of_sim)
reward_vol = np.zeros(amount_of_sim)
reward_vol2 = np.zeros(amount_of_sim)


#for time plots:
r_vol_cumsum = np.zeros(T)
r_vol = np.zeros(T)
e_vol = np.zeros(T)
e_mean_vol = np.zeros(T)
e_std_vol = np.zeros(T)

#for time plots of untransformed ML
ML_vol = np.zeros(T)
ML_mean_vol = np.zeros(T)
ML_std_vol = np.zeros(T)

for sim in range(amount_of_sim):
    k, r, Q_k_stored, eps_vol, eps_vol_mean, eps_vol_std, ML_stored, ML_mean_stored, ML_std_stored = simulate_RW_MLeps_volatile(Q_alpha=Q_alpha, reward_alpha= reward_alpha, ML_alpha_mean = ML_alpha_mean, ML_alpha_std=ML_alpha_std, T=T, Q_int=Q_int, ML_mean_int=ML_mean_int, ML_std_int=ML_std_int, rot=rot, scale=1, update = 10, reward_volatile=reward_volatile)
    
    #for bar plot:
    total_eps_vol[sim] = np.mean(eps_vol[mean_start:])
    total_eps_vol_mean[sim] = np.mean(eps_vol_mean[mean_start:])
    total_eps_vol_std[sim] = np.mean(eps_vol_std[mean_start:])
    reward_vol[sim] = np.mean(r[mean_start:])
    reward_vol2[sim] = np.mean(r[:mean_end])

    #for time plots:
    r_cumsum = np.cumsum(r)
    r_cumsum_av = np.zeros(len(r))
    for nr, i in enumerate(r):
        divide = nr+1
        r_cumsum_av[nr] = r_cumsum[nr]/divide

    r_vol_cumsum = r_vol_cumsum + r_cumsum_av
    r_vol = r_vol + r
    e_vol = e_vol + eps_vol
    e_mean_vol = e_mean_vol + eps_vol_mean
    e_std_vol = e_std_vol + eps_vol_std

    #for time plots of untransformed ML
    ML_vol = ML_vol + ML_stored
    ML_mean_vol = ML_mean_vol + ML_mean_stored
    ML_std_vol = ML_std_vol + ML_std_stored


#for bar plot:
av_eps_vol = np.mean(total_eps_vol)
av_eps_vol_mean = np.mean(total_eps_vol_mean)
av_eps_vol_std = np.mean(total_eps_vol_std)
av_reward_vol = np.mean(reward_vol)
av_reward_vol2 = np.mean(reward_vol2)

std_eps_vol = np.std(total_eps_vol)
std_eps_vol_mean = np.std(total_eps_vol_mean)
std_eps_vol_std = np.std(total_eps_vol_std)
std_reward_vol = np.std(reward_vol)
std_reward_vol2 = np.std(reward_vol2)

#for time plots:
r_vol_cumsum_end = np.divide(r_vol_cumsum, amount_of_sim)
r_vol_end = np.divide(r_vol, amount_of_sim)
e_vol_end= np.divide(e_vol, amount_of_sim)
e_mean_vol_end = np.divide(e_mean_vol, amount_of_sim)
e_std_vol_end = np.divide(e_std_vol, amount_of_sim)

#for time plots of untransformed ML
ML_vol_end = np.divide(ML_vol, amount_of_sim)
ML_mean_vol_end = np.divide(ML_mean_vol, amount_of_sim)
ML_std_vol_end = np.divide(ML_std_vol, amount_of_sim)

#################################################################
#PLOTTING
#################################################################

save_dir_first = '/Users/jareynde/OneDrive - UGent/1A_Main_Project/Models/1Simulations/Env_HideAndSeek/output/eps-8choice'
new_sim_folder = f'sim{sim_nr}'
save_dir = os.path.join(save_dir_first, new_sim_folder)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#bar plots:
title1 = f'average sampled epsilon over {amount_of_sim} simulations, Q-value learning rate of {Q_alpha} and ML learning rate of {ML_alpha_mean}\naverage gained rewards in last 1000 trials are for stable {round(av_reward_sta, 2)} +/- {round(std_reward_sta,2)}, for volatile {round(av_reward_vol,2)} +/- {round(std_reward_vol,2)} and for variable {round(av_reward_var,2)} +/- {round(std_reward_var,2)}\naverage gained rewards in first 1000 trials are for stable {round(av_reward_sta2, 2)} +/- {round(std_reward_sta2,2)}, for volatile {round(av_reward_vol2,2)} +/- {round(std_reward_vol2,2)} and for variable {round(av_reward_var2,2)} +/- {round(std_reward_var2,2)}'
fig_name = os.path.join(save_dir, f'sim{sim_nr}_sampled_epsilon')
fig, ax= plt.subplots(figsize=(10, 7))
ax.bar(['stable context', 'volatile context', 'variable context'], [av_eps_sta, av_eps_vol, av_eps_var], yerr=[std_eps_sta, std_eps_vol, std_eps_var])
ax.set_ylabel('epsilon')
plt.title(title1)
plt.savefig(fig_name)
#plt.show()


store_av_eps = {
    'av_LR_var' : av_eps_var,
    'std_LR_var' : std_eps_var,
    'av_LR_sta' : av_eps_sta,
    'std_LR_sta' : std_eps_sta,
    'av_LR_vol' : av_eps_vol,
    'std_LR_vol' : std_eps_vol,


}

title_excel = os.path.join(save_dir, f'sim{sim_nr}av_eps{eps_mean_int}.xlsx')
df = pd.DataFrame(data=store_av_eps, index=[1])
df.to_excel(title_excel, index=False)

title2 = f'average updated mean epsilon over {amount_of_sim} simulations, Q-value learning rate of {Q_alpha} and ML learning rate of {ML_alpha_mean}\naverage gained rewards are for stable {round(av_reward_sta, 2)} +/- {round(std_reward_sta,2)}, for volatile {round(av_reward_vol,2)} +/- {round(std_reward_vol,2)} and for variable {round(av_reward_var,2)} +/- {round(std_reward_var,2)}'
fig_name = os.path.join(save_dir, f'sim{sim_nr}_mean_epsilon')
fig, ax= plt.subplots(figsize=(10, 7))
ax.bar(['stable context', 'volatile context', 'variable context'], [av_eps_sta_mean, av_eps_vol_mean, av_eps_var_mean], yerr=[std_eps_sta_mean, std_eps_vol_mean, std_eps_var_mean])
ax.set_ylabel('epsilon')
plt.title(title2) 
plt.savefig(fig_name)
#plt.show()

#time plots:

time = np.linspace(1, T, T, endpoint=True)

#time plot for epsilon:
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,15))

evar, = ax1.plot(time, e_var_end, label=f'sampled epsilon')
evar_mean, = ax1.plot(time, e_mean_var_end, label=f'mean epsilon')
evar_std, = ax1.plot(time, e_std_var_end, label=f'standard deviation of epsilon')

esta, = ax2.plot(time, e_sta_end, label=f'sampled epsilon')
esta_mean, = ax2.plot(time, e_mean_sta_end, label=f'mean epsilon')
esta_std, = ax2.plot(time, e_std_sta_end, label=f'standard deviation of epsilon')

evol, = ax3.plot(time, e_vol_end, label=f'sampled epsilon')
evol_mean, = ax3.plot(time, e_mean_vol_end, label=f'mean epsilon')
evol_std, = ax3.plot(time, e_std_vol_end, label=f'standard deviation of epsilon')

ax1.set_title('variable context')
ax1.set_xlabel('trials')
ax1.set_ylabel('epsilon')
ax1.legend(handles=[evar, evar_mean, evar_std])

ax2.set_title('stable context')
ax2.set_xlabel('trials')
ax2.set_ylabel('epsilon')
ax2.legend(handles=[esta, esta_mean, esta_std])

ax3.set_title('volatile context')
ax3.set_xlabel('trials')
ax3.set_ylabel('epsilon')
ax3.legend(handles=[evol, evol_mean, evol_std])

fig_name = os.path.join(save_dir, f'sim{sim_nr}_eps_ifo_time')
f.suptitle(f'epsilon averaged over {amount_of_sim} simulations with ML learning rate of {ML_alpha_mean}')
plt.savefig(fig_name)
#plt.show()

#time plot for untransformed ML:
f2, (ax4, ax5, ax6) = plt.subplots(3, 1, figsize=(12,15))

MLvar, = ax4.plot(time, ML_var_end, label=f'sampled ML')
MLvar_mean, = ax4.plot(time, ML_mean_var_end, label=f'mean ML')
MLvar_std, = ax4.plot(time, ML_std_var_end, label=f'standard deviation of ML')

MLsta, = ax5.plot(time, ML_sta_end, label=f'sampled ML')
MLsta_mean, = ax5.plot(time, ML_mean_sta_end, label=f'mean ML')
MLsta_std, = ax5.plot(time, ML_std_sta_end, label=f'standard deviation of ML')

MLvol, = ax6.plot(time, ML_vol_end, label=f'sampled ML')
MLvol_mean, = ax6.plot(time, ML_mean_vol_end, label=f'mean ML')
MLvol_std, = ax6.plot(time, ML_std_vol_end, label=f'standard deviation of ML')

ax4.set_title('variable context')
ax4.set_xlabel('trials')
ax4.set_ylabel('ML before transforming to eps')
ax4.legend(handles=[MLvar, MLvar_mean, MLvar_std])

ax5.set_title('stable context')
ax5.set_xlabel('trials')
ax5.set_ylabel('ML before transforming to eps')
ax5.legend(handles=[MLsta, MLsta_mean, MLsta_std])

ax6.set_title('volatile context')
ax6.set_xlabel('trials')
ax6.set_ylabel('ML before transforming to eps')
ax6.legend(handles=[MLvol, MLvol_mean, MLvol_std])

fig_name = os.path.join(save_dir, f'sim{sim_nr}_ML_ifo_time')
f2.suptitle(f'untransformed meta-learning (ML) parameter averaged over {amount_of_sim} simulations with ML learning rate of {ML_alpha_mean}')
plt.savefig(fig_name)
#plt.show()

#time plot for untransformed ML vs transformed eps:
f3, (ax7, ax8, ax9) = plt.subplots(3, 1, figsize=(12,15))

var1, = ax7.plot(time, ML_var_end, label=f'sampled ML')
var2, = ax7.plot(time, ML_mean_var_end, label=f'mean ML')
var3, = ax7.plot(time, e_var_end, label=f'sampled eps (transformed ML)')
var4, = ax7.plot(time, e_mean_var_end, label=f'mean eps (transformed ML)')

sta1, = ax8.plot(time, ML_sta_end, label=f'sampled ML')
sta2, = ax8.plot(time, ML_mean_sta_end, label=f'mean ML')
sta3, = ax8.plot(time, e_sta_end, label=f'sampled eps (transformed ML)')
sta4, = ax8.plot(time, e_mean_sta_end, label=f'mean eps (transformed ML)')

vol1, = ax9.plot(time, ML_vol_end, label=f'sampled ML')
vol2, = ax9.plot(time, ML_mean_vol_end, label=f'mean ML')
vol3, = ax9.plot(time, e_vol_end, label=f'sampled eps (transformed ML)')
vol4, = ax9.plot(time, e_mean_vol_end, label=f'mean eps (transformed ML)')

ax7.set_title('variable context')
ax7.set_xlabel('trials')
ax7.set_ylabel('ML/eps')
ax7.legend(handles=[var1, var2, var3, var4])

ax8.set_title('stable context')
ax8.set_xlabel('trials')
ax8.set_ylabel('ML/eps')
ax8.legend(handles=[sta1, sta2, sta3, sta4])

ax9.set_title('volatile context')
ax9.set_xlabel('trials')
ax9.set_ylabel('ML/eps')
ax9.legend(handles=[vol1, vol2, vol3, vol4])

fig_name = os.path.join(save_dir, f'sim{sim_nr}_MLvsEps_ifo_time')
f2.suptitle(f'untransformed meta-learning (ML) parameter and transformed epsilon averaged over {amount_of_sim} simulations with ML learning rate of {ML_alpha_mean}')
plt.savefig(fig_name)
#plt.show()



#time plot for reward:
f4, (ax10, ax11) = plt.subplots(2, 1, figsize=(12,15))

Rvar, = ax10.plot(time, r_var_end, label=f'reward in variable context')
Rsta, = ax10.plot(time, r_sta_end, label=f'reward in stable context')
Rvol, = ax10.plot(time, r_vol_end, label=f'reward in volatile context')

Rsum_var, = ax11.plot(time, r_var_cumsum_end, label=f'cumulative reward in variable context')
Rsum_sta, = ax11.plot(time, r_sta_cumsum_end, label=f'cumulative reward in stable context')
Rsum_vol, = ax11.plot(time, r_vol_cumsum_end, label=f'cumulative reward in volatile context')

ax10.set_title('reward in funcion of time')
ax10.set_xlabel('trials')
ax10.set_ylabel('Reward')
ax10.legend(handles=[Rvar, Rsta, Rvol])

ax11.set_title('cumulative reward in function of time')
ax11.set_xlabel('trials')
ax11.set_ylabel('cumulative reward')
ax11.legend(handles=[Rsum_var, Rsum_sta, Rsum_vol])


fig_name = os.path.join(save_dir, f'sim{sim_nr}_rewards_ifo_time')
f4.suptitle(f'rewards in each context averaged over {amount_of_sim} simulations with ML learning rate of {ML_alpha_mean}')
plt.savefig(fig_name)
#plt.show()

#SNE paper figure:
fig_name = os.path.join(save_dir, f'sim{sim_nr}_SNE_plot')
fig, ax12 = plt.subplots(figsize=(6, 3))
epsilonsta, = ax12.plot(time, e_sta_end, label=f'stable environment')
epsilonvol, = ax12.plot(time, e_vol_end, label=f'volatile environment')
epsilonvar, = ax12.plot(time, e_var_end, label=f'variable environment')
#ax12.legend(handles=[epsilonsta, epsilonvol, epsilonvar])
ax12.set_xlabel('trials', fontsize=18)
ax12.set_ylabel('epsilon', fontsize=18)
plt.ylim([0, 1])
plt.xlim([0, T])
plt.yticks(fontsize=15)
plt.xticks(fontsize = 15)
#ax12.legend(handles=[epsilonsta,epsilonvol,epsilonvar])
#ax12.set_title(f'meta-learning of epsilon, based on {amount_of_sim} simulations')
plt.savefig(fig_name)
plt.show()

global_mean_reward = (av_reward_sta + av_reward_var + av_reward_vol)/3
global_ste_reward = (np.sqrt((std_reward_var**2)+(std_reward_vol**2)+(std_reward_sta**2)))/np.sqrt(amount_of_sim*3)

store_av_reward = {
    'av_reward_var' : av_reward_var,
    'std_reward_var' : std_reward_var,
    'av_reward_sta' : av_reward_sta,
    'std_reward_sta' : std_reward_sta,
    'av_reward_vol' : av_reward_vol,
    'std_reward_vol' : std_reward_vol,
    'global_mean_reward' : global_mean_reward,
    'global_ste_reward' : global_ste_reward
}

title_excel = os.path.join(save_dir, f'sim{sim_nr}av_rewards_eps{eps_mean_int}.xlsx')
df = pd.DataFrame(data=store_av_reward, index=[1])
df.to_excel(title_excel, index=False)





store_param_values = {
    'simulation number' : sim_nr,
    'amount of trials' : T,
    'amount of simulations' : amount_of_sim,
    'amount of choice options' : K,
    'amount of trials after which meta-learning parameters are updated' : update,
    'initial Q-value' : Q_int,
    'learning rate for Q-value' : Q_alpha,
    'learning rate for the mean of the meta-learning parameter' : ML_alpha_mean,
    'learning rate for the std of the meta-learning parameter' : ML_alpha_std,
    'initial mean epsilon value' : eps_mean_int,
    'initial standard deviation of meta-learning parameter in logit transform' : ML_std_int,
    'amount of trials after which reward probabilities are shuffled in volatile context' : rot,
    'threshold of least chosen options (percentage that will be reinforced) in variability context' : percentage,
    'percentage of least freuquently occuring responses are rewarded': percentile,
    'reward probabilities in stable context' : reward_st,
    'reward probabiities in volatile context' : reward_vl
     }

title_excel = os.path.join(save_dir, f'sim{sim_nr}a_fixed_parameter_values.xlsx')
df = pd.DataFrame(data=store_param_values, index=[1])
df.to_excel(title_excel, index=False)