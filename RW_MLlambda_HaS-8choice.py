#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rescorla Wagner model with epsilon-greedy decision making policy
Meta-learning of lambda's with policy-gradient method
Contexts:
*Stable: reward probabiliies stay the same during all trials
*Volatile: reward probabilities shuffle every couple of trials
*Reinforced variability: reward dependent on how variable options are chosen according to Hide and Seek game
First reward schedule test:
    stable: [0.7,0.7,0.7,0.3,0.3,0.3,0.3,0.3]
    volatile: [0.9,0.9,0.9,0.1,0.1,0.1,0.1,0.1]
    variable: least frequent 60% of sequences, then 1
                         
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
def simulate_RW_MLlambda_has8(Q_alpha, eps, reward_alpha, MLunchosen_alpha_mean, MLunchosen_alpha_std, MLchosen_alpha_mean, MLchosen_alpha_std, T, Q_int, MLunchosen_mean_int, MLunchosen_std_int, MLchosen_mean_int, MLchosen_std_int, scale, update, percentile, operation):    
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
    av_reward_stored = np.zeros((T), dtype=float)

    #storing of ML unchosen:
    MLunchosen_stored = np.zeros((T), dtype=float)
    MLunchosen_mean_stored = np.zeros((T), dtype=float)
    MLunchosen_std_stored = np.zeros((T), dtype=float)
    unchosen_var = np.zeros((T), dtype=float)
    unchosen_var_mean = np.zeros((T), dtype=float)
    unchosen_var_std = np.zeros((T), dtype=float)

    #storing of ML chosen
    MLchosen_stored = np.zeros((T), dtype=float)
    MLchosen_mean_stored = np.zeros((T), dtype=float)
    MLchosen_std_stored = np.zeros((T), dtype=float)
    chosen_var = np.zeros((T), dtype=float)
    chosen_var_mean = np.zeros((T), dtype=float)
    chosen_var_std = np.zeros((T), dtype=float)

    
    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    #initial values of the meta-learning (ML) parameter 
    #ML_mean_int = (1/scale)*np.log(eps_mean_int/(1-eps_mean_int))
    #ML_std_int = (1/scale)*np.log(eps_std_int/(1-eps_std_int))

    #ML_int = np.random.normal(loc = ML_mean_int, scale=ML_std_int)
    #intitiliazing meta-learning of unchosen
    MLunchosen_int = MLunchosen_mean_int
    MLunchosen_mean = MLunchosen_mean_int
    MLunchosen_std = MLunchosen_std_int
    MLunchosen = MLunchosen_int 

    #intitiliazing meta-learning of chosen
    MLchosen_int = MLchosen_mean_int
    MLchosen_mean = MLchosen_mean_int
    MLchosen_std = MLchosen_std_int
    MLchosen = MLchosen_int  

    log_MLunchosen_mean = np.log(MLunchosen_mean) #the parameter that gets updated at the end of the loop
    log_MLunchosen_std = np.log(MLunchosen_std) #the parameter that gets updated at the end of the loop

    log_MLchosen_mean = np.log(MLchosen_mean) #the parameter that gets updated at the end of the loop
    log_MLchosen_std = np.log(MLchosen_std) #the parameter that gets updated at the end of the loop

    for t in range(T):
        #unchosen is calculated with a logit transformation of the ML
        #unchosen = np.exp(scale*MLunchosen)/(1+np.exp(scale*MLunchosen))
        #unchosen_mean = np.exp(scale*MLunchosen_mean)/(1+np.exp(scale*MLunchosen_mean))
        #unchosen_std = np.exp(scale*MLunchosen_std)/(1+np.exp(scale*MLunchosen_std))

        unchosen = MLunchosen
        unchosen_mean = MLunchosen_mean
        unchosen_std = MLunchosen_std

        #chosen is calculated with a logit transformation of the ML
        #chosen = np.exp(scale*MLchosen)/(1+np.exp(scale*MLchosen))
        #chosen_mean = np.exp(scale*MLchosen_mean)/(1+np.exp(scale*MLchosen_mean))
        #chosen_std = np.exp(scale*MLchosen_std)/(1+np.exp(scale*MLchosen_std))

        chosen = MLchosen
        chosen_mean = MLchosen_mean
        chosen_std = MLchosen_std

        unchosen_var[t] = unchosen
        unchosen_var_mean[t] = unchosen_mean
        unchosen_var_std[t] = unchosen_std

        chosen_var[t] = chosen
        chosen_var_mean[t] = chosen_mean
        chosen_var_std[t] = chosen_std
        # store values for Q and unchosen/chosen
        Q_k_stored[t,:] = Q_k

        MLunchosen_stored[t] = MLunchosen
        MLunchosen_mean_stored[t] = MLunchosen_mean
        MLunchosen_std_stored[t] = MLunchosen_std

        MLchosen_stored[t] = MLchosen
        MLchosen_mean_stored[t] = MLchosen_mean
        MLchosen_std_stored[t] = MLchosen_std

      
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
        

        # update Q values for chosen option:
        for option in range(K):
           if option == k[t]: #chosen option
                delta_k = r[t] - Q_k[k[t]]
                Q_k[k[t]] = Q_k[k[t]] + Q_alpha * delta_k
                if operation == '*':               
                    Q_k[k[t]] = Q_k[k[t]]*chosen
                else:
                    Q_k[k[t]] = Q_k[k[t]]+chosen
           else: #unchosen option
                #delta_k = r[t] - Q_k[option]
                #Q_k[option] = Q_k[option] + Q_alpha * delta_k
                if operation == '*':
                    Q_k[option] = Q_k[option]*unchosen
                else:
                    Q_k[option] = Q_k[option]+unchosen

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

            #UNCHOSEN PARAMETER
            dif_unchosen = MLunchosen - MLunchosen_mean 
            dif2_unchosen = ((dif_unchosen)**2) 
            ### UPDATE MEAN AND STD IN NORMAL SPACE
            ### MEAN:
            update_MLunchosen_mean = (MLunchosen_alpha_mean*baseline_reward*dif_unchosen) / ((MLunchosen_std)**2)            
            MLunchosen_mean = MLunchosen_mean + update_MLunchosen_mean
            ### STD:
            #update_MLunchosen_std = MLunchosen_alpha_std*baseline_reward*((dif2_unchosen - (MLunchosen_std)**2)/(MLunchosen_std)**3)
            #MLunchosen_std = MLunchosen_std + update_MLunchosen_std

            ### UPDATE LOG(MEAN) AND LOG(STD) 
            ### LOG(MEAN) 
            #update_unchosen_log_mean = (MLunchosen_alpha_mean*baseline_reward*dif_unchosen*MLunchosen) / ((MLunchosen_std)**2) 
            #log_MLunchosen_mean = log_MLunchosen_mean + update_unchosen_log_mean
            #MLunchosen_mean = np.exp(log_MLunchosen_mean)
            ### LOG(STD)
            update_unchosen_log_std =  MLunchosen_alpha_std*baseline_reward*((dif2_unchosen /(MLunchosen_std)**2)-1) 
            log_MLunchosen_std = log_MLunchosen_std + update_unchosen_log_std 
            MLunchosen_std = np.exp(log_MLunchosen_std)
            ###################################################################################
            ###################################################################################
            MLunchosen_mean = np.min([MLunchosen_mean,10])
            MLunchosen_mean = np.max([MLunchosen_mean, -10])
            MLunchosen_std = np.min([MLunchosen_std, 5])
            MLunchosen = np.random.normal(loc=MLunchosen_mean, scale=MLunchosen_std)

            #CHOSEN PARAMETER
            dif_chosen = MLchosen - MLchosen_mean 
            dif2_chosen = ((dif_chosen)**2) 
            ### UPDATE MEAN AND STD IN NORMAL SPACE
            ### MEAN:
            update_MLchosen_mean = (MLchosen_alpha_mean*baseline_reward*dif_chosen) / ((MLchosen_std)**2)            
            MLchosen_mean = MLchosen_mean + update_MLchosen_mean
            ### STD:
            #update_MLchosen_std = MLchosen_alpha_std*baseline_reward*((dif2_chosen - (MLchosen_std)**2)/(MLchosen_std)**3)
            #MLchosen_std = MLchosen_std + update_MLchosen_std

            ### UPDATE LOG(MEAN) AND LOG(STD) 
            ### LOG(MEAN) 
            #update_chosen_log_mean = (MLchosen_alpha_mean*baseline_reward*dif_chosen*MLchosen) / ((MLchosen_std)**2) 
            #log_MLchosen_mean = log_MLchosen_mean + update_chosen_log_mean
            #MLchosen_mean = np.exp(log_MLchosen_mean)
            ### LOG(STD)
            update_chosen_log_std =  MLchosen_alpha_std*baseline_reward*((dif2_chosen /(MLchosen_std)**2)-1) 
            log_MLchosen_std = log_MLchosen_std + update_chosen_log_std 
            MLchosen_std = np.exp(log_MLchosen_std)
            ###################################################################################
            ###################################################################################
            MLchosen_mean = np.min([MLchosen_mean,10])
            MLchosen_mean = np.max([MLchosen_mean, -10])
            MLchosen_std = np.min([MLchosen_std, 5])
            MLchosen = np.random.normal(loc=MLchosen_mean, scale=MLchosen_std)
            
  
    return k, r, Q_k_stored, unchosen_var, unchosen_var_mean, unchosen_var_std, chosen_var, chosen_var_mean, chosen_var_std, MLunchosen_stored, MLunchosen_mean_stored, MLunchosen_std_stored, MLchosen_stored, MLchosen_mean_stored, MLchosen_std_stored


def simulate_RW_MLlambda_stable(Q_alpha, eps, reward_alpha, MLunchosen_alpha_mean, MLunchosen_alpha_std, MLchosen_alpha_mean, MLchosen_alpha_std, T, Q_int, MLunchosen_mean_int, MLunchosen_std_int, MLchosen_mean_int, MLchosen_std_int, scale, update, operation, reward_stable):    
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
    av_reward_stored = np.zeros((T), dtype=float)

#storing of ML unchosen:
    MLunchosen_stored = np.zeros((T), dtype=float)
    MLunchosen_mean_stored = np.zeros((T), dtype=float)
    MLunchosen_std_stored = np.zeros((T), dtype=float)
    unchosen_sta = np.zeros((T), dtype=float)
    unchosen_sta_mean = np.zeros((T), dtype=float)
    unchosen_sta_std = np.zeros((T), dtype=float)

    #storing of ML chosen
    MLchosen_stored = np.zeros((T), dtype=float)
    MLchosen_mean_stored = np.zeros((T), dtype=float)
    MLchosen_std_stored = np.zeros((T), dtype=float)
    chosen_sta = np.zeros((T), dtype=float)
    chosen_sta_mean = np.zeros((T), dtype=float)
    chosen_sta_std = np.zeros((T), dtype=float)
    
    
    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    #initial values of the meta-learning (ML) parameter
    #ML_mean_int = (1/scale)*np.log(eps_mean_int/(1-eps_mean_int))
    #ML_std_int = (1/scale)*np.log(eps_std_int/(1-eps_std_int))
    
    #ML_int = np.random.normal(loc = ML_mean_int, scale=ML_std_int)
    #intitiliazing meta-learning of unchosen
    MLunchosen_int = MLunchosen_mean_int
    MLunchosen_mean = MLunchosen_mean_int
    MLunchosen_std = MLunchosen_std_int
    MLunchosen = MLunchosen_int 

    #intitiliazing meta-learning of chosen
    MLchosen_int = MLchosen_mean_int
    MLchosen_mean = MLchosen_mean_int
    MLchosen_std = MLchosen_std_int
    MLchosen = MLchosen_int  

    log_MLunchosen_mean = np.log(MLunchosen_mean) #the parameter that gets updated at the end of the loop
    log_MLunchosen_std = np.log(MLunchosen_std) #the parameter that gets updated at the end of the loop

    log_MLchosen_mean = np.log(MLchosen_mean) #the parameter that gets updated at the end of the loop
    log_MLchosen_std = np.log(MLchosen_std) #the parameter that gets updated at the end of the loop

    for t in range(T):
        #unchosen is calculated with a logit transformation of the ML
        #unchosen = np.exp(scale*MLunchosen)/(1+np.exp(scale*MLunchosen))
        #unchosen_mean = np.exp(scale*MLunchosen_mean)/(1+np.exp(scale*MLunchosen_mean))
        #unchosen_std = np.exp(scale*MLunchosen_std)/(1+np.exp(scale*MLunchosen_std))

        unchosen = MLunchosen
        unchosen_mean = MLunchosen_mean
        unchosen_std = MLunchosen_std

        #chosen is calculated with a logit transformation of the ML
        #chosen = np.exp(scale*MLchosen)/(1+np.exp(scale*MLchosen))
        #chosen_mean = np.exp(scale*MLchosen_mean)/(1+np.exp(scale*MLchosen_mean))
        #chosen_std = np.exp(scale*MLchosen_std)/(1+np.exp(scale*MLchosen_std))

        chosen = MLchosen
        chosen_mean = MLchosen_mean
        chosen_std = MLchosen_std

        unchosen_sta[t] = unchosen
        unchosen_sta_mean[t] = unchosen_mean
        unchosen_sta_std[t] = unchosen_std

        chosen_sta[t] = chosen
        chosen_sta_mean[t] = chosen_mean
        chosen_sta_std[t] = chosen_std
        # store values for Q and unchosen/chosen
        Q_k_stored[t,:] = Q_k

        MLunchosen_stored[t] = MLunchosen
        MLunchosen_mean_stored[t] = MLunchosen_mean
        MLunchosen_std_stored[t] = MLunchosen_std

        MLchosen_stored[t] = MLchosen
        MLchosen_mean_stored[t] = MLchosen_mean
        MLchosen_std_stored[t] = MLchosen_std

      
        # make choice based on choice probababilities
        rand[t] = np.random.choice(2, p=[1-eps,eps])
        if rand[t] == 0:
         k[t] = np.argmax(Q_k)
        if rand[t] == 1:
         k[t] = np.random.choice(range(K))
           
        a1 = reward_prob[k[t]]
        a0 = 1-a1
        r[t] = np.random.choice([1, 0], p=[a1, a0])

        # update Q values for chosen option:
        for option in range(K):
           if option == k[t]: #chosen option
                delta_k = r[t] - Q_k[k[t]]
                Q_k[k[t]] = Q_k[k[t]] + Q_alpha * delta_k
                if operation == '*':
                    Q_k[k[t]] = Q_k[k[t]]*chosen
                else: 
                    Q_k[k[t]] = Q_k[k[t]]+chosen
           else: #unchosen option
                #delta_k = r[t] - Q_k[option]
                #Q_k[option] = Q_k[option] + Q_alpha * delta_k
                if operation == '*':
                    Q_k[option] = Q_k[option]*unchosen
                else:
                    Q_k[option] = Q_k[option]+unchosen

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

            #UNCHOSEN PARAMETER
            dif_unchosen = MLunchosen - MLunchosen_mean 
            dif2_unchosen = ((dif_unchosen)**2) 
            ### UPDATE MEAN AND STD IN NORMAL SPACE
            ### MEAN:
            update_MLunchosen_mean = (MLunchosen_alpha_mean*baseline_reward*dif_unchosen) / ((MLunchosen_std)**2)            
            MLunchosen_mean = MLunchosen_mean + update_MLunchosen_mean
            ### STD:
            #update_MLunchosen_std = MLunchosen_alpha_std*baseline_reward*((dif2_unchosen - (MLunchosen_std)**2)/(MLunchosen_std)**3)
            #MLunchosen_std = MLunchosen_std + update_MLunchosen_std

            ### UPDATE LOG(MEAN) AND LOG(STD) 
            ### LOG(MEAN) 
            #update_unchosen_log_mean = (MLunchosen_alpha_mean*baseline_reward*dif_unchosen*MLunchosen) / ((MLunchosen_std)**2) 
            #log_MLunchosen_mean = log_MLunchosen_mean + update_unchosen_log_mean
            #MLunchosen_mean = np.exp(log_MLunchosen_mean)
            ### LOG(STD)
            update_unchosen_log_std =  MLunchosen_alpha_std*baseline_reward*((dif2_unchosen /(MLunchosen_std)**2)-1) 
            log_MLunchosen_std = log_MLunchosen_std + update_unchosen_log_std 
            MLunchosen_std = np.exp(log_MLunchosen_std)
            ###################################################################################
            ###################################################################################
            MLunchosen_mean = np.min([MLunchosen_mean,10])
            MLunchosen_mean = np.max([MLunchosen_mean, -10])
            MLunchosen_std = np.min([MLunchosen_std, 5])
            MLunchosen = np.random.normal(loc=MLunchosen_mean, scale=MLunchosen_std)

            #CHOSEN PARAMETER
            dif_chosen = MLchosen - MLchosen_mean 
            dif2_chosen = ((dif_chosen)**2) 
            ### UPDATE MEAN AND STD IN NORMAL SPACE
            ### MEAN:
            update_MLchosen_mean = (MLchosen_alpha_mean*baseline_reward*dif_chosen) / ((MLchosen_std)**2)            
            MLchosen_mean = MLchosen_mean + update_MLchosen_mean
            ### STD:
            #update_MLchosen_std = MLchosen_alpha_std*baseline_reward*((dif2_chosen - (MLchosen_std)**2)/(MLchosen_std)**3)
            #MLchosen_std = MLchosen_std + update_MLchosen_std

            ### UPDATE LOG(MEAN) AND LOG(STD) 
            ### LOG(MEAN) 
            #update_chosen_log_mean = (MLchosen_alpha_mean*baseline_reward*dif_chosen*MLchosen) / ((MLchosen_std)**2) 
            #log_MLchosen_mean = log_MLchosen_mean + update_chosen_log_mean
            #MLchosen_mean = np.exp(log_MLchosen_mean)
            ### LOG(STD)
            update_chosen_log_std =  MLchosen_alpha_std*baseline_reward*((dif2_chosen /(MLchosen_std)**2)-1) 
            log_MLchosen_std = log_MLchosen_std + update_chosen_log_std 
            MLchosen_std = np.exp(log_MLchosen_std)
            ###################################################################################
            ###################################################################################
            MLchosen_mean = np.min([MLchosen_mean,10])
            MLchosen_mean = np.max([MLchosen_mean, -10])
            MLchosen_std = np.min([MLchosen_std, 5])
            MLchosen = np.random.normal(loc=MLchosen_mean, scale=MLchosen_std)
         
  
    return k, r, Q_k_stored, unchosen_sta, unchosen_sta_mean, unchosen_sta_std, chosen_sta, chosen_sta_mean, chosen_sta_std, MLunchosen_stored, MLunchosen_mean_stored, MLunchosen_std_stored, MLchosen_stored, MLchosen_mean_stored, MLchosen_std_stored


def simulate_RW_MLlambda_volatile(Q_alpha, eps, reward_alpha, MLunchosen_alpha_mean, MLunchosen_alpha_std, MLchosen_alpha_mean, MLchosen_alpha_std, T, Q_int, MLunchosen_mean_int, MLunchosen_std_int, MLchosen_mean_int, MLchosen_std_int, rot, scale, update, operation, reward_volatile):
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
    av_reward_stored = np.zeros((T), dtype=float)

    #storing of ML unchosen:
    MLunchosen_stored = np.zeros((T), dtype=float)
    MLunchosen_mean_stored = np.zeros((T), dtype=float)
    MLunchosen_std_stored = np.zeros((T), dtype=float)
    unchosen_vol = np.zeros((T), dtype=float)
    unchosen_vol_mean = np.zeros((T), dtype=float)
    unchosen_vol_std = np.zeros((T), dtype=float)

    #storing of ML chosen
    MLchosen_stored = np.zeros((T), dtype=float)
    MLchosen_mean_stored = np.zeros((T), dtype=float)
    MLchosen_std_stored = np.zeros((T), dtype=float)
    chosen_vol = np.zeros((T), dtype=float)
    chosen_vol_mean = np.zeros((T), dtype=float)
    chosen_vol_std = np.zeros((T), dtype=float)

    
    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    #initial values of the meta-learning (ML) parameter
    #ML_mean_int = (1/scale)*np.log(eps_mean_int/(1-eps_mean_int))
    #ML_std_int = eps_std_int #ML_std_int = (1/scale)*np.log(eps_std_int/(1-eps_std_int))
    
    #ML_int = np.random.normal(loc = ML_mean_int, scale=ML_std_int)
    #intitiliazing meta-learning of unchosen
    MLunchosen_int = MLunchosen_mean_int
    MLunchosen_mean = MLunchosen_mean_int
    MLunchosen_std = MLunchosen_std_int
    MLunchosen = MLunchosen_int 

    #intitiliazing meta-learning of chosen
    MLchosen_int = MLchosen_mean_int
    MLchosen_mean = MLchosen_mean_int
    MLchosen_std = MLchosen_std_int
    MLchosen = MLchosen_int  

    log_MLunchosen_mean = np.log(MLunchosen_mean) #the parameter that gets updated at the end of the loop
    log_MLunchosen_std = np.log(MLunchosen_std) #the parameter that gets updated at the end of the loop

    log_MLchosen_mean = np.log(MLchosen_mean) #the parameter that gets updated at the end of the loop
    log_MLchosen_std = np.log(MLchosen_std) #the parameter that gets updated at the end of the loop

    first_update = 15
    for t in range(T):
        #unchosen is calculated with a logit transformation of the ML
        #unchosen = np.exp(scale*MLunchosen)/(1+np.exp(scale*MLunchosen))
        #unchosen_mean = np.exp(scale*MLunchosen_mean)/(1+np.exp(scale*MLunchosen_mean))
        #unchosen_std = np.exp(scale*MLunchosen_std)/(1+np.exp(scale*MLunchosen_std))

        unchosen = MLunchosen
        unchosen_mean = MLunchosen_mean
        unchosen_std = MLunchosen_std

        #chosen is calculated with a logit transformation of the ML
        #chosen = np.exp(scale*MLchosen)/(1+np.exp(scale*MLchosen))
        #chosen_mean = np.exp(scale*MLchosen_mean)/(1+np.exp(scale*MLchosen_mean))
        #chosen_std = np.exp(scale*MLchosen_std)/(1+np.exp(scale*MLchosen_std))

        chosen = MLchosen
        chosen_mean = MLchosen_mean
        chosen_std = MLchosen_std

        unchosen_vol[t] = unchosen
        unchosen_vol_mean[t] = unchosen_mean
        unchosen_vol_std[t] = unchosen_std

        chosen_vol[t] = chosen
        chosen_vol_mean[t] = chosen_mean
        chosen_vol_std[t] = chosen_std
        # store values for Q and unchosen/chosen
        Q_k_stored[t,:] = Q_k

        MLunchosen_stored[t] = MLunchosen
        MLunchosen_mean_stored[t] = MLunchosen_mean
        MLunchosen_std_stored[t] = MLunchosen_std

        MLchosen_stored[t] = MLchosen
        MLchosen_mean_stored[t] = MLchosen_mean
        MLchosen_std_stored[t] = MLchosen_std
      
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

        # update Q values for chosen option:
        for option in range(K):
           if option == k[t]: #chosen option
                delta_k = r[t] - Q_k[k[t]]
                Q_k[k[t]] = Q_k[k[t]] + Q_alpha * delta_k
                if operation == True:
                    Q_k[k[t]] = Q_k[k[t]]*chosen
                else:
                    Q_k[k[t]] = Q_k[k[t]]+chosen
           else: #unchosen option
                #delta_k = r[t] - Q_k[option]
                #Q_k[option] = Q_k[option] + Q_alpha * delta_k
                if operation == True:
                    Q_k[option] = Q_k[option]*unchosen
                else:
                    Q_k[option] = Q_k[option]+unchosen
        
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
            #the first lines evolblish an update of mean and std in normal space
            #the next lines evolblish an update of mean and std in log space
            #lines that are not used need to be commented out

            #UNCHOSEN PARAMETER
            dif_unchosen = MLunchosen - MLunchosen_mean 
            dif2_unchosen = ((dif_unchosen)**2) 
            ### UPDATE MEAN AND STD IN NORMAL SPACE
            ### MEAN:
            update_MLunchosen_mean = (MLunchosen_alpha_mean*baseline_reward*dif_unchosen) / ((MLunchosen_std)**2)            
            MLunchosen_mean = MLunchosen_mean + update_MLunchosen_mean
            ### STD:
            #update_MLunchosen_std = MLunchosen_alpha_std*baseline_reward*((dif2_unchosen - (MLunchosen_std)**2)/(MLunchosen_std)**3)
            #MLunchosen_std = MLunchosen_std + update_MLunchosen_std

            ### UPDATE LOG(MEAN) AND LOG(STD) 
            ### LOG(MEAN) 
            #update_unchosen_log_mean = (MLunchosen_alpha_mean*baseline_reward*dif_unchosen*MLunchosen) / ((MLunchosen_std)**2) 
            #log_MLunchosen_mean = log_MLunchosen_mean + update_unchosen_log_mean
            #MLunchosen_mean = np.exp(log_MLunchosen_mean)
            ### LOG(STD)
            update_unchosen_log_std =  MLunchosen_alpha_std*baseline_reward*((dif2_unchosen /(MLunchosen_std)**2)-1) 
            log_MLunchosen_std = log_MLunchosen_std + update_unchosen_log_std 
            MLunchosen_std = np.exp(log_MLunchosen_std)
            ###################################################################################
            ###################################################################################
            MLunchosen_mean = np.min([MLunchosen_mean,10])
            MLunchosen_mean = np.max([MLunchosen_mean, -10])
            MLunchosen_std = np.min([MLunchosen_std, 5])
            MLunchosen = np.random.normal(loc=MLunchosen_mean, scale=MLunchosen_std)

            #CHOSEN PARAMETER
            dif_chosen = MLchosen - MLchosen_mean 
            dif2_chosen = ((dif_chosen)**2) 
            ### UPDATE MEAN AND STD IN NORMAL SPACE
            ### MEAN:
            update_MLchosen_mean = (MLchosen_alpha_mean*baseline_reward*dif_chosen) / ((MLchosen_std)**2)            
            MLchosen_mean = MLchosen_mean + update_MLchosen_mean
            ### STD:
            #update_MLchosen_std = MLchosen_alpha_std*baseline_reward*((dif2_chosen - (MLchosen_std)**2)/(MLchosen_std)**3)
            #MLchosen_std = MLchosen_std + update_MLchosen_std

            ### UPDATE LOG(MEAN) AND LOG(STD) 
            ### LOG(MEAN) 
            #update_chosen_log_mean = (MLchosen_alpha_mean*baseline_reward*dif_chosen*MLchosen) / ((MLchosen_std)**2) 
            #log_MLchosen_mean = log_MLchosen_mean + update_chosen_log_mean
            #MLchosen_mean = np.exp(log_MLchosen_mean)
            ### LOG(STD)
            update_chosen_log_std =  MLchosen_alpha_std*baseline_reward*((dif2_chosen /(MLchosen_std)**2)-1) 
            log_MLchosen_std = log_MLchosen_std + update_chosen_log_std 
            MLchosen_std = np.exp(log_MLchosen_std)
            ###################################################################################
            ###################################################################################
            MLchosen_mean = np.min([MLchosen_mean,10])
            MLchosen_mean = np.max([MLchosen_mean, -10])
            MLchosen_std = np.min([MLchosen_std, 5])
            MLchosen = np.random.normal(loc=MLchosen_mean, scale=MLchosen_std)
            
  
    return k, r, Q_k_stored, unchosen_vol, unchosen_vol_mean, unchosen_vol_std, chosen_vol, chosen_vol_mean, chosen_vol_std, MLunchosen_stored, MLunchosen_mean_stored, MLunchosen_std_stored, MLchosen_stored, MLchosen_mean_stored, MLchosen_std_stored




sim_nr = 'SNE_good'
reward_stable = [0.70,0.70,0.70,0.30,0.30,0.30,0.30,0.30]
reward_st = '3-5 70-30'
reward_volatile = [0.90,0.90,0.90,0.10,0.10,0.10,0.10,0.10]
reward_vl = '3-5 90-10'
percentile = 60
Q_alpha = 0.5
eps = 0.5

update = 10
threshold = 0.5
percentage=50
T=10000
Q_int = 1
reward_alpha = 0.25
amount_of_sim = 300
rot=10
K=8
MLunchosen_alpha_mean = 0.5
MLunchosen_alpha_std = 0.1 
MLchosen_alpha_mean = 0.5
MLchosen_alpha_std = 0.1 
operation = '+'
update = 10

window = 50



unchosen_mean_int = 0
#MLunchosen_mean_int = np.log(unchosen_mean_int/(1-unchosen_mean_int))
ML_unchosen_mean_int = unchosen_mean_int
unchosen_std_int = 1
MLunchosen_std_int = unchosen_std_int

chosen_mean_int = 0
#MLchosen_mean_int = np.log(chosen_mean_int/(1-chosen_mean_int))
MLchosen_mean_int = chosen_mean_int
chosen_std_int = 1
MLchosen_std_int = chosen_std_int

mean_start = 9000

#################################################################
#SIMULATIONS
#################################################################

#variable context:
#for time plots:
r_var_cumsum = np.zeros(T)
r_var = np.zeros(T)

g_var = np.zeros(T)
g_mean_var = np.zeros(T)
g_std_var = np.zeros(T)

l_var = np.zeros(T)
l_mean_var = np.zeros(T)
l_std_var = np.zeros(T)

total_uc_var = np.zeros(amount_of_sim)
total_c_var = np.zeros(amount_of_sim)

reward_var = np.zeros(amount_of_sim)
#simulation:
for sim in range(amount_of_sim):
    k, r, Q_k_stored, unchosen_var, unchosen_var_mean, unchosen_var_std, chosen_var, chosen_var_mean, chosen_var_std, MLunchosen_stored, MLunchosen_mean_stored, MLunchosen_std_stored, MLchosen_stored, MLchosen_mean_stored, MLchosen_std_stored = simulate_RW_MLlambda_has8 (Q_alpha=Q_alpha, eps=eps, reward_alpha=reward_alpha, MLunchosen_alpha_mean=MLunchosen_alpha_mean, MLunchosen_alpha_std=MLunchosen_alpha_std, MLchosen_alpha_mean=MLchosen_alpha_mean, MLchosen_alpha_std=MLchosen_alpha_std, T=T, Q_int=Q_int, MLunchosen_mean_int=ML_unchosen_mean_int, MLunchosen_std_int=MLunchosen_std_int, MLchosen_mean_int=MLchosen_mean_int, MLchosen_std_int=MLchosen_std_int, scale=1, update=update, operation=operation, percentile=percentile)
    #for time plot:
    r_cumsum = np.cumsum(r)
    r_cumsum_av = np.zeros(len(r))
    for nr, i in enumerate(r):
        divide = nr+1
        r_cumsum_av[nr] = r_cumsum[nr]/divide

    reward_var[sim] = np.mean(r[mean_start:])
    total_uc_var[sim] = np.mean(unchosen_var[mean_start:])
    total_c_var[sim] = np.mean(chosen_var[mean_start:])

    r_var_cumsum = r_var_cumsum + r_cumsum_av
    r_var = r_var + r

    g_var = g_var + unchosen_var
    g_mean_var = g_mean_var + unchosen_var_mean
    g_std_var = g_std_var + unchosen_var_std

    l_var = l_var + chosen_var
    l_mean_var = l_mean_var + chosen_var_mean
    l_std_var = l_std_var + chosen_var_std


#for average:
av_uc_var = np.mean(total_uc_var)
std_uc_var = np.std(total_uc_var)
av_c_var = np.mean(total_c_var)
std_c_var = np.std(total_c_var)
#for time plot:
r_var_cumsum_end = np.divide(r_var_cumsum, amount_of_sim)
r_var_end = np.divide(r_var, amount_of_sim)

g_var_end= np.divide(g_var, amount_of_sim)
g_mean_var_end = np.divide(g_mean_var, amount_of_sim)
g_std_var_end = np.divide(g_std_var, amount_of_sim)

l_var_end= np.divide(l_var, amount_of_sim)
l_mean_var_end = np.divide(l_mean_var, amount_of_sim)
l_std_var_end = np.divide(l_std_var, amount_of_sim)

av_reward_var = np.mean(reward_var)
std_reward_var = np.std(reward_var)

#stable context:
#for time plots:
r_sta_cumsum = np.zeros(T)
r_sta = np.zeros(T)

g_sta = np.zeros(T)
g_mean_sta = np.zeros(T)
g_std_sta = np.zeros(T)

l_sta = np.zeros(T)
l_mean_sta = np.zeros(T)
l_std_sta = np.zeros(T)

total_uc_sta = np.zeros(amount_of_sim)
total_c_sta = np.zeros(amount_of_sim)

reward_sta = np.zeros(amount_of_sim)
#simulation:
for sim in range(amount_of_sim):
    k, r, Q_k_stored, unchosen_sta, unchosen_sta_mean, unchosen_sta_std, chosen_sta, chosen_sta_mean, chosen_sta_std, MLunchosen_stored, MLunchosen_mean_stored, MLunchosen_std_stored, MLchosen_stored, MLchosen_mean_stored, MLchosen_std_stored = simulate_RW_MLlambda_stable(Q_alpha=Q_alpha, eps=eps, reward_alpha=reward_alpha, MLunchosen_alpha_mean=MLunchosen_alpha_mean, MLunchosen_alpha_std=MLunchosen_alpha_std, MLchosen_alpha_mean=MLchosen_alpha_mean, MLchosen_alpha_std=MLchosen_alpha_std, T=T, Q_int=Q_int, MLunchosen_mean_int=ML_unchosen_mean_int, MLunchosen_std_int=MLunchosen_std_int, MLchosen_mean_int=MLchosen_mean_int, MLchosen_std_int=MLchosen_std_int, scale=1, update = 10,reward_stable=reward_stable, operation = operation)
    #for time plot:
    r_cumsum = np.cumsum(r)
    r_cumsum_av = np.zeros(len(r))
    for nr, i in enumerate(r):
        divide = nr+1
        r_cumsum_av[nr] = r_cumsum[nr]/divide

    reward_sta[sim] = np.mean(r[mean_start:])

    total_uc_sta[sim] = np.mean(unchosen_sta[mean_start:])
    total_c_sta[sim] = np.mean(chosen_sta[mean_start:])

    r_sta_cumsum = r_sta_cumsum + r_cumsum_av
    r_sta = r_sta + r

    g_sta = g_sta + unchosen_sta
    g_mean_sta = g_mean_sta + unchosen_sta_mean
    g_std_sta = g_std_sta + unchosen_sta_std

    l_sta = l_sta + chosen_sta
    l_mean_sta = l_mean_sta + chosen_sta_mean
    l_std_sta = l_std_sta + chosen_sta_std


#for average:
av_uc_sta = np.mean(total_uc_sta)
std_uc_sta = np.std(total_uc_sta)
av_c_sta = np.mean(total_c_sta)
std_c_sta = np.std(total_c_sta)

#for time plot:
r_sta_cumsum_end = np.divide(r_sta_cumsum, amount_of_sim)
r_sta_end = np.divide(r_sta, amount_of_sim)

g_sta_end= np.divide(g_sta, amount_of_sim)
g_mean_sta_end = np.divide(g_mean_sta, amount_of_sim)
g_std_sta_end = np.divide(g_std_sta, amount_of_sim)

l_sta_end= np.divide(l_sta, amount_of_sim)
l_mean_sta_end = np.divide(l_mean_sta, amount_of_sim)
l_std_sta_end = np.divide(l_std_sta, amount_of_sim)

av_reward_sta = np.mean(reward_sta)
std_reward_sta = np.std(reward_sta)

#volatile context:
#for time plots:
r_vol_cumsum = np.zeros(T)
r_vol = np.zeros(T)

g_vol = np.zeros(T)
g_mean_vol = np.zeros(T)
g_std_vol = np.zeros(T)

l_vol = np.zeros(T)
l_mean_vol = np.zeros(T)
l_std_vol = np.zeros(T)

total_uc_vol = np.zeros(amount_of_sim)
total_c_vol = np.zeros(amount_of_sim)

reward_vol = np.zeros(amount_of_sim)
#simulation:
for sim in range(amount_of_sim):
    k, r, Q_k_stored, unchosen_vol, unchosen_vol_mean, unchosen_vol_std, chosen_vol, chosen_vol_mean, chosen_vol_std, MLunchosen_stored, MLunchosen_mean_stored, MLunchosen_std_stored, MLchosen_stored, MLchosen_mean_stored, MLchosen_std_stored = simulate_RW_MLlambda_volatile(Q_alpha=Q_alpha, eps=eps, reward_alpha=reward_alpha, MLunchosen_alpha_mean=MLunchosen_alpha_mean, MLunchosen_alpha_std=MLunchosen_alpha_std, MLchosen_alpha_mean=MLchosen_alpha_mean, MLchosen_alpha_std=MLchosen_alpha_std, T=T, Q_int=Q_int, MLunchosen_mean_int=ML_unchosen_mean_int, MLunchosen_std_int=MLunchosen_std_int, MLchosen_mean_int=MLchosen_mean_int, MLchosen_std_int=MLchosen_std_int, rot=rot, scale=1, update = update, reward_volatile=reward_volatile, operation=operation)    #for time plot:
    r_cumsum = np.cumsum(r)
    r_cumsum_av = np.zeros(len(r))
    for nr, i in enumerate(r):
        divide = nr+1
        r_cumsum_av[nr] = r_cumsum[nr]/divide

    reward_vol[sim] = np.mean(r[mean_start:])
    total_uc_vol[sim] = np.mean(unchosen_vol[mean_start:])
    total_c_vol[sim] = np.mean(chosen_vol[mean_start:])

    r_vol_cumsum = r_vol_cumsum + r_cumsum_av
    r_vol = r_vol + r

    g_vol = g_vol + unchosen_vol
    g_mean_vol = g_mean_vol + unchosen_vol_mean
    g_std_vol = g_std_vol + unchosen_vol_std

    l_vol = l_vol + chosen_vol
    l_mean_vol = l_mean_vol + chosen_vol_mean
    l_std_vol = l_std_vol + chosen_vol_std

#for average:
av_uc_vol = np.mean(total_uc_vol)
std_uc_vol = np.std(total_uc_vol)
av_c_vol = np.mean(total_c_vol)
std_c_vol = np.std(total_c_vol)

#for time plot:
r_vol_cumsum_end = np.divide(r_vol_cumsum, amount_of_sim)
r_vol_end = np.divide(r_vol, amount_of_sim)

g_vol_end= np.divide(g_vol, amount_of_sim)
g_mean_vol_end = np.divide(g_mean_vol, amount_of_sim)
g_std_vol_end = np.divide(g_std_vol, amount_of_sim)

l_vol_end= np.divide(l_vol, amount_of_sim)
l_mean_vol_end = np.divide(l_mean_vol, amount_of_sim)
l_std_vol_end = np.divide(l_std_vol, amount_of_sim)

av_reward_vol = np.mean(reward_vol)
std_reward_vol = np.std(reward_vol)

#################################################################
#PLOTTING
#################################################################

save_dir_first = '/Users/jareynde/OneDrive - UGent/1A_Main_Project/Models/1Simulations/Env_HideAndSeek/output/lambda-8choice'
if operation == '*':
    stri = 'mult'
else: 
    stri = 'add'
new_sim_folder = f'sim{sim_nr}_{stri}'
save_dir = os.path.join(save_dir_first, new_sim_folder)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


store_av_eps = {
    'av_c_var' : av_c_var,
    'std_c_var' : std_c_var,
    'av_c_sta' : av_c_sta,
    'std_c_sta' : std_c_sta,
    'av_c_vol' : av_c_vol,
    'std_c_vol' : std_c_vol,

    'av_uc_var' : av_uc_var,
    'std_uc_var' : std_uc_var,
    'av_uc_sta' : av_uc_sta,
    'std_uc_sta' : std_uc_sta,
    'av_uc_vol' : av_uc_vol,
    'std_uc_vol' : std_uc_vol,


}

title_excel = os.path.join(save_dir, f'sim{sim_nr}av_c_uc.xlsx')
df = pd.DataFrame(data=store_av_eps, index=[1])
df.to_excel(title_excel, index=False)

#time plots:

time = np.linspace(1, T, T, endpoint=True)

fig_name = os.path.join(save_dir, f'sim{sim_nr}_unchosen-chosen_compare_contexts')
fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(10, 21))
unchosensta, = ax1.plot(time, g_sta_end, label=f'unchosen (unchosen options) stable environment')
unchosenvol, = ax2.plot(time, g_vol_end, label=f'unchosen (unchosen options) volatile environment')
unchosenvar, = ax3.plot(time, g_var_end, label=f'unchosen (unchosen options) hypervolatile environment')

chosensta, = ax1.plot(time, l_sta_end, label=f'chosen (chosen option) stable environment')
chosenvol, = ax2.plot(time, l_vol_end, label=f'chosen (chosen option) volatile environment')
chosenvar, = ax3.plot(time, l_var_end, label=f'chosen (chosen option) hypervolatile environment')
ax1.legend(handles=[unchosensta, chosensta])
ax2.legend(handles=[unchosenvol, chosenvol])
ax3.legend(handles=[unchosenvar, chosenvar])

ax1.set_xlabel('trials')
ax1.set_ylabel('chosen/unchosen')
ax2.set_xlabel('trials')
ax2.set_ylabel('chosen/unchosen')
ax3.set_xlabel('trials')
ax3.set_ylabel('chosen/unchosen')

#ax12.set_title(f'meta-learning of epsilon, based on {amount_of_sim} simulations')
plt.savefig(fig_name)
#plt.show()

f2, (ax4, ax5) = plt.subplots(2, 1, figsize=(10,14))
reward_sta_cum, = ax4.plot(time, r_sta_cumsum_end, label=f'stable context')
reward_sta, = ax5.plot(time, r_sta_end, label=f'stable context')

reward_vol_cum, = ax4.plot(time, r_vol_cumsum_end, label=f'volatile context')
reward_vol, = ax5.plot(time, r_vol_end, label=f'volatile context')

reward_var_cum, = ax4.plot(time, r_var_cumsum_end, label=f'variable context')
reward_var, = ax5.plot(time, r_var_end, label=f'variable context')

ax4.legend(handles=[reward_sta_cum, reward_vol_cum, reward_var_cum])
ax5.legend(handles=[reward_sta, reward_vol, reward_var])
fig_name = os.path.join(save_dir, f'sim{sim_nr}_reward_compare_contexts')
ax2.set_xlabel('trials')
ax2.set_xlabel('cumulative reward')
ax3.set_xlabel('trials')
ax3.set_ylabel('reward')
plt.savefig(fig_name)
#plt.show()

#SNE paper figure:
fig_name = os.path.join(save_dir, f'sim{sim_nr}SNE_unchosen_plot')
fig, ax12 = plt.subplots(figsize=(6, 3))
unchosensta, = ax12.plot(time, g_sta_end, label=f'stable environment')
unchosenvol, = ax12.plot(time, g_vol_end, label=f'volatile environment')
unchosenvar, = ax12.plot(time, g_var_end, label=f'variable environment')
#ax12.legend(handles=[unchosensta, unchosenvol, unchosenvar])
ax12.set_xlabel('trials', fontsize=18)
ax12.set_ylabel('unchosen value addition', fontsize=18)
plt.xlim([0, 10000])
plt.ylim([-3,3])
plt.yticks(fontsize=15)
plt.xticks(fontsize = 15)
#ax12.set_title(f'meta-learning of learning rate, based on {amount_of_sim} simulations')
plt.savefig(fig_name)
#plt.show()


#SNE paper figure:
fig_name = os.path.join(save_dir, f'sim{sim_nr}SNE_chosen_plot')
fig, ax12 = plt.subplots(figsize=(6, 3))
chosensta, = ax12.plot(time, l_sta_end, label=f'stable environment')
chosenvol, = ax12.plot(time, l_vol_end, label=f'volatile environment')
chosenvar, = ax12.plot(time, l_var_end, label=f'variable environment')
#ax12.legend(handles=[chosensta, chosenvol, chosenvar])
ax12.set_xlabel('trials', fontsize=18)
ax12.set_ylabel('chosen value addition', fontsize=18)
plt.xlim([0, 10000])
plt.ylim([-3,3])
plt.yticks(fontsize=15)
plt.xticks(fontsize = 15)
#ax12.set_title(f'meta-learning of learning rate, based on {amount_of_sim} simulations')
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


title_excel = os.path.join(save_dir, f'sim{sim_nr}av_rewards_lambda_ic{unchosen_mean_int}.xlsx')
df = pd.DataFrame(data=store_av_reward, index=[1])
df.to_excel(title_excel, index=False)




K=len(reward_stable)

store_param_values = {
    'simulation number' : sim_nr,
    'amount of trials' : T,
    'amount of simulations' : amount_of_sim,
    'amount of choice options' : K,
    'amount of trials after which meta-learning parameters are updated' : update,
    'initial Q-value' : Q_int,
    'learning rate for Q-value' : Q_alpha,
    'epsilon' : eps,
    'learning rate for the mean of the unchosen meta-learning parameter' : MLunchosen_alpha_mean,
    'learning rate for the std of the unchosen meta-learning parameter' : MLunchosen_alpha_std,
    'learning rate for the mean of the chosen meta-learning parameter' : MLchosen_alpha_mean,
    'learning rate for the std of the chosen meta-learning parameter' : MLchosen_alpha_std,
    'initial mean unchosen value' : unchosen_mean_int,
    'initial standard deviation of unchosen value' : unchosen_std_int,
    'initial mean chosen value' : chosen_mean_int,
    'initial standard deviation of chosen value' : chosen_std_int,
    'amount of trials after which reward probabilities are shuffled in volatile context' : rot,
    'percentage of least freuquently occuring responses are rewarded': percentile,
    'reward probabilities in stable context' : reward_st,
    'reward probabiities in volatile context' : reward_vl    }

title_excel = os.path.join(save_dir, f'sim{sim_nr}a_fixed_parameter_values.xlsx')
df = pd.DataFrame(data=store_param_values, index=[1])
df.to_excel(title_excel, index=False)

             