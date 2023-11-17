#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rescorla Wagner model with epsilon-greedy decision making policy
Constant epsilon
Contexts:
*Stable: reward probabiliies stay the same during all trials
*Volatile: reward probabilities shuffle every couple of trials
*Reinforced variability: reward dependent on how variable options are chosen according to 
                         Hide And Seek game
@author: Janne Reynders; janne.reynders@ugent.be
"""
import numpy as np                  
import pandas as pd                 
import matplotlib.pyplot as plt     
import os
import random

#simulate a Rescorla Wagner model with constant epsilon in a stable context
def simulate_RW_stable(alpha, eps, chosen, unchosen, operation, T, Q_int, reward_sta):
    reward_prob=reward_sta
    K=len(reward_prob) #amont of choice options
    #alpha      --->        learning rate
    #eps        --->        epsilon
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #reward prob -->        probabilites to recieve a reward, associated to each option

    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.zeros((T), dtype=float) #vector of rewards

    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    
    
    for t in range(T):
        
        # store values for Q
        Q_k_stored[t,:] = Q_k   
              
        # make choice based on choice probababilities
        rand[t] = np.random.choice([0,1], p=[1-eps,eps])
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
                Q_k[k[t]] = Q_k[k[t]] + alpha * delta_k
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

    return k, r, Q_k_stored

#simulate a Rescorla Wagner model with constant epsilon in a volatile context
def simulate_RW_volatile(alpha, eps, chosen, unchosen, operation, T, Q_int, reward_vol):
    reward_prob=reward_vol
    K=len(reward_prob) #amont of choice options
    #alpha      --->        learning rate
    #eps        --->        epsilon
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #reward prob -->        probabilites to recieve a reward, associated to each option
    #rot            --->        amount of trials after which mean reward values rotate among choice options

    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.zeros((T), dtype=float) #vector of rewards

    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    
    first_update = 15
    for t in range(T):
        
        # store values for Q
        Q_k_stored[t,:] = Q_k   
              
        # make choice based on choice probababilities
        rand[t] = np.random.choice([0,1], p=[1-eps,eps])
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
                Q_k[k[t]] = Q_k[k[t]] + alpha * delta_k
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

    return k, r, Q_k_stored

#simulate a Rescorla Wagner model with constant epsilon in a variable context
def simulate_RW_variable(alpha, eps, chosen, unchosen, operation, T, Q_int, percentile):
    seq_options = np.array([[a, b] for a in range(8) for b in range(8)])
    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    Freq = np.random.uniform(0.9,1.1,K_seq)

    #alpha      --->        learning rate
    #eps        --->        epsilon
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)
    #reward prob -->        probabilites to recieve a reward, associated to each option
    #threshold      --->        % of least frequently chosen options which will be rewarded

    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.zeros((T), dtype=float) #vector of rewards

    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    
    
    for t in range(T):
        
        # store values for Q
        Q_k_stored[t,:] = Q_k   
              
        # make choice based on choice probababilities
        rand[t] = np.random.choice([0,1], p=[1-eps,eps])
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
                Q_k[k[t]] = Q_k[k[t]] + alpha * delta_k
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

    return k, r, Q_k_stored

sim_nr = 'SNE'
reward_stable = [0.70,0.70,0.70,0.30,0.30,0.30,0.30,0.30]
reward_st = '3-5 70-30'
reward_volatile = [0.90,0.90,0.90,0.10,0.10,0.10,0.10,0.10]
reward_vl = '3-5 90-10'
percentile = 60
alpha = 0.5
eps = 0.5
chosen = 1
unchosen = 1

T=10000
Q_int = 1
amount_of_sim = 300
rot=10
K=10

operation = '*'



#################################################################
#SIMULATIONS
#################################################################
mean_start = 9000

reward_sta = np.zeros(amount_of_sim)
#for time plots:
r_sta_cumsum = np.zeros(T)
r_sta = np.zeros(T)
for sim in range(amount_of_sim):
    k, r, Q_k_stored = simulate_RW_stable(alpha = alpha, eps=eps, chosen=chosen, unchosen=unchosen,operation=operation, T=T, Q_int=Q_int, reward_sta=reward_stable)
    reward_sta[sim] = np.mean(r[mean_start:])
    #for time plots:
    r_cumsum = np.cumsum(r)
    r_cumsum_av = np.zeros(len(r))
    for nr, i in enumerate(r):
        divide = nr+1
        r_cumsum_av[nr] = r_cumsum[nr]/divide
    r_sta_cumsum = r_sta_cumsum + r_cumsum_av
    r_sta = r_sta + r
av_reward_sta = np.mean(reward_sta)
std_reward_sta = np.mean(reward_sta)
r_sta_cumsum_end = np.divide(r_sta_cumsum, amount_of_sim)
r_sta_end = np.divide(r_sta, amount_of_sim)


reward_vol = np.zeros(amount_of_sim)
#for time plots:
r_vol_cumsum = np.zeros(T)
r_vol = np.zeros(T)
for sim in range(amount_of_sim):
    k, r, Q_k_stored = simulate_RW_volatile(alpha = alpha, eps= eps, chosen=chosen, unchosen=unchosen,operation=operation, T=T, Q_int=Q_int, reward_vol=reward_volatile)
    reward_vol[sim] = np.mean(r[mean_start:])
    #for time plots:
    r_cumsum = np.cumsum(r)
    r_cumsum_av = np.zeros(len(r))
    for nr, i in enumerate(r):
        divide = nr+1
        r_cumsum_av[nr] = r_cumsum[nr]/divide
    r_vol_cumsum = r_vol_cumsum + r_cumsum_av
    r_vol = r_vol + r
av_reward_vol = np.mean(reward_vol)
std_reward_vol = np.std(reward_vol)
r_vol_cumsum_end = np.divide(r_vol_cumsum, amount_of_sim)
r_vol_end = np.divide(r_vol, amount_of_sim)


reward_var = np.zeros(amount_of_sim)
#for time plots:
r_var_cumsum = np.zeros(T)
r_var = np.zeros(T)
for sim in range(amount_of_sim):
    k, r, Q_k_stored = simulate_RW_variable(alpha= alpha, eps=eps, chosen=chosen, unchosen=unchosen,operation=operation, T=T, Q_int=Q_int,percentile=percentile)
    reward_var[sim] = np.mean(r[mean_start:])
    #for time plot:
    r_cumsum = np.cumsum(r)
    r_cumsum_av = np.zeros(len(r))
    for nr, i in enumerate(r):
        divide = nr+1
        r_cumsum_av[nr] = r_cumsum[nr]/divide
    r_var_cumsum = r_var_cumsum + r_cumsum_av
    r_var = r_var + r
av_reward_var = np.mean(reward_var)
std_reward_var = np.std(reward_var)
r_var_cumsum_end = np.divide(r_var_cumsum, amount_of_sim)
r_var_end = np.divide(r_var, amount_of_sim)



################################################################################################

save_dir = '/Users/jareynde/OneDrive - UGent/1A_Main_Project/Models/1Simulations/Env_HideAndSeek/output/constant-8choice'

time = np.linspace(1, T, T, endpoint=True)

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
    'global_ste_reward' : global_ste_reward,
    'reward schedule stable' : reward_st,
    'reward schedule volatle' : reward_vl,
    'percentile in variable' : percentile
}

title_excel = os.path.join(save_dir, f'sim{sim_nr}av_rewards_eps{eps}-2.xlsx')
df = pd.DataFrame(data=store_av_reward, index=[1])
df.to_excel(title_excel, index=False)

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
f4.suptitle(f'rewards in each context averaged over {amount_of_sim} simulations')
plt.savefig(fig_name)
plt.show()


