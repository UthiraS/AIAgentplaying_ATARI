#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys

import wandb

wandb.init(
    project="usivaraman-train_sample", 

    config = {
    "learning_rate": 1.5e-4,    
    "batch_size": 32,
    "epsidoes" : 100000

    }
)



import random
from collections import deque, namedtuple
from itertools import count
from typing import List

import math
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
# import torch.autograd as autograd 
from agent import Agent
from dqn_model import DQN
from environment import Environment
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

# wandb.log({"loss": loss})
# wandb.watch(DQN)
writer = SummaryWriter()

EPISODES = 400000
LEARNING_RATE = 1.5e-4  # alpha
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 5000
EPSILON = 0.02
EPSILON_END = 0.005
EPS_DECAY = 500
FINAL_EXPL_FRAME = 1000000
TARGET_UPDATE_FREQUENCY = 5000
SAVE_MODEL_AFTER = 5000
DECAY_EPSILON_AFTER = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
rew_buffer = deque([0.0], maxlen=100)
# USE_CUDA = torch.cuda.is_available()
# Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class ReplayBuffer(object):
    def __init__(self, capacity: int):
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args) -> None:
        """Save a transition"""
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        """Randomly sample 'batch_size' number of transitions from buffer"""
        # samples = random.sample(self.buffer, batch_size)
        # for elem in samples:
        #     self.buffer.remove(elem)
        # return samples

        samples = []
        for _ in range(batch_size):
            index = random.randrange(len(self.buffer))
            samples.append(self.buffer[index])
            del self.buffer[index]
        return samples

    def __len__(self):
        return len(self.buffer)

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        self.env = env
        self.actions = self.env.action_space.n
        input_channel = 4 
        
        self.Q_net = DQN(input_channel,self.actions).to(device)
        self.target_Q_net = DQN(input_channel, self.actions).to(device)
        self.target_Q_net.load_state_dict(self.Q_net.state_dict())
        self.optimizer = optim.Adam(self.Q_net.parameters(), lr=LEARNING_RATE)
        self.Q_net.eval()

        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.training_steps = 0

        self.epsilon = 1.0
        self.epsilon_lowerbound = 0.025
        self.total_train_episodes = 100000  # To make epsilon decay linearly to 2.5% only towards the end 
        self.episodes_for_epsilon_decay = 40000
        self.episodes = 0
        
        
        
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            self.Q_net.load_state_dict(torch.load('./vanilla_dueldqn_model.pth'))
            ###########################
            # YOUR IMPLEMENTATION HERE #
            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        pass
    
    
    def make_action(self, observation: np.ndarray, test: bool =True) -> int:
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # print ("Observation",observation)
        # print("Shape :" ,observation.shape)
        # state = np.asarray(observation, dtype = np.float32)/255
        # # image with HWC-layout (height, width, channels), while Pytorch requires CHW-layout. So we have to do np.transpose(image,(2,0,1)) for HWC->CHW transformation.
        # # converting to CHW for pytorch tensor 
        # state = state.transpose(2,0,1)
        # state = torch.from_numpy(state).unsqueeze(0)

        # print ("State after processing :",state)
        # # Get Q from network/model
        # Q_value = self.Q_net.forward(state)
        # action = torch.argmax(Q_value,dim=1)[0]
        # print("Seleected action :",action)
        # action.detach().item()
        # print("Selected action  after:",action)
        if random.random() > EPSILON :
            observation = observation/255
            observation = observation.transpose(2,0,1)
            observation   = torch.FloatTensor(np.float32(observation)).unsqueeze(0)
            q_value = self.Q_net.forward(observation)
            action  = q_value.max(1)[1].data[0]
            action = int(action.item())   
        else :
             action = random.randrange(4)

        # print("Seleected action :",action)
        ###########################
        # return action.detach().item()
        return action
    def get_eps_greedy_action(self, greedy_action: int, epsilon: float):
        """
        Take the deterministic action given by the network and return an epsilon
        greedy action.
        """
        probability = np.ones(self.actions) * epsilon / self.actions  # exploration
        probability[greedy_action] += 1 - epsilon  # exploitation
        return np.random.choice(np.arange(self.actions), p=probability)
    
    # def push(self):
    #     """ You can add additional arguments as you need. 
    #     Push new data to buffer and remove the old one if the buffer is full.
        
    #     Hints:
    #     -----
    #         you can consider deque(maxlen = 10000) list
    #     """
    #     ###########################
    #     # YOUR IMPLEMENTATION HERE #
        
    #     ###########################
        
        
    # def replay_buffer(self):
    #     """ You can add additional arguments as you need.
    #     Select batch from buffer.
    #     """
    #     ###########################
    #     # YOUR IMPLEMENTATION HERE #
        
        
        
    #     ###########################
    #     return 
        

    def train(self,no_of_episodes: int = EPISODES):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        reward_data =[]
        epi_data =[]
        while self.episodes <= self.total_train_episodes:

            # Linear Decay Formula: f(t) = C - r*t
            if self.epsilon > self.epsilon_lowerbound:
                self.epsilon = 1.0 - self.episodes*(1/self.episodes_for_epsilon_decay)
            epsilon = self.epsilon
            frames = 0
            self.episodes += 1  
            episode_reward = 0
            curr_state = self.env.reset()

            # for step in count():
            while True:
                # if epi_num  == DECAY_EPSILON_AFTER:  el
                    # epsilon = epsilon = (2.5/100)*epsilon
                # if epi_num > DECAY_EPSILON_AFTER:
                #     # epsilon = np.interp(step, [0, FINAL_EXPL_FRAME], [EPSILON, EPSILON_END])
                #     epsilon  = EPSILON_END + (EPSILON - EPSILON_END) * math.exp(-1. * step/EPS_DECAY)
                # else:
                #     epsilon = EPSILON
                action = self.get_eps_greedy_action(self.make_action(curr_state), epsilon)
                next_state, reward, done ,trans_prob,info  = self.env.step(action)

                # print("From Environment")
                # print("S,A,R,NS ",curr_state,action,reward,next_state)

                # Convert numpy arrays/int to tensors
                # curr_state_t = self.format_state(curr_state)
                curr_state_t = np.asarray(curr_state, dtype = np.float32)/255
                # image with HWC-layout (height, width, channels), while Pytorch requires CHW-layout. So we have to do np.transpose(image,(2,0,1)) for HWC->CHW transformation.
                # converting to CHW for pytorch tensor 
                curr_state_t = curr_state_t.transpose(2,0,1)
                curr_state_t = torch.from_numpy(curr_state_t).unsqueeze(0)
                next_state_t = np.asarray(next_state, dtype = np.float32)/255
                # image with HWC-layout (height, width, channels), while Pytorch requires CHW-layout. So we have to do np.transpose(image,(2,0,1)) for HWC->CHW transformation.
                # converting to CHW for pytorch tensor 
                next_state_t = next_state_t.transpose(2,0,1)
                next_state_t = torch.from_numpy(next_state_t).unsqueeze(0)
                # next_state_t = self.format_state(next_state)
                action_t = torch.tensor([action], device=device)
                reward_t = torch.tensor([reward], device=device)
                # frames += 1
                # if frames % self.skip_frames == 0:
                    # buff_elem = (s,a,r,s_next)
                    # self.buffer.append(buff_elem)
                self.buffer.push(curr_state_t, action_t, reward_t, next_state_t)
                # print("Current_state,Action,Reward,Next state :")
                # print(curr_state_t, action_t, reward_t, next_state_t)

                curr_state = next_state
                episode_reward += reward
                
                # print("reward :",reward)
                # wandb.log({"Episode_number":epi_num, "Reward ":reward})
                
                
                # Optimize
                self.optimize_model()

                if done:
                    rew_buffer.append(episode_reward)
                    break
            print("Mean :",np.mean(rew_buffer))
            # wandb.log({"Episode Reward":episode_reward}) 
            # wandb.log({"Episode Reward mean":np.mean(rew_buffer)}) 
            if self.episodes % 100 == 0:
                # writer.add_scalar("Mean reward(100) vs Episode", np.mean(rew_buffer), epi_num)
                reward_data.append(np.mean(rew_buffer))
                epi_data.append(self.episodes)
                data = [[x, y] for (x, y) in zip(epi_data,reward_data)]
                table = wandb.Table(data=data, columns = ["epi", "reward"])
                wandb.log({"my_custom_plot_id" : wandb.plot.line(table, "epi", "reward",
                            title="Mean Reward vs Episode Plot")})
            # Logging
            # writer.add_scalar("Epsilon vs Step", epsilon, step)
            # if epi_num % 100 == 0:
            #     writer.add_scalar("Mean reward(100) vs Episode", np.mean(rew_buffer), epi_num)
            #     writer.add_scalar(
            #         "Mean reward(100) vs Training steps",
            #         np.mean(rew_buffer),
            #         self.training_steps
            #     )

            if self.episodes % TARGET_UPDATE_FREQUENCY == 0:
                self.target_Q_net.load_state_dict(self.Q_net.state_dict())

            if self.episodes % SAVE_MODEL_AFTER == 0:
                torch.save(self.Q_net.state_dict(), "vanilla_dueldqn_model_revnov16_4.pth")

        torch.save(self.Q_net.state_dict(), "vanilla_dueldqn_model_revnov16_4.pth")
        print("Complete")
        # writer.flush()
        # writer.close()
        wandb.finish()
        ###########################

    def optimize_model(self)-> None:
        """
        """
        if len(self.buffer) < BUFFER_SIZE:
            return

        self.training_steps += 1

        transitions = self.buffer.sample(BATCH_SIZE)

        # Convert batch array of transitions to Transition of batch arrays
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)#.to(device)
        action_batch = torch.cat(batch.action)#.unsqueeze(1)#.to(device)
        reward_batch = torch.cat(batch.reward)#.unsqueeze(1)#.to(device)
        non_terminal_next_state_batch = torch.cat(
            [s for s in batch.next_state if s is not None]
        )#.to(device)

        sav_t = self.Q_net(state_batch)
        state_action_values = sav_t[torch.arange(sav_t.size(0)), action_batch]

        # Get state-action values
        non_terminal_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool
        )
        next_state_Q_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_Q_values[non_terminal_mask] = self.target_Q_net(
            non_terminal_next_state_batch
        ).max(1)[0].detach()

        # Q1 = torch.zeros(BATCH_SIZE, device=device)
        # Q1[non_terminal_mask] = self.Q_net(
        #     non_terminal_next_state_batch
        # ).max(1)[0].detach()

        # Q1_values = torch.argmax(Q1)

        # Q2 = torch.zeros(BATCH_SIZE, device=device)
        # Q2[Q1_values] = self.target_Q_net(
        #     non_terminal_next_state_batch
        # ).max(1)[0].detach()
        # Q_preds = self.Q_net(state_batch)
        
        # # #get Q values of action taken, shape (N,1)
        # Q_vals = Q_preds.gather(1, action_batch)
          
        # # """
        # # In double DQN, get Q values for next_state from both target and model
        # # """
        # # #get Q values from target and model
        # with torch.no_grad(): #don't want gradients for model
        #     model_pred = self.Q_net(non_terminal_next_state_batch)
        # target_pred = self.target_Q_net(non_terminal_next_state_batch)

        # """
        # then get the actions from the model predicted Q values, not the target
        # """
        # model_actions = model_pred.max(1)[1].unsqueeze(1)
        
        # """
        # then use these actions to get Q values from the target network
        # """
        # target_Q = target_pred.gather(1, model_actions) 
        
        # #tensor for placing target values
        # target_vals = torch.zeros(BATCH_SIZE, 1).to(device) 

        # """
        # target_vals now filled with Q values from target using model predicted actions
        # """
        # #fill in target values for non_terminal states
        # #the terminal states will stay initialized as zeros
        # target_vals[non_terminal_mask] = target_Q
            
        # ground_truth_q_values = reward_batch + (target_vals *GAMMA)

        # Compute the ground truth
        ground_truth_q_values = reward_batch + GAMMA*next_state_Q_values
        # ground_truth_q_values = reward_batch + GAMMA*Q2
        ground_truth_q_values = torch.reshape(
            ground_truth_q_values.unsqueeze(1),
            (1, BATCH_SIZE)
        )[0]


        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, ground_truth_q_values)

        # Optimize the model
        self.optimizer.zero_grad(set_to_none=True)  # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
        loss.backward()
        # print("Parameters ",self.Q_net.parameters())
        for param in self.Q_net.parameters():

            # print("Para :",param)
            if(param.grad is not None):
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
