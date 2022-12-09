import gym #0.26, 0.21<-stable
import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pybullet
import pybullet_envs
import stable_baselines3 as sb3
from sb3_contrib import TRPO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

env = gym.make('AntBulletEnv-v0')
#env = gym.make('CartPole-v1')
# Get number of actions from gym action space
#n_actions = env.action_space
n_actions = 3
#n_actions = 4
#state, _ = env.reset(return_info=True)
state = env.reset()

# Get the number of state observations
#n_observations = env.observation_space.shape[0]
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(100000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            k=policy_net(state).max(1)[1].view(1, 1)
            return int(k)
    else:
        k=np.random.randint(3)
        return int(k)

episode_durations = []


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
 
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()



# if torch.cuda.is_available():
#     num_episodes = 600
# else:
#     num_episodes = 50
model_0 = sb3.PPO.load("PPO/mlp/ppo_Mlp+8",env)
model_1 = TRPO.load("TRPO/mlp/trpo_Mlp+14",env)
model_2 = sb3.SAC.load("SAC/mlp/sac_Mlp+28",env)


for i_episode in range(300):

    #state, _ = env.reset(return_info=True)
    state = env.reset()
    obs = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    pick_action = select_action(obs)
    privious_state=state
    picker_reward = 0
    for ii in range(10):
        if pick_action==0:
            action = model_0.predict(state, deterministic=True)
        elif pick_action==1:
            action = model_1.predict(state, deterministic=True)
        elif pick_action==2:
            action = model_2.predict(state, deterministic=True)

        action=np.array(action[0],dtype=np.float32)

        right_before_state = state
        state, rewards, dones, info = env.step(action)
        picker_reward +=rewards
        if dones:
            break
    if dones:
        picker_next_state = None
    else:
        picker_next_state = state
    
    #picker_next_state=torch.tensor(picker_next_state, dtype=torch.float32, device=device).unsqueeze(0)
    #picker_reward=torch.tensor(picker_reward, dtype=torch.float32, device=device).unsqueeze(0)
    
    memory.push(privious_state, pick_action, picker_next_state, picker_reward)
    state = picker_next_state

    # Perform one step of the optimization (on the policy network)
    optimize_model()

    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    target_net.load_state_dict(target_net_state_dict)

    if dones:
        #episode_durations.append(t + 1)
        break


print('Complete')