import gym
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import time

from collections import deque
from agent import Agent, FloatTensor
from ReplayMemory import ReplayMemory, Transition
from torch.autograd import Variable


import gym
import pybullet
import pybullet_envs
import stable_baselines3 as sb3
from sb3_contrib import TRPO
from stable_baselines3.common.evaluation import evaluate_policy

    
def epsilon_annealing(i_epsiode, max_episode, min_eps: float):
    ##  if i_epsiode --> max_episode, ret_eps --> min_eps
    ##  if i_epsiode --> 1, ret_eps --> 1  
    slope = (min_eps - 1.0) / max_episode
    ret_eps = max(slope * i_epsiode + 1.0, min_eps)
    return ret_eps        

def save(directory, filename):
    torch.save(agent.q_local.state_dict(), '%s/%s_local.pth' % (directory, filename))
    torch.save(agent.q_target.state_dict(), '%s/%s_target.pth' % (directory, filename))

def run_episode(env, agent, eps):
    """Play an epsiode and train

    Args:
        env (gym.Env): gym environment (CartPole-v0)
        agent (Agent): agent will train and get action        
        eps (float): eps-greedy for exploration

    Returns:
        int: reward earned in this episode
    """

    model_0 = sb3.PPO.load("PPO/mlp/ppo_Mlp+8",env)
    model_1 = TRPO.load("TRPO/mlp/trpo_Mlp+14",env)
    model_2 = sb3.SAC.load("SAC/mlp/sac_Mlp+28",env)

    state = env.reset()
    done = False
    picker_reward = 0
    total_reward = 0

    for t in range(2710):
        picking_model = agent.get_action(FloatTensor([state]) , eps)
        print(picking_model)
        print("==================================")
        privious_state=state
        picker_reward = 0
 
        for i in range(10): #10 steps change the picking
            if picking_model == 0:
                action, _states = model_0.predict(state, deterministic=True)
            elif picking_model == 1:
                action, _states = model_1.predict(state, deterministic=True)
            else:
                action, _states = model_2.predict(state, deterministic=True)

            state, rewards, dones, info = env.step(action)
            picker_reward +=rewards
        picker_next_state = state

        agent.replay_memory.push(
            (FloatTensor([privious_state]), 
            FloatTensor([picking_model]),
            FloatTensor([picker_reward]),
            FloatTensor([picker_next_state]),
            FloatTensor([dones]) ))

        total_reward += picker_reward
        if len(agent.replay_memory) > BATCH_SIZE:
            print("ha")
            batch=agent.replay_memory.sample(BATCH_SIZE)
            agent.learn(batch,gamma)

        
        state = picker_next_state
 
    return total_reward

def train(agent, env, num_episodes, print_every, min_eps, max_eps_episode):
    scores = []
    scores_window = deque(maxlen=100)
    eps = 1.0
    for i_episode in range(1, num_episodes+1):
        eps = epsilon_annealing(i_episode, max_eps_episode, min_eps)
        score = run_episode(env, agent, eps)

        scores_window.append(score)
        scores.append(score)
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=threshold:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            save('models', 'checkpoint')
            break
    return scores

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    device = torch.device("cuda" if use_cuda else "cpu")

    BATCH_SIZE = 64  
    TAU = 0.005 # 1e-3   # for soft update of target parameters
    gamma = 0.99
    LEARNING_RATE = 0.001
    TARGET_UPDATE = 10

    num_episodes = 5000
    print_every = 10
    hidden_dim = 16 ## 12 ## 32 ## 16 ## 64 ## 16
    min_eps = 0.01
    max_eps_episode = 50

    env = gym.make("AntBulletEnv-v0")
        
    space_dim =  env.observation_space.shape[0] # n_spaces
    action_dim = 3 # n_actions pick among 3 existing policies
    print('input_dim: ', space_dim, ', output_dim: ', action_dim, ', hidden_dim: ', hidden_dim)

    threshold = env.spec.reward_threshold
    print('threshold: ', threshold)

    agent = Agent(space_dim, action_dim, hidden_dim)

    scores = train(agent, env, num_episodes, print_every, min_eps, max_eps_episode)
