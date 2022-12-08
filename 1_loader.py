#작동하는 놈 
#source /Users/yoon/Documents/python/FitML/RL_stable/bin/activate


import gym
import pybullet
import pybullet_envs
import torch as th

import stable_baselines3 as sb3
from sb3_contrib import TRPO
from stable_baselines3.common.evaluation import evaluate_policy

#env = gym.make("HumanoidBulletEnv-v0")
env = gym.make("AntBulletEnv-v0")

#redering or not
env.render()
# load ppo
#model = sb3.PPO.load("z_sac_ac_50_sp",env)
#model = sb3.PPO.load("z_ppo_Humanoid_t_(6)",env)
# load a2c
#model = sb3.A2C.load("z_a2c_ant_350",env)
# load ddpg
#model = sb3.DDPG.load("z_ddpg_ant_180",env)
# load trpo
#model = TRPO.load("z_trpo_ant_700",env)

# load sac
#model = sb3.SAC.load("y_sac_ant_40_795",env)
model = sb3.SAC.load("SAC/MLp/sac_Mlp+24",env)


mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=5)
print("=============================================================")
print("@ mean_reward : " , mean_reward)
print("@ std_reward : " , std_reward)


# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
# Watch Video
obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()