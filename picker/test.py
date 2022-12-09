from DQN import DQN, ReplayMemory as DQN
import random
import math


env = gym.make("AntBulletEnv-v0")

#redering or not
#env.render()

model_0 = sb3.PPO.load("PPO/mlp/ppo_Mlp+8",env)
model_1 = TRPO.load("TRPO/mlp/trpo_Mlp+14",env)
model_2 = sb3.SAC.load("SAC/mlp/sac_Mlp+28",env)

model_picker = DQN.DQN(28,1,3)
obs = env.reset()
picker_reward = 0
picker_memory = DQN.ReplayMemory(10000)

for t in range(2710):
    picker_state=obs
    picking_model = model_picker.predict(picker_state)
    for i in range(10):
        if picking_model == 0:
            action, _states = model_0.predict(obs, deterministic=True)
        elif picking_model == 1:
            action, _states = model_1.predict(obs, deterministic=True)
        elif picking_model == 2:
            action, _states = model_2.predict(obs, deterministic=True)

        obs, rewards, dones, info = env.step(action)
        picker_reward +=rewards
    picker_next_state = obs
    picker_memory.push(picker_state, picking_model, picker_next_state, picker_reward) 
    

    env.render()