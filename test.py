import gym
import numpy as np

import pybullet as p
import pybullet_envs
import csv

#env = gym.make('AntBulletEnv-v0')

k=[]

k.append(1)
k.append(2)
k.append(3)
p={0:1,1:2,2:3}
p[0]+=1
print(p)

with open('picker_Selection_9.csv','w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=p)
    writer.writeheader()
    writer.writerow(p)