import gym
import numpy as np

import pybullet as p
import pybullet_envs

#env = gym.make('AntBulletEnv-v0')

A=[[ 7, 59,  5, 17],
       [61, 47, 99, 16],
       [12, 85, 27, 62],
       [44, 96, 37, 30],
       [15, 17, 81, 85]]

n=5
#B=np.empty((0,6), int)

B=[]

count=0
for j in range(0,n,1):
    temp=[]
    for i in range(0,n,1):
        if count%2 == 0:
            if i%2==0 :
                temp.append(5)
            else  :
                temp.append(68)
        else :
            if i%2==1 :
                temp.append(5)
            else  :
                temp.append(68)
    B.append(temp)
    count+=1
k=np.array(B)
print(k)

# for row in A:
#     total = sum(row)
#     temp=np.array([])
#     if total == 0:
#         for i in range(len(row)):
#             temp=np.append(temp,float(1/len(row)))
#     else :
#         for i in range(len(row)):
#             temp=np.append(temp,float(row[i]/total))
#     B=np.append(B,np.array([temp]),axis=0)
    



#     print(row)
# row_sums = a.sum(axis=1)
# new_matrix = a / row_sums[:, np.newaxis]

# print(new_matrix)



# n_input_channels = env.observation_space.shape[0]
# output=env.action_space
# print(n_input_channels)
# print(output)
# print(output.shape[0])
# print(output.shape)

# k=np.array([1,2,3,4,5,6,7,8])
# print(k.shape)