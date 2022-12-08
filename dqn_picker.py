#pyenv activate scratch

import gym
import pybullet
import pybullet_envs
import torch as th

import stable_baselines3 as sb3
from sb3_contrib import TRPO
from stable_baselines3.common.evaluation import evaluate_policy