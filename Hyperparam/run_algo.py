import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
granddir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
from spinup.algos.pytorch.ppo.ppo import argsparser,weighted_ppo,ppo
from spinup.algos.pytorch.ppo import core
sys.path.insert(0,granddir)
from Components import logger
import itertools
import numpy as np
import gym

args = argsparser()

result = weighted_ppo(lambda: gym.make(args.env), actor_critic=core.MLPWeightedActorCritic, gamma=args.gamma, target_kl=0.03,pi_lr=6e-4,
seed=1, steps_per_epoch=args.steps, epochs=args.epochs,vf_lr=0.003,scale=20,gamma_coef=1.0)
