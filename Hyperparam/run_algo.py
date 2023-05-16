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

result = weighted_ppo(lambda: gym.make(args.env), actor_critic=core.MLPWeightedActorCritic,
ac_kwargs=dict(hidden_sizes=args.hid), gamma=args.gamma, target_kl=0.01,pi_lr=3e-4,train_v_iters=80,
seed=1, steps_per_epoch=args.steps, epochs=args.epochs,vf_lr=0.001,scale=40,gamma_coef=3.0)
