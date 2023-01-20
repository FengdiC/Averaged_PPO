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
from random_search import random_search,set_one_thread

param = {'env':['Hopper-v4','Swimmer-v4','Ant-v4']}

args = argsparser()
seeds = range(10)
# Torch Shenanigans fix
set_one_thread()

logger.configure(args.log_dir, ['csv'], log_suffix='mujoco_ppo_naive_tuned='+str(args.seed))

for values in list(itertools.product(param['env'])):
    args.env = values[0]
    hyperparam = random_search(226)
    checkpoint = 4000
    result = ppo(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
                 ac_kwargs=dict(hidden_sizes=args.hid), gamma=hyperparam['gamma'], pi_lr=hyperparam["pi_lr"],
                 target_kl=hyperparam['target_kl'], vf_lr=hyperparam['vf_lr'], epochs=args.epochs,
                 seed=args.seed, naive=True)

    ret = np.array(result)
    print(ret.shape)
    name = [str(k) for k in values]
    name.append(str(args.seed))
    print("hyperparam", '-'.join(name))
    logger.logkv("hyperparam", '-'.join(name))
    for n in range(ret.shape[0]):
        logger.logkv(str((n + 1) * checkpoint), ret[n])
    logger.dumpkvs()
