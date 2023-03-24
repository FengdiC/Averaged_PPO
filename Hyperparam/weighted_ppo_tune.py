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

args = argsparser()
seeds = range(10)

# Torch Shenanigans fix
set_one_thread()

logger.configure(args.log_dir, ['csv'], log_suffix='weighted-ppo-tune-'+str(args.env))

returns = []
for seed in seeds:
    hyperparam = random_search(args.seed)
    hyperparam['gamma'] = args.gamma
    checkpoint = 4000
    result = weighted_ppo(lambda: gym.make(args.env), actor_critic=core.MLPWeightedActorCritic,
    ac_kwargs=dict(hidden_sizes=args.hid,critic_hidden_sizes=hyperparam['critic_hid']),epochs=args.epochs,
    gamma=hyperparam['gamma'], target_kl=hyperparam['target_kl'],vf_lr=hyperparam['vf_lr'],pi_lr = hyperparam["pi_lr"],
    seed=seed,scale=hyperparam['scale'],gamma_coef=hyperparam['gamma_coef'])

    ret = np.array(result)
    print(ret.shape)
    returns.append(ret)
    name = list(hyperparam.keys())
    name = [str(key)+'-'+str(hyperparam[key]) for key in name]
    name.append(str(seed))
    print("hyperparam", '-'.join(name))
    logger.logkv("hyperparam", '-'.join(name))
    for n in range(ret.shape[0]):
        logger.logkv(str((n + 1) * checkpoint), ret[n])
    logger.dumpkvs()
