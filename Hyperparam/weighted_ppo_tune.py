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

param = {'vf_lr': [3e-3,1e-3,6e-4,3e-4,1e-4],'scale':[1,10,20,40],'gamma_coef':[5,1,0.2]}

args = argsparser()
seeds = range(5)

logger.configure('./data', ['csv'], log_suffix='Hopper-weighted-ppo-tune')

for values in list(itertools.product(param['vf_lr'], param['scale'], param['gamma_coef'])):
    returns = []
    for seed in seeds:
        args.seed = seed

        checkpoint = 4000
        result = weighted_ppo(lambda: gym.make(args.env), actor_critic=core.MLPWeightedActorCritic,
        ac_kwargs=dict(hidden_sizes=args.hid), gamma=args.gamma,
        seed=seed, steps_per_epoch=args.steps, epochs=args.epochs,vf_lr=values[0],
        scale=values[1],gamma_coef=values[2])

        ret = np.array(result)
        print(ret.shape)
        returns.append(ret)
        name = [str(k) for k in values]
        name.append(str(seed))
        print("hyperparam", '-'.join(name))
        logger.logkv("hyperparam", '-'.join(name))
        for n in range(ret.shape[0]):
            logger.logkv(str((n + 1) * checkpoint), ret[n])
        logger.dumpkvs()

    ret = np.array(returns)
    print(ret.shape)
    ret = np.mean(ret, axis=0)
    name = [str(k) for k in values]
    name.append('mean')
    logger.logkv("hyperparam", '-'.join(name))
    for n in range(ret.shape[0]):
        logger.logkv(str((n + 1) * checkpoint), ret[n])
    logger.dumpkvs()