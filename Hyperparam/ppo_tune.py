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

param = {'target_kl':[0.06,0.03,0.01],'pi_lr':[6e-4,3e-4,1e-4]}

args = argsparser()
seeds = range(3)

logger.configure('./data', ['csv'], log_suffix='Hopper-naive-ppo-tune')

for values in list(itertools.product( param['target_kl'], param['pi_lr'])):
    returns = []
    for seed in seeds:
        args.seed = seed

        checkpoint = 4000
        result = ppo(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=args.hid), gamma=args.gamma,
        seed=seed, steps_per_epoch=args.steps, epochs=args.epochs,
        target_kl=values[0],pi_lr = values[1],naive=True)

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