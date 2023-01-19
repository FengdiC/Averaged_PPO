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

param = {'env':['Hopper-v4','Swimmer-v4','Ant-v4']}

args = argsparser()

logger.configure(args.log_dir, ['csv'], log_suffix='mujoco_ppo_weighted_simple='+str(args.seed))

for values in list(itertools.product(param['env'])):
    args.env = values[0]
    checkpoint = 4000
    result = weighted_ppo(lambda: gym.make(args.env), actor_critic=core.MLPWeightedActorCritic,
    ac_kwargs=dict(hidden_sizes=args.hid,critic_hidden_sizes=[256,256]),epochs=args.epochs,
    gamma=0.995, target_kl=0.28,vf_lr=0.0031,
    seed=args.seed,scale=20,gamma_coef=2.31)

    ret = np.array(result)
    print(ret.shape)
    name = [str(k) for k in values]
    name.append(str(args.seed))
    print("hyperparam", '-'.join(name))
    logger.logkv("hyperparam", '-'.join(name))
    for n in range(ret.shape[0]):
        logger.logkv(str((n + 1) * checkpoint), ret[n])
    logger.dumpkvs()
