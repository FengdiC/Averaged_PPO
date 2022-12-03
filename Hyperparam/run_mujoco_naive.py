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
seeds = range(10)

logger.configure(args.log_dir, ['csv'], log_suffix='mujoco_ppo_naive_tuned='+str(args.seed))

for values in list(itertools.product(param['env'])):
    args.gamma = 0.995
    args.hid = [64,64]
    args.steps = 5500
    args.pi_lr = 0.0005373214903241354
    result = ppo(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
                 ac_kwargs=dict(hidden_sizes=args.hid), pi_lr=args.pi_lr,
                 gamma=args.gamma, target_kl=0.13490548606305505, vf_lr=0.0021184424558797726,
                 seed=args.seed, steps_per_epoch=args.steps, epochs=364, naive=True)
    checkpoint = 5500

    ret = np.array(result)
    print(ret.shape)
    name = [str(k) for k in values]
    name.append(str(args.seed))
    print("hyperparam", '-'.join(name))
    logger.logkv("hyperparam", '-'.join(name))
    for n in range(ret.shape[0]):
        logger.logkv(str((n + 1) * checkpoint), ret[n])
    logger.dumpkvs()
