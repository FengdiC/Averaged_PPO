import random
import pandas as pd

import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import sys
import inspect
"""
This file tries to compare the distance between initial and stationary distributions and see if 
the distance is related to the performance.

compute_correction: computes the stationary distributions
comput_c_D: computes the estimated corrections
"""

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import gym
from Components import logger
from reacher import DotReacherRepeat,DotReacher
from counterexample import TwoState

from torch.optim import Adam
import time
import spinup.algos.pytorch.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

from scipy.stats import norm

def set_one_thread():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

def random_search(seed):
    rng = np.random.RandomState(seed=seed)

    # # choice 1
    # pi_lr = rng.choice([3,9,20,30,40,50])/10000.0
    # choice 2
    pi_lr = rng.choice([3,10, 30]) / 10000.0
    gamma_coef = rng.randint(low=50, high=500)/100.0
    scale = rng.randint(low=1, high=150)
    target_kl = rng.randint(low=0.01*100, high=0.3*100)/100.0
    vf_lr = rng.randint(low=3, high=50)/10000.0
    gamma = rng.choice([0.95,0.97,0.99,0.995])
    hid = np.array([[16,16],[32,32],[64,64]])
    critic_hid = rng.choice(range(hid.shape[0]))
    critic_hid = hid[critic_hid]

    hyperparameters = {"pi_lr":pi_lr,"gamma_coef":gamma_coef, "scale":scale, "target_kl":target_kl,
                       "vf_lr":vf_lr,"critic_hid":list(critic_hid),"gamma":gamma}

    return hyperparameters

def argsparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./')
    parser.add_argument('--env', type=str, default='Hopper-v4')
    parser.add_argument('--hid', type=list, default=[64,32])
    parser.add_argument('--critic_hid', type=list, default=[128, 128])
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--naive', type=bool, default=False)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--pi_lr', type=float, default=3e-4)

    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--gamma_coef',  type=float, default=1.0)
    parser.add_argument('--vf_lr', type=float, default=1e-3)
    args = parser.parse_args()
    return args

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.tim_buf = np.zeros(size, dtype=np.int32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, tim, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.tim_buf[self.ptr] = tim
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 0.001)
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, tim=self.tim_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

# write a train loop

# compute the initial state distribution
def est_initial(env,bins,dim=None):
    n = env.observation_space.shape[0]
    low = env.observation_space.low
    low = np.maximum(low, -1000 * np.ones(n))
    high = env.observation_space.high
    high = np.minimum(high, 1000 * np.ones(n))
    state_steps = (high - low) / bins
    if np.any(dim!=None):
        n = dim.shape[0]
        low = low[dim]
        high = high[dim]
        state_steps = state_steps[dim]

    counts = np.zeros((bins, )*n)
    for k in range(5000):
        o = env.reset()
        if np.any(dim != None):
            o = o[dim]
        idx = (o-low)/state_steps
        idx = idx.astype(int)
        counts[idx] += 1
    counts /= 5000
    counts = counts.flatten()
    return counts

# estimate the transition probability under any policy with continuous actions
"""
velocity_{t+1} = velocity_t + force * self.power - 0.0025 * cos(3 * position_t)

position_{t+1} = position_t + velocity_{t+1}
"""
def transition_prob(action_mean,action_std,env,bins):
    n = env.observation_space.shape[0]
    trans_prob = np.zeros((bins**n,bins**n))

    low = env.observation_space.low
    high = env.observation_space.high
    state_steps = (high - low) / bins
    next_mean_low = np.zeros(2)
    next_mean_high = np.zeros(2)

    for i in range(bins):
        for j in range(bins):
            state_low = low + np.array([i*state_steps[0],j*state_steps[1]])
            next_mean_low[1] = state_low[1] + action_mean[i*bins+j]*env.power-0.0025* np.cos(3*state_low[0])
            next_mean_low[0] = state_low[0] + next_mean_low[1]

            next_mean = next_mean_low
            next_std = env.power*action_std

            state_list = np.arange(low[0]+state_steps[0],high[0],state_steps[0])
            prob_1 = np.insert(norm.cdf(state_list,next_mean[0],next_std),
                               bins-1,1) - np.insert(norm.cdf(state_list,next_mean[0],next_std),0,0)
            state_list = np.arange(low[1] + state_steps[1], high[1], state_steps[1])
            prob_2 = np.insert(norm.cdf(state_list, next_mean[1], next_std),
                               bins - 1, 1) - np.insert(norm.cdf(state_list, next_mean[1], next_std), 0, 0)

            prob = np.matmul(np.expand_dims(prob_1,axis=1),np.expand_dims(prob_2,axis=0))
            trans_prob[i*bins+j,:] = prob.flatten()
    return trans_prob

def get_states(env,num_states=256):
    # write out the state_list
    n = env.observation_space.shape[0]
    low = env.observation_space.low
    high = env.observation_space.high
    bins = int(np.power(num_states, 1 / n))
    state_steps = (high - low) / bins
    x = np.arange(low[0], high[0], state_steps[0])
    y = np.arange(low[1], high[1], state_steps[1])
    states = np.stack((np.repeat(np.expand_dims(x, 1), bins, axis=1),
                       np.repeat(np.expand_dims(y, 0), bins, axis=0)), axis=2).reshape(-1, n)
    return bins,states
# compute the approximation error

# record the distribution difference and error at checkpoints

# record the averaged distribution difference for different environments

def compute_correction(env,agent,gamma,initial):
    # get the policy mean and std
    bins, states = get_states(env)
    policy_mean = agent.pi.mu_net(torch.as_tensor(states, dtype=torch.float32)).detach().numpy()
    policy_std = np.exp(agent.pi.log_std.detach().numpy())

    # get transition matrix P
    P = transition_prob(policy_mean,policy_std,env,bins)
    n = states.shape[0]
    # # check if the matrix is a transition matrix
    # print(np.sum(P,axis=1))
    power = 1
    err = np.matmul(np.ones(n),np.linalg.matrix_power(P,power+1))-\
          np.matmul(np.ones(n), np.linalg.matrix_power(P, power))
    err = np.sum(np.abs(err))
    while err > 1.2 and power<10:
        power+=1
        err = np.matmul(np.ones(n), np.linalg.matrix_power(P,  power + 1)) - \
              np.matmul(np.ones(n), np.linalg.matrix_power(P, power))
        err = np.sum(np.abs(err))
    # print(np.sum(np.linalg.matrix_power(P, 3),axis=1))
    d_pi = np.matmul(initial, np.linalg.matrix_power(P, power + 1))
    # print("stationary distribution",d_pi,np.sum(d_pi))

    if np.sum(d_pi - np.matmul(np.transpose(d_pi),P))>0.001:
        print("not the stationary distribution")

    # compute the special transition function M
    M = np.matmul(np.diag(d_pi) , P)
    M = np.matmul(M, np.diag(1/(d_pi+0.01)))

    # uniformly initial state distribution
    correction = np.matmul(np.linalg.inv(np.eye(n)-gamma * np.transpose(M)) , (1-gamma) * initial/(d_pi+0.01))
    discounted = correction * d_pi

    return correction,d_pi,discounted

def compute_c_D(env,data,gamma,bins,num_traj):
    n = env.observation_space.shape[0]
    low = env.observation_space.low
    high = env.observation_space.high
    state_steps = (high - low) / bins

    counts = np.zeros((bins, )*n)
    numerator = np.zeros((bins, )*n)
    tim = data['tim']
    for i in range(data['obs'].size(dim=0)):
        s = data['obs'][i].numpy()
        idx = (s - low) / state_steps
        idx = idx.astype(int)
        counts[idx] += 1
        numerator[idx] += gamma ** tim[i].item()
    numerator /= num_traj
    c_D = data['obs'].size(dim=0) * (1 - gamma) * numerator / (counts + 0.001)

    counts = counts.flatten()
    est = c_D.flatten()

    sampling = counts / data['obs'].size(dim=0)
    indices = np.argwhere(counts)
    return est, sampling, indices,counts

def est_sampling(env,data,bins,dim=None):
    n = env.observation_space.shape[0]
    low = env.observation_space.low
    low = np.maximum(low,-1000*np.ones(n))
    high = env.observation_space.high
    high = np.minimum(high, 1000 * np.ones(n))
    state_steps = (high - low) / bins
    if np.any(dim!=None):
        n = dim.shape[0]
        low = low[dim]
        high = high[dim]
        state_steps = state_steps[dim]

    counts = np.zeros((bins, )*n)
    for i in range(data['obs'].size(dim=0)):
        s = data['obs'][i].numpy()
        if np.any(dim != None):
            s=s[dim]
        idx = (s - low) / state_steps
        idx = idx.astype(int)
        counts[idx] += 1

    counts = counts.flatten()
    sampling = counts / data['obs'].size(dim=0)
    return sampling

def bias_compare(discounted,sampling,indices,counts, initial,d_pi,correction,est):
    ## this method counts the number of times of each state shown in one buffer.
    err_in_buffer = np.matmul(np.transpose(sampling), np.abs(correction - est))
    approx_bias = np.sum(
        np.abs(discounted[indices] * counts[indices] - est[indices] * sampling[indices] * counts[indices]))
    miss_bias = np.sum(np.abs(discounted[indices] * counts[indices] - sampling[indices] * counts[indices]))
    dist_diff = np.sum(np.abs(initial-sampling))
    return approx_bias/miss_bias,err_in_buffer,dist_diff

def weighted_ppo(env_fn, actor_critic=core.MLPWeightedActorCritic, ac_kwargs=dict(), seed=0,
                 steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
                 vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
                 target_kl=0.01, logger_kwargs=dict(), save_freq=10, scale=1.0, gamma_coef=1.0):
    """
    Proximal Policy Optimization (by clipping),

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
            module. The ``step`` method should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing
                                           | the log probability, according to
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical:
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`.

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # discretize the state space to estimate state distributions
    bins = 11
    # but we only study three dimension of states
    n = env.observation_space.shape[0]
    if n>3:
        a = np.arange(n)
        np.random.shuffle(a)
        dim = a[:3]
    else:
        dim = None
    initial = est_initial(env, bins,dim)
    print(np.sum(initial))

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, tim, adv, logp_old = data['obs'], data['act'], data['tim'], \
                                       data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        _, w = ac.v(obs)
        # weighting = torch.as_tensor(compute_c_D(data), dtype=torch.float32)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        # loss_pi = -(torch.min(ratio * adv, clip_adv) * weighting).mean()
        loss_pi = -(torch.min(ratio * adv, clip_adv) * (w / scale)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret, tim = data['obs'], data['ret'], data['tim']
        v, w = ac.v(obs)
        loss = ((v - ret) ** 2).mean() + gamma_coef * ((w - gamma ** tim * scale) ** 2).mean()
        return loss

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(num_traj):
        data = buf.get()

        # # compute errors
        # bins,_ = get_states(env)
        # correction, d_pi, discounted= compute_correction(env, ac, gamma,initial)
        # est, sampling, indices, counts = compute_c_D(env, data, gamma, bins, num_traj)
        # ratio,_, diff_dist = bias_compare(discounted, sampling, indices, counts, initial,d_pi, correction, est)

        # only compute distribution difference between the initial and the sampling
        sampling = est_sampling(env,data,bins,dim)
        print(np.sum(sampling))
        print(initial[:10],":::",sampling[:10])
        ratio = 0
        diff_dist = np.sum(np.abs(initial-sampling))/(initial.shape[0])

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)  # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)  # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))
        return ratio,diff_dist

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    rets = []
    avgrets = []
    ratios = []
    dist_diffs = []

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        num_traj = 0
        for t in range(local_steps_per_epoch):
            a, v, w, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, ep_len - 1, v, logp)
            logger.store(VVals=v)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                num_traj+=1
                if epoch_ended and not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    rets.append(ep_ret)
                o, ep_ret, ep_len = env.reset(), 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        ratio,diff_dist = update(num_traj)

        # Log info about epoch
        avgrets.append(np.mean(rets))
        rets = []
        ratios.append(ratio)
        dist_diffs.append(diff_dist)
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()
    return avgrets,ratios,dist_diffs

def tune_Reacher():
    args = argsparser()
    seeds = range(10)

    # Torch Shenanigans fix
    set_one_thread()

    logger.configure(args.log_dir, ['csv'], log_suffix='dist-error-' + str(args.env)+'-' + str(args.seed))

    returns = []
    for seed in seeds:
        hyperparam = random_search(args.seed)
        checkpoint = 4000
        result, ratio, dist_diff = weighted_ppo(lambda: gym.make(args.env), actor_critic=core.MLPWeightedActorCritic,
                              ac_kwargs=dict(hidden_sizes=args.hid, critic_hidden_sizes=hyperparam['critic_hid']),
                              epochs=args.epochs,
                              gamma=hyperparam['gamma'], target_kl=hyperparam['target_kl'], vf_lr=hyperparam['vf_lr'],
                              pi_lr=hyperparam["pi_lr"],
                              seed=seed, scale=hyperparam['scale'], gamma_coef=hyperparam['gamma_coef'])

        ret = np.array(result)
        print(ret.shape)
        returns.append(ret)
        name = list(hyperparam.values())
        name = [str(s) for s in name]
        name.append(str(seed))
        print("hyperparam", '-'.join(name))
        logger.logkv("hyperparam", '-'.join(name))
        for n in range(ret.shape[0]):
            logger.logkv(str((n + 1) * checkpoint), ret[n])
        logger.dumpkvs()
        logger.logkv("hyperparam", '-'.join(name) + '-ratio')
        for n in range(ret.shape[0]):
            logger.logkv(str((n + 1) * checkpoint), ratio[n])
        logger.dumpkvs()
        logger.logkv("hyperparam", '-'.join(name) + '-dist')
        for n in range(ret.shape[0]):
            logger.logkv(str((n + 1) * checkpoint), dist_diff[n])
        logger.dumpkvs()

def setsizes():
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['lines.markeredgewidth'] = 1.0
    plt.rcParams['lines.markersize'] = 3
    plt.rcParams['axes.labelsize'] = 19.0
    plt.rcParams['axes.titlesize'] = 19.0
    plt.rcParams['xtick.labelsize'] = 16.0
    plt.rcParams['ytick.labelsize'] = 16.0
    plt.rcParams['xtick.direction'] = "out"
    plt.rcParams['ytick.direction'] = "in"
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['ytick.minor.pad'] = 50.0


def setaxes():
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gcf().subplots_adjust(left=0.2)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # ax.spines['left'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', direction='out', which='minor', width=2, length=3,
                   labelsize=16, pad=8)
    ax.tick_params(axis='both', direction='out', which='major', width=2, length=8,
                   labelsize=16, pad=8)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    # for tick in ax.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(ax.getxticklabelsize())
    # for tick in ax.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(ax.getxticklabelsize())

def plot_result():
    for filename in os.listdir('./dist'):
        file = os.path.join('./dist', filename)
        if not file.endswith('.csv'):
            continue
        # checking if it is a file
        dummy = os.path.join('./dist_ready', filename)
        print(file)
        with open(file, 'r') as read_obj, open(dummy, 'w') as write_obj:
            # Iterate over the given list of strings and write them to dummy file as lines
            Lines = read_obj.readlines()
            Lines[0] = Lines[0].replace('\n', ',hyperparam2\n')
            for line in Lines:
                write_obj.write(line)

    plt.figure()
    for filename in os.listdir('./dist_ready'):
        file = os.path.join('./dist_ready', filename)
        if not file.endswith('.csv'):
            continue

        data = pd.read_csv(file,header=0, parse_dates={'timestamp': ['hyperparam','hyperparam2']},
                           index_col='timestamp')
        data.columns = data.columns.astype(int)
        data = data.sort_index(axis=1, ascending=True)
        name = data.index[0]
        name = name[:-2]
        returns = []
        ratios =[]
        dist_diffs = []
        for seed in range(10):
            rets = data.loc[name +'-' +str(seed)].to_numpy()
            returns.append(rets)
            ratios.append(data.loc[name+'-' +str(seed)+'-ratio'].to_numpy())
            dist_diffs.append(data.loc[name+'-' +str(seed)+'-dist'].to_numpy())
        mean = np.mean(dist_diffs, axis=0)
        plt.subplot(122)
        plt.plot(data.columns, mean, label=filename)

    plt.xlabel("steps")
    plt.ylabel("Distribution Difference")
    setaxes()
    # define y_axis, x_axis
    setsizes()
    # plt.xticks(fontsize=15, rotation=45)
    # plt.yticks(fontsize=17)
    # plt.legend(prop={"size": 16})
    plt.legend()
    plt.tight_layout()
    plt.show()

tune_Reacher()
# plot_result()