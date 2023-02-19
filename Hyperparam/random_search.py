import numpy as np
import os
import torch

def set_one_thread():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

def random_search(seed):
    rng = np.random.RandomState(seed=seed)

    # choice 1
    pi_lr = rng.choice([3,9,20,30,40,50])/10000.0
    # # choice 2
    # pi_lr = rng.choice([3, 9, 30]) / 10000.0
    gamma_coef = rng.randint(low=50, high=500)/100.0
    scale = rng.randint(low=1, high=120)
    target_kl = rng.randint(low=0.01*100, high=0.3*100)/100.0
    vf_lr = rng.randint(low=3, high=50)/10000.0
    gamma = rng.choice([0.95,0.97,0.99,0.995])
    hid = np.array([[64,64],[128,128],[256,256]])
    critic_hid = rng.choice(range(hid.shape[0]))
    critic_hid = hid[critic_hid]

    hyperparameters = {"pi_lr":pi_lr,"gamma_coef":gamma_coef, "scale":scale, "target_kl":target_kl,
                       "vf_lr":vf_lr,"critic_hid":list(critic_hid),"gamma":gamma}

    return hyperparameters