import numpy as np
import os
import torch

def set_one_thread():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

def random_search(seed):
    rng = np.random.RandomState(seed=seed)

    gamma_coef = rng.randint(low=10, high=1000)/100.0
    scale = rng.randint(low=1, high=150)
    target_kl = rng.randint(low=0.005*1000, high=0.5*1000)/1000.0
    vf_lr = rng.randint(low=3, high=50)/10000.0
    gamma = rng.choice([0.9,0.95,0.97,0.99,0.995])
    hid = np.array([[64,64],[128,128],[256,256]])
    critic_hid = rng.choice(range(hid.shape[0]))
    critic_hid = hid[critic_hid]
    print(critic_hid)

    hyperparameters = {"gamma_coef":gamma_coef, "scale":scale, "target_kl":target_kl,
                       "vf_lr":vf_lr,"critic_hid":list(critic_hid),"gamma":gamma}

    return hyperparameters