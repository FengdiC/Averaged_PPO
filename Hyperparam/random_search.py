import numpy as np

def random_search(seed):
    rng = np.random.RandomState(seed=seed)

    gamma_coef = rng.uniform(low=0.1, high=10)
    scale = rng.uniform(low=1, high=150)
    target_kl = rng.uniform(low=0.005, high=0.5)
    vf_lr = rng.uniform(low=3e-4, high=0.01)
    pi_lr = rng.uniform(low=1e-4, high=1e-3)
    steps_per_epoch = rng.choice([2500,4000,5500])
    critic_hid = rng.choice(["[32,32]","[64,64]","[64,32]","[128,128]"])

    hyperparameters = {"gamma_coef":gamma_coef, "scale":scale, "target_kl":target_kl,
                       "vf_lr":vf_lr, "pi_lr":pi_lr, "steps_per_epoch":steps_per_epoch,
                       "critic_hid":critic_hid}