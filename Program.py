import multiprocessing as mp
import numpy as np
import scipy.stats as sc
import matplotlib.pyplot as pl
import time, math

# black scholes valuation of call option on non-dividend-paying equity
def VanillaCall(s, x, t, r, v):
    pv = 0.0
    # option value at maturity
    if((t - 0.0) < (1 / 365)):
        pv = max(s - x, 0.0)
    # option value before its maturity
    else:
        d1 = (math.log(s / x) + (r + 0.5 * v ** 2) * t) / (v * math.sqrt(t))
        d2 = (math.log(s / x) + (r - 0.5 * v ** 2) * t) / (v * math.sqrt(t))
        pv = (s * sc.norm.cdf(d1, 0.0, 1.0) - x * math.exp(-r * t) * sc.norm.cdf(d2, 0.0, 1.0))
    return pv

# globalize local parameters
def Globalizer(local_parameters):
    global global_parameters
    global_parameters = local_parameters

# simulate exposures by using
# - a given one-factor process (func_process)
# - a given pricing function (func_pricing)
def SimulateExposures(arg):
    # seed random number generator
    np.random.seed(arg)
    # unzip globalized tuple
    n_paths_per_process, n_steps, t, s_0, r, vol, func_pricing, func_process = global_parameters
    s = 0.0
    pv = 0.0
    dt = (t / n_steps)
    # create container for results
    paths = np.zeros(shape=(n_paths_per_process, n_steps + 1), dtype=float)
    for i in range(n_paths_per_process):
        s = s_0
        t_expiry = t
        # value product at path inception
        paths[i, 0] = func_pricing(s, t_expiry)
        # create array of normal random variates
        e = np.random.standard_normal(n_steps)
        for j in range(n_steps):
            s = func_process(s, e[j])
            t_expiry = (t - dt * (j + 1))
            # value product at after inception
            paths[i, j + 1] = func_pricing(s, t_expiry)
    return paths

t_0 = time.time()

# monte carlo parameters
n_paths = 10000
n_steps = 250
n_processes = 2 # 1 = 359 sec, 2 = 228 sec
n_paths_per_process = int(n_paths / n_processes)
seeds = np.random.randint(low=1, high=999999, size=n_processes)

# product- and process-related parameters
t = 1.0
s_0 = 100.0
r = 0.05
vol = 0.15
x = 100.0
drift = ((r - 0.5 * vol * vol) * (t / n_steps))
diffusion = (vol * math.sqrt(t / n_steps))

# method for calculating pv as a function of spot and time to maturity
func_pricing = lambda s_, t_: VanillaCall(s_, x, t_, r, vol)
# method for simulating stochastic process
func_process = lambda s_, e_: s_ * math.exp(drift + diffusion * e_)

# container for storing results from worker methods
results = []

# local parameters, which will be globalized for pool workers
simulation_parameters = (n_paths_per_process, n_steps, t, s_0, r, vol, func_pricing, func_process)
pool = mp.Pool(processes=n_processes, initializer=Globalizer, initargs=(simulation_parameters,))
for r in pool.imap_unordered(SimulateExposures, seeds): results.append(r)

t_1 = time.time()
print('processing time', t_1 - t_0)

# stack results from workers into one array
exposures = np.empty(shape = (0, results[0].shape[1]))
for i in range(n_processes):
    exposures = np.vstack([exposures, results[i]])

# average of all exposures
expected_exposure = np.mean(exposures, axis=0)
pl.plot(expected_exposure)
pl.show()
