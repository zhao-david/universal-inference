import argparse
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from datetime import datetime
import pickle


def main(d_list=[10,20,50,100], n=10, mu=0, sigma=1, alpha=0.1, theta_low=0, theta_high=30, theta_num=151):
    grid_theta_norm_sq = np.linspace(theta_low, theta_high, theta_num)
    power_dict = {}
    pbar = tqdm(total=len(d_list), desc='Power curve for fixed d, varying theta')
    for d in d_list:
        result = [np.mean([universal_inference_MVN(theta_norm_sq=t, d=d, n=n, mu=mu, sigma=sigma, alpha=alpha)
                           for i in range(1000)]) for t in grid_theta_norm_sq]
        power_dict[d] = pd.Series(result, index=grid_theta_norm_sq)
        pbar.update(1)
    date_str = datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
    with open('power_dict_' + date_str + '.pkl', 'wb') as handle:
        pickle.dump(power_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return

def universal_inference_MVN(theta_norm_sq, d=10, n=10, mu=0, sigma=1, alpha=0.1):
    mean = np.repeat(mu,d)
    cov = sigma**2*np.eye(d)
    x_obs = multivariate_normal.rvs(mean=mean, cov=cov, size=int(n))
    theta = np.repeat(np.sqrt(theta_norm_sq / d), d)
    mle_0 = x_obs[int(n/2):].mean(axis=0)
    mle_1 = x_obs[:int(n/2)].mean(axis=0)
    num_0 = multivariate_normal.pdf(x=x_obs[:int(n/2)], mean=mle_0, cov=cov)
    num_1 = multivariate_normal.pdf(x=x_obs[int(n/2):], mean=mle_1, cov=cov)
    den_0 = multivariate_normal.pdf(x=x_obs[:int(n/2)], mean=theta, cov=cov)
    den_1 = multivariate_normal.pdf(x=x_obs[int(n/2):], mean=theta, cov=cov)
    T_0 = np.sum(np.log(num_0)) - np.sum(np.log(den_0))
    T_1 = np.sum(np.log(num_1)) - np.sum(np.log(den_1))
    S = np.log((np.exp(T_0) + np.exp(T_1)) / 2)
    reject = 0
    if S > np.log(1.0/alpha):
        reject = 1
    return reject



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', action="store", type=float, default=10,
                        help='number of observations')
    parser.add_argument('--alpha', action="store", type=float, default=0.1,
                        help='Statistical confidence level')
    argument_parsed = parser.parse_args()
    
    main(
        n=argument_parsed.n,
        alpha=argument_parsed.alpha
    )