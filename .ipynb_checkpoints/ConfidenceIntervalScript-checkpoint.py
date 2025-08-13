import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.optimize import minimize, brentq

from prettytable import PrettyTable

'PARAMETERS'
data_size = 10**4
sim_size = 10**5
n_iterations = 5
n_bootstraps = 500
epsilon = 1e-8
rcond = 1e-3

mu_true, var_true = 0.2, 0.81
mu_gen, var_gen = 0.0, 1.0

'FUNCTIONS'
normalize = lambda x: x / np.sum(x, axis=0)

def create_response_matrix(gen, sim, bins):
    H, _, _ = np.histogram2d(gen.ravel(), sim.ravel(), bins=[bins, bins])
    H = normalize(H)
    H[np.isnan(H)] = 0
    return H

def bayesian_unfolding_step(R, f, data_hist):
    reweight = data_hist / (R @ f + epsilon)
    reweight[np.isnan(reweight)] = 0
    reweight[(R @ f) == 0] = 0
    f_prime = f * (R @ reweight)
    return normalize(f_prime)

def iterative_bayesian_unfolding(data, gen, sim, bins, n_iterations):
    fs = np.empty((n_iterations, len(bins) - 1, ))
    R = create_response_matrix(gen, sim, bins)
    f, _ = np.histogram(gen, bins=bins)
    f = normalize(f)
    data_hist, _ = np.histogram(data, bins=bins)
    data_hist = normalize(data_hist)
    sim_hist, _ = np.histogram(sim, bins=bins)
    sim_hist = normalize(sim_hist)
    
    for i in range(n_iterations):
        f = bayesian_unfolding_step(R, f, data_hist)
        fs[i] = f
    return fs

def neg_log_likelihood(params, x, y, cov):
    mu, var = params
    cdf_values = norm.cdf(x, mu, np.sqrt(var))
    y_model = np.diff(cdf_values)
    return 0.5 * (y - y_model).T @ np.linalg.pinv(cov, rcond = rcond) @ (y - y_model)

def find_confidence_interval(nll, param_index, param_value, nll_min, x, y, cov, bounds):
    def target_function(test_value):
        test_params = param_value.copy()
        test_params[param_index] = test_value
        return nll(test_params, x, y, cov) - (nll_min + 0.5)
    lower_bound = brentq(target_function, bounds[param_index][0], param_value[param_index])
    upper_bound = brentq(target_function, param_value[param_index], bounds[param_index][1])
    return lower_bound, upper_bound
    
'DATASETS'
rng = np.random.default_rng(seed = 5048)
truth = rng.normal(mu_true, np.sqrt(var_true), (n_bootstraps, data_size))
gen = rng.normal(mu_gen, np.sqrt(var_gen), (n_bootstraps, sim_size))

def main(smearing):
    data = rng.normal(truth, smearing)
    sim = rng.normal(gen, smearing)

    'BINS'
    bins = [truth.min()]
    i = 0
    while bins[-1] < truth.max() and i < len(bins):
        for binhigh in np.linspace(bins[i] + epsilon, truth.max(), 20):
            in_bin = (truth[0] > bins[i]) & (truth[0] < binhigh)
            in_reco_bin = (data[0] > bins[i]) & (data[0] < binhigh)
            if np.sum(in_bin) > 0:
                purity = np.sum(in_bin & in_reco_bin) / np.sum(in_bin)
                if purity > (0.5):
                    i += 1
                    bins.append(binhigh)
                    break
        else:
            break
    bins.append(truth.max())

    bins = np.array(bins)
    bin_widths = np.diff(bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    n_bins = len(bins) - 1

    'IMPLEMENTATION'
    unfolded_results = np.empty((n_bootstraps, n_iterations, n_bins))
    for i in range(n_bootstraps):
        unfolded_results[i] = iterative_bayesian_unfolding(data[i], gen[i], sim[i], bins, n_iterations)
    unfolded_results_last = unfolded_results[:, -1]

    cov = np.cov(unfolded_results_last.T)
    cov_diag = np.diag(np.diag(cov))

    fitted_params = np.empty((n_bootstraps, 2))
    intervals = np.empty((n_bootstraps, 2, 2))
    intervals_diag = np.empty((n_bootstraps, 2, 2))
    fitted_params_diag = np.empty((n_bootstraps, 2))
    
    bounds = [(-1, 1), (epsilon, 2)]
    for i in range(n_bootstraps):
        initial_guess = np.random.normal(0.5, 0.1, 2)
        result = minimize(neg_log_likelihood,
                          initial_guess,
                          args=(bins, unfolded_results_last[i], cov),
                          bounds=bounds)
    
        nll_min = result.fun  # Minimum NLL value
        fitted_params[i] = result.x
        intervals[i, 0] = find_confidence_interval(neg_log_likelihood,
                                               0, result.x,
                                               nll_min,
                                               bins,
                                               unfolded_results_last[i],
                                               cov,
                                               bounds)
        intervals[i, 1] = find_confidence_interval(neg_log_likelihood,
                                                  1,
                                                  result.x,
                                                  nll_min,
                                                  bins,
                                                  unfolded_results_last[i],
                                                  cov,
                                                  bounds)
    
        result_diag = minimize(neg_log_likelihood,
                               initial_guess,
                               args=(bins, unfolded_results_last[i], cov_diag),
                               #method='L-BFGS-B',
                               bounds = bounds
                              )
        nll_min_diag = result_diag.fun  # Minimum NLL value
        fitted_params_diag[i] = result_diag.x
        intervals_diag[i, 0] = find_confidence_interval(neg_log_likelihood,
                                               0, result_diag.x,
                                               nll_min_diag,
                                               bins,
                                               unfolded_results_last[i],
                                               cov_diag,
                                               bounds)
        intervals_diag[i, 1] = find_confidence_interval(neg_log_likelihood,
                                                  1,
                                                  result_diag.x,
                                                  nll_min_diag,
                                                  bins,
                                                  unfolded_results_last[i],
                                                  cov_diag,
                                                  bounds)
    return {
        "full_cov": {
            "mu": np.mean(fitted_params[:, 0]),
            "sigma_on_mu": np.std(fitted_params[:, 0]),
            "asy_mu": 0.5*np.diff(np.mean(intervals, axis=0)[0])[0],
            "var": np.mean(fitted_params[:, 1]),
            "sigma_on_var": np.std(fitted_params[:, 1]),
            "asy_var": 0.5*np.diff(np.mean(intervals, axis=0)[1])[0]
        },
        "diag_cov": {
            "mu": np.mean(fitted_params_diag[:, 0]),
            "sigma_on_mu": np.std(fitted_params_diag[:, 0]),
            "asy_mu": 0.5*np.diff(np.mean(intervals_diag, axis=0)[0])[0],
            "var": np.mean(fitted_params_diag[:, 1]),
            "sigma_on_var": np.std(fitted_params_diag[:, 1]),
            "asy_var": 0.5*np.diff(np.mean(intervals_diag, axis=0)[1])[0]
            
        }
    }