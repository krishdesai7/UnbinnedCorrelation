import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chisquare, norm
from sklearn.utils import resample

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
from IPython.display import display, Latex
rc('font', size=20)

epsilon = 1e-10

def generate_data(mu_true, mu_gen, smearing, N):
    rng = np.random.default_rng()
    sig_true = sig_gen = 1
    truth = rng.normal(mu_true, sig_true, N)
    data = rng.normal(truth, smearing)
    gen = rng.normal(mu_gen, sig_gen, N)
    sim = rng.normal(gen, smearing)
    data_streams = np.array([truth, data, gen, sim])
    return data_streams

def create_bins(truth, data, purity_threshold):
    bins = [truth.min()]
    i = 0
    while bins[-1] < truth.max() and i < len(bins):
        for binhigh in np.linspace(bins[i] + 0.1, truth.max(), 200):
            in_bin = (truth > bins[i]) & (truth < binhigh)
            in_reco_bin = (data > bins[i]) & (data < binhigh)
            if np.sum(in_bin) > 0:
                purity = np.sum(in_bin & in_reco_bin) / np.sum(in_bin)
                if purity > purity_threshold:
                    print(f"{binhigh = }, {purity = }")
                    i += 1
                    bins.append(binhigh)
                    break
        else:
            break
    bins = np.array(bins)
    n_bins = len(bins) - 1
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    bin_widths = np.diff(bins)
    return bins, n_bins, bin_centers, bin_widths

def bootstrap_data(N, n_bootstraps, data_streams):
    indices = np.random.choice(np.arange(N), size=(n_bootstraps, N), replace=True)
    bootstrapped_data = np.take(data_streams, indices, axis=1)
    return bootstrapped_data

def create_response_matrices(bootstrapped_data, bins, n_bootstraps, n_bins):
    digitized_data = np.digitize(bootstrapped_data, bins) - 1
    response_matrices = np.empty((n_bootstraps, 2, n_bins, n_bins))

    for i in range(n_bootstraps):
        H1, _, _ = np.histogram2d(digitized_data[0, i, :], digitized_data[1, i, :], bins=[range(n_bins+1), range(n_bins+1)])
        H2, _, _ = np.histogram2d(digitized_data[2, i, :], digitized_data[3, i, :], bins=[range(n_bins+1), range(n_bins+1)])    
        response_matrices[i, 0, :, :] = H1
        response_matrices[i, 1, :, :] = H2
    normalized_matrices = response_matrices / (response_matrices.sum(axis=3, keepdims=True) + epsilon)
    return digitized_data, response_matrices, normalized_matrices

def compute_marginals(response_matrices, n_bootstraps, n_bins):
    marginals = np.empty((4, n_bootstraps, n_bins))
    for i in range(4):
        marginals[i, :, :] = np.sum(response_matrices[:, i // 2, :, :], axis=(2 - i % 2))
    return marginals

def compute_IBU(prior, data_marginal, alt_response_matrix, n_iterations):
    posterior = [prior]
    for i in range(n_iterations):
        m = alt_response_matrix * posterior[-1]
        m /= (m.sum(axis=1)[:,np.newaxis] + epsilon)
        posterior.append(m.T @ data_marginal)
    return posterior[-1]

def generate_ibu_results(n_bootstraps, n_bins, response_matrices, marginals, ibu_iterations, bin_widths):
    ibu_results = np.empty((n_bootstraps, n_bins), dtype=np.float64)
    for i in range(n_bootstraps):
        ibu_results[i] = compute_IBU(marginals[2, i, :], marginals[1, i, :],
                         response_matrices[i, 1, :, :].T, ibu_iterations)

    ibu_normalized = ibu_results / (ibu_results @ bin_widths)[:, np.newaxis]
    return ibu_results, ibu_normalized

def main():
    mu_true = 0
    mu_gen = 0.5
    smearing = 0.2
    N = int(1e5)
    n_boostraps = 10**4
    ibu_iterations = 10

    data_streams = generate_data(mu_true, mu_gen, smearing, N)
    bins, n_bins, bin_centers, bin_widths = create_bins(data_streams[0], data_streams[1])
    bootstrapped_data = bootstrap_data(N, n_bootstraps, data_streams)
    response_matrices, normalized_matrices = create_response_matrices(bootstrapped_data, n_bootstraps, n_bins)
    marginals = compute_marginals(normalized_matrices, n_bootstraps, n_bins)
    ibu_results, ibu_normalized = generate_ibu_results(n_bootstraps, n_bins, response_matrices, marginals, ibu_iterations)
    
    if __name__ == "__main__":
        main()
    
    