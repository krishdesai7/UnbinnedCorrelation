import numpy as np
from iminuit import Minuit

import os
import csv
import concurrent.futures
from scipy.special import erf

from GenerateInputSamples import smearings, fluctuate_data_size

n_iterations = 5
n_bootstraps = 500
epsilon = 1e-8
rcond = 1e-15  # Define rcond for pseudo-inverse

# FUNCTIONS
normalize = lambda x: x / np.sum(x, axis=0)

def create_response_matrix(gen, sim, bins):
    H, _, _ = np.histogram2d(gen.ravel(), sim.ravel(), bins=[bins, bins])
    H = normalize(H)
    H[np.isnan(H)] = 0
    return H

def bayesian_unfolding_step(R, f, data_hist):
    reweight = np.divide(data_hist, R @ f, out=np.zeros_like(data_hist, dtype=np.float64), where=(R @ f) != 0)
    reweight[np.isnan(reweight)] = 0
    f_prime = f * (R @ reweight)
    return f_prime

def iterative_bayesian_unfolding(data, gen, sim, bins, n_iterations):
    R = create_response_matrix(gen, sim, bins)
    f, _ = np.histogram(gen, bins=bins)
    data_hist, _ = np.histogram(data, bins=bins)
    
    for i in range(n_iterations):
        f = bayesian_unfolding_step(R, f, data_hist)
    return f

def gaussian_integral(a, b, A, mu, var):
    sigma = np.sqrt(var)
    if sigma == 0:
        return np.zeros_like(a)
    z_a = (a - mu) / (np.sqrt(2) * sigma)
    z_b = (b - mu) / (np.sqrt(2) * sigma)
    integral = 0.5 * (erf(z_b) - erf(z_a))
    return A * integral

def compute_fits(smearing):
    try:
        output_dir = f'input-samples/input-samples-smearing-{smearing:.4f}-v1{'b' if fluctuate_data_size else 'a'}'
        filename = f'fit_results/fit_results_{smearing:.4f}-v1{'b' if fluctuate_data_size else 'a'}.csv'
        if os.path.exists(filename):
            print(f"Directory '{output_dir}' exists. Skipping.")
            return
        # Load binning information
        binning_file = os.path.join(output_dir, 'binning.npy')
        with open(binning_file, 'rb') as f:
            bins = np.load(f)
            bin_widths = np.load(f)
            bin_centers = np.load(f)
            n_bins = int(np.load(f))

        def chi2(A, mu, var):
            y_model = np.array([
                gaussian_integral(a, b, A, mu, var)
                for a, b in zip(bins[:-1], bins[1:])
            ])
            diff = hist_counts - y_model
            chi2_value = diff.T @ cov_inv @ diff
            return chi2_value
        
        def chi2_diag(A, mu, var):
            y_model = np.array([
                gaussian_integral(a, b, A, mu, var)
                for a, b in zip(bins[:-1], bins[1:])
            ])
            diff = hist_counts - y_model
            chi2_value = diff.T @ cov_inv_diag @ diff
            return chi2_value

        unfolded_results = np.empty((n_bootstraps, n_bins))

        for i in range(n_bootstraps):
            file_path = os.path.join(output_dir, f'sample-{i:04d}.npz')
            with np.load(file_path) as data_file:
                data_i = data_file['data']
                gen_i = data_file['gen']
                sim_i = data_file['sim']    
            unfolded_results[i] = iterative_bayesian_unfolding(data_i, gen_i, sim_i, bins, n_iterations)

        cov = np.cov(unfolded_results.T)
        cov_diag = np.diag(np.diag(cov))
        cov_inv = np.linalg.pinv(cov, rcond=rcond)
        cov_inv_diag = np.linalg.pinv(cov_diag, rcond=rcond)
        
        # Number of parameters: amplitude, mean, var
        n_params = 3
        fitted_params_cov = np.empty((n_bootstraps, n_params))
        fitted_errors_cov = np.empty((n_bootstraps, n_params, 2))
        fitted_params_cov_diag = np.empty((n_bootstraps, n_params))
        fitted_errors_cov_diag = np.empty((n_bootstraps, n_params, 2))
        
        bin_mids = 0.5 * (bins[:-1] + bins[1:])

        results = [['Parameter', 'Value', 'Error_Lower', 'Error_Upper', 'Fit_Type', 'Bootstrap_Index']]
        
        for i in range(n_bootstraps):
            hist_counts = unfolded_results[i, :]
            
            initial_A = np.max(hist_counts)
            initial_mu = np.average(bin_mids, weights=hist_counts)
            initial_var = np.average((bin_mids - initial_mu)**2, weights=hist_counts)
            
            # Fit using cov
            m = Minuit(chi2, A=initial_A, mu=initial_mu, var=initial_var)
            m.limits["var"] = (0, None)  # Variance must be positive
            m.migrad()
            
            try:
                m.minos()
                merrors = m.merrors
                A_err = (-merrors["A"].lower, merrors["A"].upper)
                mu_err = (-merrors["mu"].lower, merrors["mu"].upper)
                var_err = (-merrors["var"].lower, merrors["var"].upper)
            except RuntimeError:
                A_err = (np.NaN, np.NaN)
                mu_err = (np.NaN, np.NaN)
                var_err = (np.NaN, np.NaN)
            
            # Store the fitted parameters and errors
            fitted_params_cov[i, :] = [m.values["A"], m.values["mu"], m.values["var"]]
            fitted_errors_cov[i, :] = [A_err, mu_err, var_err]
            
            # Fit using cov_diag
            m_diag = Minuit(chi2_diag, A=initial_A, mu=initial_mu, var=initial_var)
            m_diag.limits["var"] = (0, None)
            m_diag.migrad()
            
            try:
                m_diag.minos()
                merrors_diag = m_diag.merrors
                A_err_diag = (-merrors_diag["A"].lower, merrors_diag["A"].upper)
                mu_err_diag = (-merrors_diag["mu"].lower, merrors_diag["mu"].upper)
                var_err_diag = (-merrors_diag["var"].lower, merrors_diag["var"].upper)
            except RuntimeError:
                A_err_diag = (np.NaN, np.NaN)
                mu_err_diag = (np.NaN, np.NaN)
                var_err_diag = (np.NaN, np.NaN)
            
            # Store the fitted parameters and errors for cov_diag
            fitted_params_cov_diag[i, :] = [m_diag.values["A"], m_diag.values["mu"], m_diag.values["var"]]
            fitted_errors_cov_diag[i, :] = [A_err_diag, mu_err_diag, var_err_diag]

            # Results from the fit using cov
            results.append(['A', fitted_params_cov[i, 0], fitted_errors_cov[i, 0, 0], fitted_errors_cov[i, 0, 1], 'cov', i])
            results.append(['mu', fitted_params_cov[i, 1], fitted_errors_cov[i, 1, 0], fitted_errors_cov[i, 1, 1], 'cov', i])
            results.append(['var', fitted_params_cov[i, 2], fitted_errors_cov[i, 2, 0], fitted_errors_cov[i, 2, 1], 'cov', i])
            
            # Results from the fit using cov_diag
            results.append(['A', fitted_params_cov_diag[i, 0], fitted_errors_cov_diag[i, 0, 0], fitted_errors_cov_diag[i, 0, 1], 'cov_diag', i])
            results.append(['mu', fitted_params_cov_diag[i, 1], fitted_errors_cov_diag[i, 1, 0], fitted_errors_cov_diag[i, 1, 1], 'cov_diag', i])
            results.append(['var', fitted_params_cov_diag[i, 2], fitted_errors_cov_diag[i, 2, 0], fitted_errors_cov_diag[i, 2, 1], 'cov_diag', i])
        
        # Write the results to a CSV file
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for row in results:
                csvwriter.writerow(row)
        
        print(f"Combined fitted parameters and errors have been saved to {filename}")

    except Exception as e:
        print(f"An error occurred for smearing {smearing}: {e}")

def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(compute_fits, smearings)

if __name__ == '__main__':
    main()