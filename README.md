# Unbinned Inference with Correlated Events

Code repository for the paper [*Unbinned Inference with Correlated Events*](https://doi.org/10.1140/epjc/s10052-025-14835-1), published in *European Physical Journal C* (2025).

> **Krish Desai, Owen Long, Benjamin Nachman**
> Eur. Phys. J. C **85**, 1089 (2025) · [arXiv:2504.14072](https://arxiv.org/abs/2504.14072)

## Overview

This repository investigates **unbinned statistical inference** in the presence of correlations between truth-level quantities and their detector-level reconstructions. The analysis uses **Iterative Bayesian Unfolding (IBU)** combined with **bootstrapping** to:

- Unfold detector smearing effects from reconstructed distributions
- Fit Gaussian parameters (mean μ, variance σ²) to the unfolded distributions
- Compare inference quality using **full** vs **diagonal** covariance matrices
- Estimate confidence intervals via profile likelihood and asymptotic (MINOS) methods
- Study how detector resolution (smearing) affects parameter recovery

The study compares the unbinned approach against traditional binned methods over a range of smearing values.

## Repository Structure

```
UnbinnedCorrelation/
├── OneVarBootstrap.py            # Minimal example: data generation, binning, IBU, bootstrap
├── ConfidenceIntervalScript.py   # Confidence interval estimation via profile likelihood
├── BootstrapAndInfer.ipynb       # Interactive walkthrough of bootstrap + inference
├── BinnedGaussian-Copy1.ipynb    # Binned Gaussian analysis
├── MeanBinnedGaussian.ipynb      # Mean of binned Gaussian
├── OwenComparison.ipynb          # Comparison with binned methods
├── Bootstrapped.ipynb            # Bootstrap experiments
├── BinnedCorrs/                  # Binned correlation studies at varying resolutions
│   ├── GoodResolution.ipynb
│   ├── OwenRes.ipynb
│   └── ResRanges.ipynb
├── OwenComparison/               # Full analysis pipeline (data → fits → plots)
│   ├── GenerateInputSamples.py   # Generate synthetic bootstrap samples (parallelised)
│   ├── iMinuitFits.py            # Gaussian fits with iMinuit (full & diagonal cov.)
│   └── plotting.py               # Publication-quality comparison plots
├── CITATION.bib                  # BibTeX citation
└── CITATION.cff                  # Citation File Format
```

## Requirements

Python 3.8+ with the following packages:

```bash
pip install numpy scipy matplotlib scikit-learn iminuit pandas
```

## Usage

### Quick start

Run the minimal self-contained example (data generation → binning → IBU → bootstrap):

```bash
python OneVarBootstrap.py
```

### Confidence interval estimation

Estimate confidence intervals for a given detector smearing value (e.g. 0.3):

```bash
python ConfidenceIntervalScript.py 0.3
```

The script generates synthetic Gaussian data, bootstraps the IBU unfolding 500 times, fits a Gaussian using both full and diagonal covariance matrices, and reports 1σ confidence intervals obtained from a profile-likelihood scan.

### Full OwenComparison pipeline

This reproduces the main results of the paper across 20 smearing values.

```bash
cd OwenComparison

# 1. Generate bootstrap input samples (runs in parallel)
python GenerateInputSamples.py

# 2. Perform Gaussian fits with iMinuit
python iMinuitFits.py

# 3. Produce comparison plots
python plotting.py
```

Output plots (PDF) are written to the working directory:

- `mu_error_plot_with_errorbars_ratio.pdf`
- `var_error_plot_with_errorbars_ratio.pdf`
- `mu_mean_values_with_errorbars_ratio.pdf`
- `var_mean_values_with_errorbars_ratio.pdf`

### Jupyter notebooks

Open any notebook for an interactive walkthrough:

```bash
jupyter notebook BootstrapAndInfer.ipynb
```

## Methods

### Iterative Bayesian Unfolding (IBU)

Detector effects are inverted via iterative reweighting of a prior distribution using the response matrix **R** (the conditional probability of reconstructing a true-level event in a given reco bin):

```
f_{n+1} ∝ f_n · Rᵀ [data / (R · f_n)]
```

Five iterations are used throughout the analysis. Binning is chosen dynamically to enforce a minimum purity (~50%) per bin.

### Bootstrap error estimation

The full analysis chain (binning → response matrix → IBU → fit) is repeated 500 times on resampled data. The RMS of the resulting best-fit parameters provides a reliable estimate of the statistical uncertainty, which is compared against asymptotic MINOS errors from iMinuit.

### Covariance matrix treatment

Two fitting strategies are compared:

| Mode | Description |
|------|-------------|
| **Full covariance** | All bin-to-bin correlations included in the χ² |
| **Diagonal covariance** | Only per-bin variances used |

The ratio of asymptotic to RMS errors quantifies the impact of ignoring correlations.

## Default parameters

| Parameter | Value |
|-----------|-------|
| Truth distribution | Gaussian, μ = 0.2, σ² = 0.81 |
| Generator (MC) distribution | Gaussian, μ = 0.0, σ² = 1.0 |
| Data sample size | 10 000 |
| Simulation sample size | 100 000 |
| Number of bootstraps | 500 |
| IBU iterations | 5 |
| Smearing values scanned | 20 values in [0, 0.75] |
| RNG seed | 5048 |

## Citation

If you use this code, please cite:

```bibtex
@article{desai2025unbinned,
  author       = {Desai, Krish and Long, Owen and Nachman, Benjamin},
  title        = {Unbinned Inference with Correlated Events},
  journaltitle = {European Physical Journal C},
  volume       = {85},
  pages        = {1089},
  date         = {2025-10-01},
  doi          = {10.1140/epjc/s10052-025-14835-1},
  eprint       = {2504.14072},
  eprintclass  = {physics.data-an},
}
```
