#!/usr/bin/env python3
"""
Quantitative Benchmarking: Fuzzy fBm with Poisson Jumps vs Geometric Brownian Motion

This script validates the proposed fuzzy fractional Brownian motion model with
Poisson jumps against classical Geometric Brownian Motion (GBM) using empirical
VIX data and established performance evaluation indicators.

Outputs:
  - Comparison table (printed + CSV)
  - Figure image35.png: Return distribution overlay
  - Figure image36.png: QQ-plots
  - Figure image37.png: Autocorrelation of squared returns
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
VIX_PATH = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/VIX DATA SET/VIX_daily.csv"
MEDIA_DIR = SCRIPT_DIR / "media" / "media"

# ============================================================
# SIMULATION PARAMETERS (matching the thesis Chapter 5)
# ============================================================
T = 1            # 1 year
N = 252          # trading days
H = 0.15         # Hurst exponent (rough volatility regime)
RISK_FREE = 0.05 # risk-free rate
LAMBDA_ = 0.5    # Poisson jump intensity
JUMP_MU = 0.0    # mean jump size
JUMP_SIGMA = 0.02  # jump size std
M_PATHS = 10000  # Monte Carlo paths


# ============================================================
# 1. LOAD AND PREPROCESS VIX DATA
# ============================================================
def load_vix_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date').reset_index(drop=True)
    df['Last Price'] = pd.to_numeric(df['Last Price'], errors='coerce')
    df = df.dropna(subset=['Last Price'])
    prices = df['Last Price'].values
    log_returns = np.diff(np.log(prices))
    return prices, log_returns


# ============================================================
# 2. STATISTICAL HELPER FUNCTIONS
# ============================================================
def hurst_exponent_rs(ts, min_lag=10, max_lag=None):
    """Estimate Hurst exponent via rescaled range (R/S) analysis."""
    n = len(ts)
    if max_lag is None:
        max_lag = n // 4
    max_lag = min(max_lag, n // 2)
    if max_lag <= min_lag:
        return np.nan
    lags = np.arange(min_lag, max_lag + 1)
    rs_values = []
    for lag in lags:
        n_sub = n // lag
        if n_sub < 1:
            rs_values.append(np.nan)
            continue
        rs_list = []
        for k in range(n_sub):
            sub = ts[k * lag:(k + 1) * lag]
            m = np.mean(sub)
            cumdev = np.cumsum(sub - m)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(sub, ddof=1)
            if S > 1e-12:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append(np.mean(rs_list))
        else:
            rs_values.append(np.nan)
    valid = [(l, r) for l, r in zip(lags, rs_values) if np.isfinite(r) and r > 0]
    if len(valid) < 5:
        return np.nan
    log_l = np.log([v[0] for v in valid])
    log_r = np.log([v[1] for v in valid])
    slope, _ = np.polyfit(log_l, log_r, 1)
    return slope


def acf_at_lag(series, lag):
    """Compute autocorrelation at a specific lag."""
    n = len(series)
    if lag >= n:
        return np.nan
    mu = np.mean(series)
    var = np.var(series)
    if var < 1e-15:
        return 0.0
    c = np.mean((series[:n - lag] - mu) * (series[lag:] - mu))
    return c / var


def compute_statistics(log_returns):
    """Compute the full set of distributional and temporal statistics."""
    sq = log_returns ** 2
    return {
        'mean': np.mean(log_returns),
        'std': np.std(log_returns, ddof=1),
        'skewness': float(stats.skew(log_returns)),
        'excess_kurtosis': float(stats.kurtosis(log_returns)),
        'hurst': hurst_exponent_rs(log_returns),
        'acf_sq_1': acf_at_lag(sq, 1),
        'acf_sq_5': acf_at_lag(sq, 5),
        'acf_sq_21': acf_at_lag(sq, 21),
    }


# ============================================================
# 3. FUZZY LOGIC (self-contained, no scikit-fuzzy dependency)
# ============================================================
def trimf(x, params):
    """Triangular membership function."""
    a, b, c = params
    y = np.zeros_like(x, dtype=float)
    if b != a:
        mask1 = (x >= a) & (x <= b)
        y[mask1] = (x[mask1] - a) / (b - a)
    if c != b:
        mask2 = (x > b) & (x <= c)
        y[mask2] = (c - x[mask2]) / (c - b)
    y[x == b] = 1.0
    return y


def fuzzy_volatility(sentiment_value):
    """Compute fuzzy-adjusted volatility from a market sentiment score [0,10]."""
    sent = np.linspace(0, 10, 500)
    vol = np.linspace(0.1, 0.5, 500)
    pess = trimf(sent, [0, 0, 5])
    neut = trimf(sent, [0, 5, 10])
    opti = trimf(sent, [5, 10, 10])

    low_v = trimf(vol, [0.1, 0.1, 0.3])
    med_v = trimf(vol, [0.1, 0.3, 0.5])
    high_v = trimf(vol, [0.3, 0.5, 0.5])

    pess_level = np.interp(sentiment_value, sent, pess)
    neut_level = np.interp(sentiment_value, sent, neut)
    opti_level = np.interp(sentiment_value, sent, opti)

    agg = np.fmax(np.fmax(
        np.fmin(pess_level, high_v),
        np.fmin(neut_level, med_v)),
        np.fmin(opti_level, low_v))

    if np.sum(agg) < 1e-12:
        return 0.3
    return np.sum(vol * agg) / np.sum(agg)


# ============================================================
# 4. SIMULATION: GEOMETRIC BROWNIAN MOTION
# ============================================================
def simulate_gbm_returns(mu_daily, sigma_daily, n_steps, n_paths):
    """Simulate daily log returns under GBM."""
    dt = 1.0
    returns = mu_daily + sigma_daily * np.random.randn(n_steps, n_paths)
    return returns


# ============================================================
# 5. SIMULATION: PROPOSED MODEL (fuzzy fBm + Poisson jumps)
# ============================================================
def build_cholesky_factor(H, T, N):
    """Pre-compute Cholesky factor for fBm covariance matrix."""
    t = np.linspace(T / N, T, N)
    gamma = 0.5 * (t[:, None] ** (2 * H) + t[None, :] ** (2 * H)
                    - np.abs(t[:, None] - t[None, :]) ** (2 * H))
    gamma += np.eye(N) * 1e-8
    L = np.linalg.cholesky(gamma)
    return L


def simulate_proposed_returns(L, sigma_scale, H, T, N, n_paths,
                              lam, jump_mu, jump_sigma, r):
    """
    Simulate daily log returns under the fuzzy fBm + Poisson jump model.
    sigma_scale is applied to the fBm component.
    """
    dt = T / N
    Z = np.random.randn(N, n_paths)
    fBm_paths = L @ Z  # (N, n_paths) — fBm values at t_1,...,t_N

    fBm_inc = np.diff(np.vstack([np.zeros((1, n_paths)), fBm_paths]), axis=0)

    jump_inc = np.zeros((N, n_paths))
    for m in range(n_paths):
        n_jumps = np.random.poisson(lam * T)
        if n_jumps > 0:
            jtimes = np.random.uniform(0, T, n_jumps)
            jsizes = np.random.normal(jump_mu, jump_sigma, n_jumps)
            jindices = np.clip(np.floor(jtimes / dt).astype(int), 0, N - 1)
            for ji, js in zip(jindices, jsizes):
                jump_inc[ji, m] += js

    drift = (r - 0.5 * sigma_scale ** 2) * dt
    returns = drift + sigma_scale * fBm_inc + jump_inc
    return returns


# ============================================================
# 6. MAIN EXPERIMENT
# ============================================================
def main():
    print("=" * 70)
    print("QUANTITATIVE BENCHMARKING EXPERIMENT")
    print("Fuzzy fBm + Poisson Jumps  vs  Geometric Brownian Motion")
    print("=" * 70)

    # --- Load empirical data ---
    print("\n[1/6] Loading VIX data ...")
    prices, emp_returns = load_vix_data(VIX_PATH)
    print(f"  VIX series: {len(prices)} prices, {len(emp_returns)} daily log returns")

    emp_stats = compute_statistics(emp_returns)
    print(f"  Empirical daily std: {emp_stats['std']:.6f}")
    print(f"  Empirical Hurst (R/S): {emp_stats['hurst']:.4f}")

    mu_daily = np.mean(emp_returns)
    sigma_daily = np.std(emp_returns, ddof=1)

    # --- Calibrate proposed model volatility parameter ---
    print("\n[2/6] Calibrating proposed model volatility ...")
    L = build_cholesky_factor(H, T, N)
    dt = T / N

    # Fuzzy logic: evaluate neutral market sentiment (score = 5 on [0,10])
    SENTIMENT_VALUE = 5.0
    sigma_fuzzy = fuzzy_volatility(SENTIMENT_VALUE)
    print(f"  Fuzzy volatility at neutral sentiment ({SENTIMENT_VALUE}): {sigma_fuzzy:.4f}")

    # Variance budget: empirical_var = fBm_var + jump_var
    #   fBm_var  = (sigma_scale * sigma_fuzzy)^2 * dt^(2H)
    #   jump_var = lambda * (mu_J^2 + sigma_J^2) * dt
    emp_var_daily = sigma_daily ** 2
    jump_var_daily = LAMBDA_ * (JUMP_MU ** 2 + JUMP_SIGMA ** 2) * dt
    target_fbm_var = max(0.0, emp_var_daily - jump_var_daily)

    dt_2H = dt ** (2 * H)
    sigma_scale = np.sqrt(target_fbm_var / (sigma_fuzzy ** 2 * dt_2H))
    sigma_effective = sigma_scale * sigma_fuzzy

    print(f"  Jump variance contribution (daily): {jump_var_daily:.8f}")
    print(f"  Target fBm variance (daily):        {target_fbm_var:.8f}")
    print(f"  dt^(2H) = {dt_2H:.6f}")
    print(f"  Calibrated sigma_scale = {sigma_scale:.6f}")
    print(f"  Effective volatility (scale * fuzzy) = {sigma_effective:.6f}")

    # --- Run GBM Monte Carlo ---
    print(f"\n[3/6] Simulating {M_PATHS} GBM paths ...")
    gbm_returns = simulate_gbm_returns(mu_daily, sigma_daily, N - 1, M_PATHS)

    # --- Run proposed model Monte Carlo ---
    print(f"\n[4/6] Simulating {M_PATHS} proposed model paths ...")
    prop_returns = simulate_proposed_returns(
        L, sigma_effective, H, T, N, M_PATHS,
        LAMBDA_, JUMP_MU, JUMP_SIGMA, RISK_FREE
    )
    prop_returns = prop_returns[1:, :]  # skip the t=0 -> t=1 increment to get N-1 returns

    # --- Compute per-path statistics, then average ---
    print("\n[5/6] Computing statistics ...")
    def stats_across_paths(all_returns, label):
        n_steps, n_p = all_returns.shape
        path_stats = {k: [] for k in ['mean', 'std', 'skewness', 'excess_kurtosis',
                                       'hurst', 'acf_sq_1', 'acf_sq_5', 'acf_sq_21']}
        for m in range(n_p):
            s = compute_statistics(all_returns[:, m])
            for k in path_stats:
                path_stats[k].append(s[k])
        result = {}
        for k in path_stats:
            vals = [v for v in path_stats[k] if np.isfinite(v)]
            result[k] = np.mean(vals) if vals else np.nan
            result[k + '_se'] = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan
        return result

    gbm_stats = stats_across_paths(gbm_returns, "GBM")
    prop_stats = stats_across_paths(prop_returns, "Proposed")

    # Kolmogorov-Smirnov tests (pooled returns vs empirical)
    gbm_pooled = gbm_returns.ravel()
    prop_pooled = prop_returns.ravel()

    ks_gbm, pval_gbm = stats.ks_2samp(emp_returns, gbm_pooled)
    ks_prop, pval_prop = stats.ks_2samp(emp_returns, prop_pooled)

    # Jarque-Bera on pooled returns
    jb_gbm, jb_p_gbm = stats.jarque_bera(gbm_pooled)
    jb_prop, jb_p_prop = stats.jarque_bera(prop_pooled)
    jb_emp, jb_p_emp = stats.jarque_bera(emp_returns)

    # --- Build comparison table ---
    print("\n[6/6] Building comparison table and figures ...")
    metrics = [
        ('Mean daily return', 'mean'),
        ('Std of daily returns', 'std'),
        ('Skewness', 'skewness'),
        ('Excess kurtosis', 'excess_kurtosis'),
        ('Hurst exponent (R/S)', 'hurst'),
        ('ACF(r², lag 1)', 'acf_sq_1'),
        ('ACF(r², lag 5)', 'acf_sq_5'),
        ('ACF(r², lag 21)', 'acf_sq_21'),
    ]

    print("\n" + "=" * 80)
    print(f"{'Statistic':<28} {'Empirical':>12} {'GBM':>12} {'Proposed':>12}")
    print("-" * 80)
    table_rows = []
    for label, key in metrics:
        e = emp_stats[key]
        g = gbm_stats[key]
        p = prop_stats[key]
        print(f"{label:<28} {e:>12.4f} {g:>12.4f} {p:>12.4f}")
        table_rows.append({'Statistic': label, 'Empirical': e, 'GBM': g, 'Proposed': p})

    print("-" * 80)
    print(f"{'KS statistic vs empirical':<28} {'—':>12} {ks_gbm:>12.4f} {ks_prop:>12.4f}")
    print(f"{'KS p-value':<28} {'—':>12} {pval_gbm:>12.4e} {pval_prop:>12.4e}")
    print(f"{'Jarque-Bera statistic':<28} {jb_emp:>12.1f} {jb_gbm:>12.1f} {jb_prop:>12.1f}")
    print(f"{'Jarque-Bera p-value':<28} {jb_p_emp:>12.4e} {jb_p_gbm:>12.4e} {jb_p_prop:>12.4e}")
    print("=" * 80)

    table_rows.append({'Statistic': 'KS statistic vs empirical', 'Empirical': '—',
                       'GBM': ks_gbm, 'Proposed': ks_prop})
    table_rows.append({'Statistic': 'KS p-value', 'Empirical': '—',
                       'GBM': pval_gbm, 'Proposed': pval_prop})
    table_rows.append({'Statistic': 'Jarque-Bera statistic', 'Empirical': jb_emp,
                       'GBM': jb_gbm, 'Proposed': jb_prop})
    table_rows.append({'Statistic': 'Jarque-Bera p-value', 'Empirical': jb_p_emp,
                       'GBM': jb_p_gbm, 'Proposed': jb_p_prop})

    df_table = pd.DataFrame(table_rows)
    df_table.to_csv(SCRIPT_DIR / "benchmark_results.csv", index=False)
    print(f"\nTable saved to {SCRIPT_DIR / 'benchmark_results.csv'}")

    # Absolute-error summary (RMSE of moment vector)
    moment_keys = ['mean', 'std', 'skewness', 'excess_kurtosis', 'hurst',
                   'acf_sq_1', 'acf_sq_5', 'acf_sq_21']
    emp_vec = np.array([emp_stats[k] for k in moment_keys])
    gbm_vec = np.array([gbm_stats[k] for k in moment_keys])
    prop_vec = np.array([prop_stats[k] for k in moment_keys])
    mae_gbm = np.mean(np.abs(gbm_vec - emp_vec))
    mae_prop = np.mean(np.abs(prop_vec - emp_vec))
    rmse_gbm = np.sqrt(np.mean((gbm_vec - emp_vec) ** 2))
    rmse_prop = np.sqrt(np.mean((prop_vec - emp_vec) ** 2))
    print(f"\nAggregate MAE  of statistics:  GBM = {mae_gbm:.4f},  Proposed = {mae_prop:.4f}")
    print(f"Aggregate RMSE of statistics:  GBM = {rmse_gbm:.4f},  Proposed = {rmse_prop:.4f}")

    # ============================================================
    # FIGURES
    # ============================================================
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

    # --- Figure 5.2: Return distribution overlay ---
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(-0.25, 0.25, 120)
    ax.hist(emp_returns, bins=bins, density=True, alpha=0.45, color='#2c3e50',
            label='Empirical VIX', edgecolor='none')
    from scipy.stats import gaussian_kde
    x_kde = np.linspace(-0.25, 0.25, 500)
    kde_gbm = gaussian_kde(gbm_pooled, bw_method=0.05)
    kde_prop = gaussian_kde(prop_pooled, bw_method=0.05)
    ax.plot(x_kde, kde_gbm(x_kde), lw=2.2, color='#e74c3c', label='GBM')
    ax.plot(x_kde, kde_prop(x_kde), lw=2.2, color='#27ae60', label='Proposed model')
    ax.set_xlabel('Daily log return')
    ax.set_ylabel('Probability density')
    ax.set_title('Return Distribution: Empirical vs Simulated Models')
    ax.legend(fontsize=10)
    ax.set_xlim(-0.25, 0.25)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(MEDIA_DIR / 'image35.png', dpi=300, bbox_inches='tight')
    print(f"Saved {MEDIA_DIR / 'image35.png'}")
    plt.close(fig)

    # --- Figure 5.3: QQ-plots ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    emp_sorted = np.sort(emp_returns)
    n_emp = len(emp_sorted)
    q_levels = np.linspace(0.001, 0.999, min(n_emp, 500))
    emp_q = np.quantile(emp_returns, q_levels)
    gbm_q = np.quantile(gbm_pooled, q_levels)
    prop_q = np.quantile(prop_pooled, q_levels)

    axes[0].scatter(emp_q, gbm_q, s=6, alpha=0.6, color='#e74c3c')
    lims0 = [min(emp_q.min(), gbm_q.min()), max(emp_q.max(), gbm_q.max())]
    axes[0].plot(lims0, lims0, 'k--', lw=1, alpha=0.5)
    axes[0].set_xlabel('Empirical quantiles')
    axes[0].set_ylabel('GBM quantiles')
    axes[0].set_title('(a) GBM vs Empirical')
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(emp_q, prop_q, s=6, alpha=0.6, color='#27ae60')
    lims1 = [min(emp_q.min(), prop_q.min()), max(emp_q.max(), prop_q.max())]
    axes[1].plot(lims1, lims1, 'k--', lw=1, alpha=0.5)
    axes[1].set_xlabel('Empirical quantiles')
    axes[1].set_ylabel('Proposed model quantiles')
    axes[1].set_title('(b) Proposed model vs Empirical')
    axes[1].grid(True, alpha=0.3)
    fig.suptitle('Quantile-Quantile Plots of Daily Log Returns', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(MEDIA_DIR / 'image36.png', dpi=300, bbox_inches='tight')
    print(f"Saved {MEDIA_DIR / 'image36.png'}")
    plt.close(fig)

    # --- Figure 5.4: ACF of squared returns ---
    max_lag = 60
    lags = np.arange(1, max_lag + 1)
    acf_emp = [acf_at_lag(emp_returns ** 2, l) for l in lags]
    acf_gbm_avg = np.zeros(max_lag)
    acf_prop_avg = np.zeros(max_lag)
    n_sample = min(M_PATHS, 500)
    for m in range(n_sample):
        for li, l in enumerate(lags):
            acf_gbm_avg[li] += acf_at_lag(gbm_returns[:, m] ** 2, l)
            acf_prop_avg[li] += acf_at_lag(prop_returns[:, m] ** 2, l)
    acf_gbm_avg /= n_sample
    acf_prop_avg /= n_sample

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(lags, acf_emp, 'o-', ms=3, lw=1.5, color='#2c3e50', label='Empirical VIX')
    ax.plot(lags, acf_gbm_avg, 's-', ms=3, lw=1.5, color='#e74c3c', label='GBM')
    ax.plot(lags, acf_prop_avg, '^-', ms=3, lw=1.5, color='#27ae60', label='Proposed model')
    ci = 1.96 / np.sqrt(len(emp_returns))
    ax.axhline(ci, ls='--', color='grey', alpha=0.5, lw=0.8)
    ax.axhline(-ci, ls='--', color='grey', alpha=0.5, lw=0.8)
    ax.axhline(0, color='grey', alpha=0.3, lw=0.5)
    ax.set_xlabel('Lag (trading days)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Autocorrelation of Squared Returns')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(MEDIA_DIR / 'image37.png', dpi=300, bbox_inches='tight')
    print(f"Saved {MEDIA_DIR / 'image37.png'}")
    plt.close(fig)

    # --- Print LaTeX-ready table ---
    print("\n\n=== LATEX TABLE (copy into main.tex) ===\n")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Quantitative comparison of simulated return statistics against empirical VIX daily log returns. The proposed model (fuzzy fBm with Poisson jumps, $H=0.15$) is benchmarked against classical Geometric Brownian Motion (GBM) across distributional and temporal dependence indicators. Bold values indicate the model closer to the empirical value.}")
    print(r"\label{tab:benchmark}")
    print(r"\begin{tabular}{l r r r}")
    print(r"\toprule")
    print(r"\textbf{Statistic} & \textbf{Empirical} & \textbf{GBM} & \textbf{Proposed} \\")
    print(r"\midrule")
    for label, key in metrics:
        e = emp_stats[key]
        g = gbm_stats[key]
        p = prop_stats[key]
        g_closer = abs(g - e) <= abs(p - e)
        p_closer = abs(p - e) < abs(g - e)
        g_str = f"\\textbf{{{g:.4f}}}" if g_closer else f"{g:.4f}"
        p_str = f"\\textbf{{{p:.4f}}}" if p_closer else f"{p:.4f}"
        label_tex = label.replace('²', '$^2$').replace('r²', '$r^2$')
        print(f"{label_tex} & {e:.4f} & {g_str} & {p_str} \\\\")
    print(r"\midrule")
    ks_g_closer = ks_gbm <= ks_prop
    ks_p_closer = ks_prop < ks_gbm
    ks_g_str = f"\\textbf{{{ks_gbm:.4f}}}" if ks_g_closer else f"{ks_gbm:.4f}"
    ks_p_str = f"\\textbf{{{ks_prop:.4f}}}" if ks_p_closer else f"{ks_prop:.4f}"
    print(f"KS statistic vs empirical & --- & {ks_g_str} & {ks_p_str} \\\\")
    print(f"KS $p$-value & --- & {pval_gbm:.2e} & {pval_prop:.2e} \\\\")
    print(f"Jarque--Bera statistic & {jb_emp:.1f} & {jb_gbm:.1f} & {jb_prop:.1f} \\\\")
    print(r"\midrule")
    mae_g_str = f"\\textbf{{{mae_gbm:.4f}}}" if mae_gbm <= mae_prop else f"{mae_gbm:.4f}"
    mae_p_str = f"\\textbf{{{mae_prop:.4f}}}" if mae_prop < mae_gbm else f"{mae_prop:.4f}"
    rmse_g_str = f"\\textbf{{{rmse_gbm:.4f}}}" if rmse_gbm <= rmse_prop else f"{rmse_gbm:.4f}"
    rmse_p_str = f"\\textbf{{{rmse_prop:.4f}}}" if rmse_prop < rmse_gbm else f"{rmse_prop:.4f}"
    print(f"Aggregate MAE of statistics & --- & {mae_g_str} & {mae_p_str} \\\\")
    print(f"Aggregate RMSE of statistics & --- & {rmse_g_str} & {rmse_p_str} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print("\n\nDone. All figures saved to", MEDIA_DIR)


if __name__ == '__main__':
    main()
