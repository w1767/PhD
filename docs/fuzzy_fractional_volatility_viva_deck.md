# From Pricing to Forecasting: Fuzzy Mixed Fractional Models for Volatility

## Motivation and Contributions
- Classical Gaussian and memoryless models overlook long-memory effects (Hurst exponent H ≠ 0.5), volatility roughness, jumps, and parameter ambiguity.
- We propose a unified fuzzy fractional framework that captures rough dynamics, compound Poisson jumps, and fuzzy parameter uncertainty to generate superior implied-volatility inputs for VIX/VVIX forecasting.

## Empirical Premise: Volatility is Fractal
- Empirical diagnostics show volatility clustering and Hurst exponents departing from 0.5, especially in rough regimes with H < 1/2.
- Multi-horizon clustering and H-estimation reveal persistence/anti-persistence patterns ignored by standard ARIMA, LSTM, or N-BEATS baselines.
- These observations motivate a structural approach grounded in fractional stochastic calculus.

## Core Model: Mixed Fractional Brownian Motion with Jumps and Fuzziness
- Driving noise combines standard Brownian motion (B_t) and fractional Brownian motion (B_t^H) to represent simultaneous semimartingale and rough features.
- Jump component (sum_{i=1}^{N_t} V_i) follows a compound Poisson process with lognormal jump sizes.
- Parameters (mu, sigma, lambda) and jump magnitudes are modelled as fuzzy numbers evaluated via alpha-cuts to encode uncertainty bands across belief levels.

## Risk-Neutral Pricing with Jumps and Fuzzy Parameters
- No-arbitrage drift adjustment: mu -> r - lambda (J-1), where J = E[V_i] = exp(mu_j + 0.5 sigma_j^2).
- European call pricing obtained by conditioning on Poisson jump counts and summing the Black-Scholes-like contributions with Phi(d1), Phi(d2).
- Fuzzy layer evaluates the pricing functional over alpha-cuts to produce interval-valued prices; inversion yields fuzzy implied volatility ranges.

## Key Derivation Highlights
- Jump product handled via product V_i = exp(sum ln V_i) using the jump integral representation.
- Exponential moment identities: E[e^{sigma B_t}] = e^{0.5 sigma^2 t} and E[e^{sigma_H B_t^H}] = e^{0.5 sigma_H^2 t^{2H}}.
- Compound Poisson expectations computed by conditioning on N_t; independence enables closed-form expressions.
- Interpolation and belief-degree mapping maintain consistency between fuzzy inputs and option-price membership functions.

## Handling the Rough Regime (H < 1/2)
- Non-semimartingale nature complicates classical Ito and Girsanov approaches.
- Apply Jost transformation (H -> 1-H) and Volterra integral representations to manage kernel singularities.
- Molchan martingale estimates and Burkholder-Davis-Gundy inequalities provide bounds for sup |B_t^H|.
- Result: existence and uniqueness for solutions with fuzzy amplitudes while preserving crisp noise realizations.

## Fuzzy Gaussian Aggregation via alpha-Cuts
- For each belief level alpha, parameters instantiate a Gaussian distribution N(mu_alpha, sigma_alpha^2).
- Aggregator principle: membership of value x equals sup{alpha : exists (mu, sigma^2) in alpha-cut with pdf >= tau}, ensuring coherent fuzzy-normal semantics.
- The mapping from fuzzy inputs to fuzzy outputs adheres to the Zadeh extension principle.

## Simulation Architecture (Illustrative Configuration)
- Fractional Brownian motion simulated with H = 0.15 over T = 1 year at 252 daily steps via Cholesky decomposition of the covariance matrix Gamma_{ij}.
- Increments recovered by differencing the simulated levels; compound Poisson jumps use intensity lambda = 0.5 and lognormal sizes.
- Fuzzy volatility represented by sentiment-driven triangular membership functions yielding sigma intervals for each alpha.

## Pipeline: From Pricing to VIX Forecasting
1. Ingest SPX option strips aligning with official VIX constituents.
2. Invert the fuzzy fractional-jump pricing model for each alpha level to obtain implied-volatility intervals per strike and maturity.
3. Aggregate across strikes/maturities to derive a model-aware VIX time series capturing roughness, jumps, and fuzziness.
4. Feed the enhanced series into downstream forecasters (ARIMA, LSTM, N-BEATS) and evaluate out-of-sample performance.

## Extending to VVIX (Volatility-of-Volatility)
- Apply identical inversion to VIX options to derive interval estimates of implied volatility-of-volatility.
- The resulting VVIX-like series inherently respects the structural features of the model, enabling consistent higher-moment forecasting.

## Innovations and Scholarly Output
- Unified fuzzy-fractional-jump pricing with closed-form risk-neutral valuation under Q.
- Existence and uniqueness proofs for rough regimes leveraging Jost transformations, Molchan martingales, and BDG inequalities.
- Fuzzy implied-volatility inversion via alpha-cut root finding, yielding interval-valued VIX/VVIX measures.
- Simulation stack combining Cholesky fBm sampling, compound Poisson jumps, and fuzzy volatility scenarios.
- Interpolation framework translating fuzzy beliefs into actionable price and volatility bands.
- Placeholder for referencing published papers and preprints that document theoretical and empirical advances.

## Limitations and Roadmap
- Full-scale empirical inversion and backtesting remain future work due to data-ingestion and computational constraints.
- Next steps include:
  - Building high-frequency options pipelines and robust alpha-cut inversion tooling.
  - Generating interval VIX/VVIX series, forecasting them, and benchmarking with Diebold-Mariano tests.
  - Conducting ablation studies to quantify contributions of Hurst exponent variation, jumps, and fuzziness.

## GAN-Based Feature Generator Overview
- Training data: 100-point time-series segments reshaped into 10x10 grids, zero-padded to 32x32 with a single channel, yielding tensors of shape (N, 32, 32, 1).
- Latent dimension: 100; training spans 20 epochs with batch size 64.
- Discriminator architecture: four Conv2D blocks (128 filters, 3x3 kernels, stride 2) each with LeakyReLU (alpha=0.2) and Dropout (rate 0.25); Flatten -> Dense(1, sigmoid).
- Generator architecture: Dense(128*4*4) -> LeakyReLU -> Reshape(4,4,128) followed by three Conv2DTranspose blocks (128 filters, 4x4 kernels, stride 2, "same" padding) each with LeakyReLU; final Conv2D(1, 3x3, tanh) delivering (None, 32, 32, 1) outputs.
- Training regimen: discriminator trained on mini-batches of real and synthetic samples; generator updated via adversarial loss with GAN optimizer Adam (lr=1e-4, beta1=0.5).
- Synthetic outputs visualized as grayscale 32x32 arrays encoding normalized time-series structure rather than natural imagery.

## Strengths
- Rich feature synthesis through adversarial training captures latent temporal structure and augments limited datasets.
- Layer diversity (convolutions + optional bidirectional LSTMs) supports both local pattern extraction and long-term dependency modelling.
- Regularization via dropout and LeakyReLU mitigates vanishing gradients and stabilizes training.
- Modular design enables independent refinement of generator, discriminator, and downstream forecasting components.

## Weaknesses
- Unconventional reshaping of 1-D time series into 2-D images obscures temporal ordering and requires rigorous justification.
- GAN training is susceptible to mode collapse, especially on small, noisy financial datasets.
- Hyperparameters (learning rates, batch sizes) were fixed heuristically without systematic tuning.
- Expanding to 32x32 representations increases memory and compute overhead compared with native 1-D processing.

## Suggested Improvements
- Finalize a 1-D GAN variant using Conv1D and bidirectional LSTM layers to respect temporal structure.
- Incorporate stabilization techniques such as Wasserstein loss with gradient penalty or spectral normalization.
- Employ automated hyperparameter search (e.g., Keras Tuner) to optimize learning rates, dropout, latent dimensionality, and batch sizes.
- Document or revise the data reshaping pipeline to justify the 2-D representation or replace it with a strictly sequential approach.
