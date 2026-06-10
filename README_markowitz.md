# Portfolio Optimization - Markowitz & Beyond

**Author:** Louen MARX

## Description

This project explores **portfolio optimization** through the lens of Markowitz's Modern Portfolio Theory (MPT) and its extensions. The core idea is to construct an optimal portfolio of financial assets that minimizes risk (variance) while achieving a target expected return.

The project is split into two parts:

- **Part 1** (`Marx_Projet_Q1_Q2_Q3_final.py`): warm-up on constrained quadratic optimization with CVXPY and SciPy, comparing solver performance and scalability.
- **Part 2** (`Marx_projet_final.py`): the main Markowitz portfolio optimization project, covering standard mean-variance optimization, probabilistic constraints, higher-order moment minimization (kurtosis), and advanced optimization techniques (SQP, SCP, SDP relaxation).

Financial data is downloaded from Yahoo Finance via `yfinance`, using real stock prices from 2016 to 2021. The training period (2016-2019) is used to estimate expected returns and the covariance matrix, while the testing period (2020-2021) evaluates out-of-sample performance, including the impact of the COVID-19 market turmoil.

## Mathematical Background

### Standard Markowitz Problem (Q5)

We solve the classic mean-variance optimization:

```
min_x   x^T Σ x
s.t.    μ^T x >= r_min
        Σ x_i = 1
        x >= 0
```

Where:
- `x` = vector of portfolio weights (allocation to each asset)
- `Σ` = covariance matrix of asset returns
- `μ` = vector of expected returns
- `r_min` = minimum acceptable return

### Probabilistic Markowitz (Q6)

We add a chance constraint to control the probability of the portfolio return falling below a threshold `alpha`:

```
P(r^T x <= alpha) <= beta
```

This is reformulated using the inverse CDF of the normal distribution (`k = -Φ⁻¹(β)`), yielding a second-order cone constraint:

```
||Σ^{1/2} x|| <= (μ^T x - alpha) / k
```

### Kurtosis Minimization (Q13, Q14, Q17)

Beyond variance, we minimize the **kurtosis** (4th-order moment) of the portfolio to reduce the likelihood of extreme returns (fat tails). This involves:
- Computing the co-kurtosis matrix `S4`
- Using duplication and elimination matrices (`D2`, `L2`) for vectorization
- Solving via SCP (Sequential Convex Programming), SLSQP, or SDP relaxation

## Project Structure

### Part 1: Solver Comparison (`Marx_Projet_Q1_Q2_Q3_final.py`)

| Question | Topic | Description |
|----------|-------|-------------|
| Q1 | Constrained Least Squares | Solve `min 0.5||Ax - b||²` subject to `0 <= x <= 1` using CVXPY |
| Q2 | Scalability Study | Measure solve time vs problem dimension (n from 20 to 2000) |
| Q3 | CVXPY vs SciPy | Compare CVXPY and SLSQP solver performance and solution accuracy |

### Part 2: Markowitz Portfolio (`Marx_projet_final.py`)

| Question | Topic | Description |
|----------|-------|-------------|
| Q5 | Standard Markowitz | Mean-variance portfolio optimization for different numbers of assets |
| Q6 | Probabilistic Markowitz | Chance-constrained optimization with varying beta and asset counts |
| Q7 | Out-of-sample Testing | Re-evaluate portfolios on extended test data (2020-2021) |
| Q8 | Data Visualization | Plot time series of asset returns (training vs testing) |
| Q9 | Monte Carlo Simulation | Generate random portfolios to approximate the efficient frontier |
| Q11 | SQP Solver | Implement Sequential Quadratic Programming for a non-convex problem |
| Q13 | SCP for Kurtosis | Sequential Convex Programming with trust region for kurtosis minimization |
| Q14 | SLSQP for Kurtosis | Scipy's SLSQP applied to kurtosis minimization |
| Q15 | Visualization | Monte Carlo scatter plots and portfolio composition pie charts |
| Q17 | SDP Relaxation | Semidefinite programming relaxation for kurtosis, with tightness analysis |

## Data

- **Source:** Yahoo Finance (`yfinance`)
- **Assets:** 23 US stocks (JCI, TGT, CMCSA, JPM, MSFT, BA, etc.)
- **Training:** 2016-01-01 to 2019-12-30
- **Testing:** 2020-01-01 to 2021-12-30
- **Returns:** Daily percentage returns computed from adjusted closing prices

## Installation

```bash
pip install numpy pandas matplotlib yfinance cvxpy scipy riskfolio-lib pylops
```

## Usage

```bash
# Part 1: Solver comparison (Q1-Q3)
python Marx_Projet_Q1_Q2_Q3_final.py

# Part 2: Markowitz portfolio optimization (Q5-Q17)
python Marx_projet_final.py
```

Note: Part 2 requires an internet connection to download financial data from Yahoo Finance.

## Key Results

- **Markowitz vs Probabilistic:** The probabilistic variant produces more conservative allocations, reducing tail risk at the cost of slightly lower returns.
- **Impact of COVID-19:** Out-of-sample testing shows higher volatility and lower returns in 2020-2021 compared to the training period.
- **Diversification:** Increasing the number of assets improves risk-adjusted returns up to a point, after which marginal benefits diminish.
- **Kurtosis vs Variance:** Minimizing kurtosis (4th moment) instead of variance (2nd moment) produces portfolios that are more robust to extreme events.
- **SDP Relaxation:** The semidefinite relaxation is often tight (`X ≈ xx^T`), meaning it provides near-exact solutions to the original non-convex kurtosis problem.

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `yfinance`
- `cvxpy`
- `scipy`
- `riskfolio-lib`
