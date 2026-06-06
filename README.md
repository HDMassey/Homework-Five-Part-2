# GSE 544 — Homework Five, Part 2
## A Copula/NCO Horse Race Under Realistic Constraints

---

## Acknowledgement

I used Claude (Anthropic) and the class notebooks provided by the instructor to complete this assignment. The code closely follows the lecture notebook `4_copula_nco.ipynb` from the course repository, extended with long-only and no-borrowing constraints and CRRA utility for two risk-aversion regimes.

---

**Course:** GSE 544, Spring 2026  
**Deliverable:** Jupyter notebook (`A_Copula_NCO_Horse_Race_Under_Realistic_Constraints.ipynb`)  
**Data:** `vmls_portfolio_returns.csv` · `copula_model.pkl` (from class repo `06_Portfolio_optimization_and_copulas/`)

---

## Background

This is a constrained variant of the Topic-6 lecture notebook `4_copula_nco.ipynb`. Same data (the 19 risky VMLS assets, 2000-day training window), same fitted Student-$t$ marginals + Gaussian copula loaded from `copula_model.pkl`, same Monte Carlo machinery. Three changes from the lecture:

1. **No shorting.** Every risky weight is constrained $w_j \geq 0$.
2. **No borrowing.** The risk-free asset absorbs the residual budget $b(w) = 1 - \sum_j w_j$, lent at the training mean $\bar{r}^f$, and that residual is constrained to be non-negative — so $\sum_j w_j \leq 1$.
3. **Two utility regimes.** Everything is solved twice: under log utility ($\gamma = 1$, full Kelly) and under CRRA $\gamma = 3$.

With no shorting and no borrowing, the daily gross growth is:

$$g_t(w) = b(w)\,\bar{r}^f + \sum_j w_j r_{t,j}, \qquad b(w) = 1 - \sum_j w_j \geq 0,\quad w_j \geq 0$$

Generate Monte Carlo draws by loading `copula_model.pkl` and simulating with the fixed random seeds used in the lecture notebook (`RNG_SEED = 778`, `N_MC = 500,000`). Apply the lecture's per-asset $2\times$-historical clip to simulated draws only, and reuse the same converged sample across both $\gamma$ regimes so any cross-regime difference is purely an objective effect.

**Three portfolios run side by side in each regime:**

- **A — NCO.** Cluster the 19 assets by correlation distance $D_{ij} = \sqrt{\tfrac{1}{2}(1-\rho_{ij})}$ ($k$-means on the distance columns, $K=5$ clusters); fit a Gaussian copula and run the constrained CRRA solve within each cluster; fit a copula on the cluster portfolios and solve across them; compose the final weights.
- **B — All-at-once.** One Monte Carlo over the full 19-asset joint model, one constrained CRRA solve.
- **1/N benchmark.** $w_j = 1/20$ for every asset, including the risk-free asset.

---

## Questions (50 points total)

### Part (a) — Build the Three Portfolios (15 pts)

Implement the constrained CRRA optimizer (long-only weights summing to at most 1) and the copula-simulation helpers, then construct Portfolios A, B, and 1/N for both $\gamma = 1$ and $\gamma = 3$.

For each portfolio and regime, report:
- The risky weights
- The residual share left in the risk-free asset, $1 - \sum_j w_j$

For Portfolio A, also report:
- The across-cluster weights
- The within-cluster weight sums

---

### Part (b) — Out-of-Sample Horse Race (15 pts)

Apply each portfolio's fixed training weights to the real 500-day held-out test returns (no clipping, no re-simulation, no rebalancing).

For each $\gamma$ regime:

**(i)** Plot the three wealth paths together starting from $10,000.

**(ii)** Plot the three realized daily log-growth histograms together, marking each mean.

Then assemble a summary table reporting, for each (regime, portfolio):

| Column | Description |
|---|---|
| RF share | $1 - \sum_j w_j$ |
| Final wealth ($) | Terminal portfolio value |
| Total return | $(V_T / 10{,}000) - 1$ |
| Mean daily log-growth | $\frac{1}{T}\sum_t \log(1+g_t)$ |
| Worst daily log-growth | $\min_t \log(1+g_t)$ |
| Realized $\mathbb{E}[u_\gamma]$ | CRRA utility on the test set |

---

### Part (c) — Why the Correlation Distance? (8 pts)

MLdP's clustering recipe does not feed the correlation matrix $\rho_{ij}$ to $k$-means directly — it first maps it to the correlation distance $D_{ij} = \sqrt{\tfrac{1}{2}(1-\rho_{ij})}$.

In a short paragraph, explain why. Address:

1. Why a clustering algorithm needs a true metric and correlation is not one.
2. What goes wrong with the sign of $\rho$ if you cluster on the raw number — think about a $\rho = -0.95$ hedge versus a $\rho = +0.95$ co-mover.
3. What it means that we cluster on each asset's full distance profile (its column of $D$) rather than on pairwise numbers.

---

### Part (d) — Synthesis (12 pts)

Using your figures and summary table, answer the three horse-race questions in a few short paragraphs:

**1. Does NCO still beat the all-at-once approach once shorting and borrowing are disallowed?**  
What does the no-short constraint do to NCO's across-cluster step in particular — look at where the across-cluster weights end up — and what does that imply about where NCO's lecture-notebook edge actually came from?

**2. How much does each optimizer choose to leave in the risk-free asset, and how does that change with $\gamma$?**  
Compare the risk-free share $1 - \sum_j w_j$ for Portfolios A and B across the two regimes. Which portfolio holds more cash, and does the more risk-averse agent ($\gamma = 3$) pull more weight out of the risky assets, as you'd expect?

**3. Does raising $\gamma$ from 1 to 3 reshuffle the ranking?**  
In particular, how do the optimized portfolios compare to the 1/N benchmark at $\gamma = 3$ — on mean log-growth and on the worst day — and what is the general lesson (cf. DeMiguel, Garlappi & Uppal 2009, https://doi.org/10.1093/rfs/hhm075) about sophisticated optimization versus naive equal-weighting once realistic constraints are imposed?

---

## File Structure

```
.
├── README.md                              # This file
├── A_Copula_NCO_Horse_Race_Under_Realistic_Constraints.ipynb   # Solution notebook
└── data_and_class_notes/
    ├── vmls_portfolio_returns.csv         # Return data (from class repo)
    ├── copula_model.pkl                   # Fitted copula model (from class repo)
    └── 4_copula_nco.ipynb                 # Lecture notebook (reference)
```

## Dependencies

```
numpy
pandas
scipy
matplotlib
seaborn
```

## Reference

DeMiguel, V., Garlappi, L., & Uppal, R. (2009). Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy? *Review of Financial Studies*, 22(5), 1915–1953. https://doi.org/10.1093/rfs/hhm075
