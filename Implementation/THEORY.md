# Mathematical specification

This is a research implementation, not a claim that the components or their combination are novel. The finite-population formula, inverse-probability estimation and martingale inequality are established tools. The candidate policy is a hypothesis to test against simpler sampling schemes.

## 1. Immediate experiment: order-independent information concentration

For one stored numeric vector, let `N` be its number of inputs and set `y_i = x_i²`, `mu = sum(y_i)/N`, and `S² = sum((y_i-mu)²)/(N-1)`. The final stored code is excluded. A simple random sample of `m` positions without replacement gives:

\[
\widehat\mu_m=\frac1m\sum_{i\in S_m}y_i,\qquad
\mathbb E[\widehat\mu_m\mid x]=\mu,\qquad
\operatorname{Var}(\widehat\mu_m\mid x)
=\left(1-\frac mN\right)\frac{S^2}{m}.
\]

This statement conditions on the observed vector. It does not require temporal order, independent values, or a channel interpretation. Each Monte Carlo replicate uses a random permutation; different budgets are nested within that replicate. Monte Carlo mean squared error should approach the exact variance as repetitions increase.

For `mu > 0`, Chebyshev gives the conservative relative-error bound:

\[
\Pr\{ |\widehat\mu_m-\mu|>\epsilon\mu\mid x\}
\le \min\!\left(1,\frac{(1-m/N)S^2}{m\epsilon^2\mu^2}\right).
\]

A sufficient budget for this upper bound to be at most `alpha` is:

\[
m\ge\left\lceil\frac{NS^2}{S^2+N\alpha\epsilon^2\mu^2}\right\rceil.
\]

The software reports this as an **oracle characterization**: `S²` and `mu` come from the full vector. It is not a deployable stopping criterion and is not the minimum budget actually needed. Constant vectors need one observation for their constant mean; the all-zero vector is handled explicitly. RMS is the square root of mean square: its estimator is generally biased even when the power estimator is unbiased. In mixed channels, vector RMS is only a numeric summary, not a physical single-sensor RMS.

The largest-value concentration curve sorts `x_i²` and shows the cumulative fraction of total squared amplitude. Sorting is permitted for this full-data descriptor; it is not used by a sensing policy. It cannot by itself establish whether large values form temporal bursts.

## 2. Waveform target and measurement cost

This branch requires confirmed raw waveform status, within-record time order, channel count and layout. Each channel is treated separately. Sampling rate may remain unknown; axes then use sample index and cycles/sample. No inter-channel timing or localization inference is made.

Partition a channel into `T` nonoverlapping blocks of at most `L` consecutive values. Retain the final short block. Let `n_t` be its length and `w_t=n_t/N`. The six-dimensional nonnegative feature vector `Y_t` contains:

1. Raw block mean square.
2. Block-demeaned mean square (within-block AC power).
3. Four band energies, each covering one quarter of the Nyquist interval, from the block-demeaned periodic-Hann periodogram.

The FFT length is `L`, including for a final short block. Hann weighting uses its actual length. Band energy is the sum of the relevant one-sided PSD bins times the frequency spacing. Nyquist is included in the fourth band. The full-data reference is:

\[
\theta=\sum_{t=1}^{T}w_tY_t.
\]

Raw power is exactly the full-channel mean square. AC power removes each block's own mean, not just the global mean. The spectral reference is the weighted average of these block spectra, **not the full-record Fourier spectrum**. Therefore different block sizes define different spectral targets; compare methods at a common block size, and label any block-size sensitivity study accordingly. Band powers need not sum to the unwindowed AC mean square.

If `A_t` indicates acquiring block `t`, the cost proxy is:

\[
c=\sum_t w_tA_t.
\]

This counts all acquired values, including the compulsory first adaptive block. It does not measure battery energy, wake-up cost, analog front-end power, feature-computation cost, or transmission overhead. Real deployment must measure these separately.

## 3. Candidate causal acquisition policy

For each monitoring record the policy resets. For `q < 1`, the first block is acquired with probability one. Later probabilities are chosen before the current block is observed:

\[
p_t=\operatorname{clip}\{q(a+g s_{t-1}),\ q\rho,\ 1\}.
\]

Defaults are `a=0.5`, `g=3`, `rho=0.25`. `q` is a nominal control parameter, not the realized acquisition fraction. After an acquired block, compare its raw power to an exponential moving average, and compare its normalized four-band spectrum to the moving average's normalized spectrum:

\[
d_P=\frac{|Y_{t,1}-\bar Y_{t-1,1}|}{Y_{t,1}+\bar Y_{t-1,1}},\qquad
d_S=\tfrac12\left\|\frac{Y_{t,3:6}}{\sum_{j=3}^{6}Y_{t,j}}
-\frac{\bar Y_{t-1,3:6}}{\sum_{j=3}^{6}\bar Y_{t-1,j}}\right\|_1.
\]

Zero denominators use the zero vector and a numerical floor. The first acquired block initializes the moving average and has innovation zero. The main policy uses `(d_P+d_S)/2`; the power-only ablation uses `d_P`. The score becomes `max(previous score, innovation)` after an acquisition, then decays by `0.9` every block. The feature moving average updates only upon acquisition, with rate `0.2`. A block's observed content can affect only later decisions. No labels, full-data RMS, event intervals or future spectra enter the decision rule.

The evaluator precomputes all reference features for replay efficiency, but `sample_record` accesses current features only after a positive acquisition decision. Enumeration and hidden-block perturbation tests check this boundary. The method can fail to see an isolated first burst: positive exploration is essential but cannot remove that risk. A repeated-burst signal may be more predictable; this is an experimental hypothesis, not a promised result.

## 4. Conditional inverse-probability estimator

Condition on the fixed full waveform. Let the history contain previous random decisions and previously acquired features. Use fresh independent uniforms so that `A_t`, conditional on the history, is Bernoulli with predictable `p_t >= p_min > 0`. Define:

\[
\widehat\theta=\sum_t w_t\frac{A_t}{p_t}Y_t.
\]

The error increments `D_t=w_t(A_t/p_t-1)Y_t` have conditional mean zero. Thus:

\[
\mathbb E[\widehat\theta\mid x]=\theta,\quad
\operatorname{Var}(\widehat\theta_j\mid x)
=\mathbb E\!\left[\sum_t w_t^2\frac{1-p_t}{p_t}Y_{t,j}^{2}\,\middle|\,x\right].
\]

An unbiased estimator of that unconditional, within-record design variance is:

\[
\widehat V_j=\sum_t w_t^2 A_t\frac{1-p_t}{p_t^2}Y_{t,j}^{2}.
\]

This is conditional-probability weighting for a sequential design; do not silently replace `p_t` with an observed overall fraction. The `adaptive_naive` ablation divides observed weighted feature sums by observed total weight. It uses the same policy masks as the main candidate but lacks the inverse-probability correction. A biased estimator can still have smaller mean squared error; report both signed bias and error.

The uniform fixed-count baseline samples blocks without replacement. Its block inclusion probability is `m/T`, and it retains `w_t` in the estimator, including the partial tail. The Bernoulli variance formula above is **not** applied to this baseline. Prefix and periodic schedules are descriptive baselines without a design-based uncertainty claim. `q=1` always acquires everything.

## 5. A conservative fixed-horizon uncertainty bound

Suppose a justified bound `|x_i| <= A` is supplied before evaluating the record. Then valid feature bounds are `U=[A²,A²,4A²,4A²,4A²,4A²]`. The AC variance is at most `A²`. After demeaning, each value has magnitude at most `2A`, and Parseval's identity bounds the Hann-weighted total spectral energy by `4A²`; any individual nonnegative band is also bounded by this value. An observed maximum is not a justified bound for unseen values.

For coordinate `j`, both signs of the error martingale have increments bounded in magnitude by `b_j=(max w_t)U_j/p_min` and predictable quadratic variation at most:

\[
v_j=U_j^2\frac{1-p_{\min}}{p_{\min}}\sum_t w_t^2.
\]

For simultaneous coverage of `d=6` features at one specified horizon, set `ell=log(2d/alpha)`. A two-sided Freedman/Bernstein bound and a union bound yield:

\[
h_j=\frac{b_j\ell}{3}+\sqrt{2v_j\ell+\left(\frac{b_j\ell}{3}\right)^2},
\qquad \Pr\{\forall j:|\widehat\theta_j-\theta_j|\le h_j\}\ge1-\alpha.
\]

The code projects the center to `[0,U_j]` and intersects the interval with `[0,U_j]`, which cannot reduce coverage relative to the unprojected bound. At full acquisition it returns the exact target. If a supplied amplitude bound is violated, that record's formal intervals are disabled and the violation is reported. With no justified bound, formal intervals are unavailable, even though descriptive errors can still be calculated.

The guarantee concerns **sampling randomness for this fixed waveform and horizon**, not new sensors, new physical conditions or future records. We do not implement an anytime confidence sequence for the unobserved full-record target, and we do not claim arbitrary data-dependent stopping is valid. Bounds may be almost the entire parameter range. Report joint coverage, width and vacuity together. Normal intervals use `V_hat` and a Bonferroni-adjusted Gaussian quantile as a heuristic; missing a burst may make them severely overconfident.

## 6. Probability of never acquiring an event

For a fixed event occupying `m_e` eligible blocks and any predictable design with `p_t >= p_min`, iterated conditioning gives:

\[
\Pr\{\text{no block overlapping event }e\text{ acquired}\}\le(1-p_{\min})^{m_e}.
\]

A union bound over specified events bounds the probability that at least one event is wholly missed by `min(1,sum_e(1-p_min)^m_e)`. This can be conservative. Acquiring an overlapping block does not prove the event was detected or correctly classified. Only synthetic burst families have known event intervals in this package; there are no fabricated real event annotations.

## 7. Source-only empirical interval calibration

For each held-out domain, use other domains for calibration and exclude parent groups shared with the held-out domain. Use only the prespecified first sampling replicate. A supplied acquisition run ID groups all its records and channels; otherwise exact-input identity is the available grouping, with independence unverified.

The calibration score for a record/channel is the largest feature error divided by `abs(prediction)+source_floor`, where the floor comes from source truth values only. Maximize over all members of a parent group, then take order statistic `ceil((n_source_groups+1)*nominal_level)`. If that index exceeds the number of source groups, the radius is infinite and is reported as such. Source-only scale estimation and physical distribution shift mean these are **empirical transfer intervals**, not a conformal coverage theorem. Calibration rows, group counts, widths and held-out coverage are exported. Held-out full-data values are used only for scoring.

## References and attribution

- D. G. Horvitz and D. J. Thompson (1952), *A Generalization of Sampling Without Replacement from a Finite Universe*, JASA 47:663–685. https://doi.org/10.1080/01621459.1952.10483446 . Inverse-probability weighting is established prior art.
- D. A. Freedman (1975), *On Tail Probabilities for Martingales*, Annals of Probability 3:100–118. https://doi.org/10.1214/aop/1176996452 . The scalar inequality is also stated in Theorem 1.1 of J. A. Tropp (2011): https://tropp.caltech.edu/papers/Tro11-Freedmans-Inequality.pdf . Our coordinate-wise finite-horizon application is not a new concentration theorem.
- I. Waudby-Smith and A. Ramdas (2020), *Confidence Sequences for Sampling Without Replacement*. https://proceedings.neurips.cc/paper/2020/hash/e96c7de8f6390b1e6c71556e4e0a4959-Abstract.html . Relevant related work; its confidence sequences are not implemented here.
- TISEAN surrogate-data documentation: https://www.pks.mpg.de/tisean/TISEAN_2.1/docs/docs_f/surrogates.html . The implemented simple Fourier phase surrogate preserves global Fourier magnitudes and RMS, not the exact amplitude histogram or blockwise spectra. IAAFT is not implemented.
