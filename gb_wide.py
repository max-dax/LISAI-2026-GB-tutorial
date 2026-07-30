"""Wide-prior galactic-binary simulator and prior conditioning (Part 2a).

Part 1 assumes a post-search frequency prior, :math:`\\delta f \\sim U[\\pm 10^{-7}]` Hz, narrow
enough that the complex envelope is resolved by ``N = 256`` samples.  Part 2a widens that prior
by a factor ``W``.  The envelope then winds ``W`` times faster and has to be sampled ``R`` times
more finely, ``N_WIDE = R * N``.

The physics is *unchanged*: ``envelope`` below is ``gb_simulator.gb_envelope`` evaluated on a
different time grid, obtained by swapping the module-level grid rather than by copying the body,
so the repository holds exactly one copy of the waveform.  The only new convention is the
:math:`1/\\sqrt{R}` in ``signal_wide``: whitened samples taken ``R`` times more often carry
:math:`\\sqrt{R}` less signal each at fixed total signal-to-noise ratio.

``condition(d, df_hat)`` is the operation the tutorial is about.  It demodulates the wide data by
a frequency proxy, block-averages ``R -> 1``, and rescales to unit noise variance; what comes back
is a 256-sample data vector distributed exactly as Part 1's, with
:math:`\\delta f \\to \\delta f - \\hat{\\delta f}`.

The source of truth for the simulator is the setup cell of
``part2a_conditioning_SOLUTIONS.ipynb``, which defines the same functions inline so that the
notebook runs standalone on Colab.  Run ``python gb_wide.py`` for a self-test.

    from gb_wide import configure, sample_prior_wide, simulate_wide, log_likelihood_wide, condition
"""

import numpy as np
import torch

import gb_simulator as gb
from gb_simulator import N, T_OBS, YEAR, device

DT = T_OBS / (N - 1)          # Part 1 sample spacing
DF_P1 = 1e-7                  # Part 1 prior half-width in delta f


def envelope(theta, t):
    """``gb_simulator.gb_envelope`` evaluated on an arbitrary time grid. (B, 5) -> (B, len(t))."""
    t_save = gb.t
    gb.t = t
    try:
        return gb.gb_envelope(theta)
    finally:
        gb.t = t_save


TILE_HALF = DF_P1 / 2         # tile half-width: half of what the network was trained on


def configure(W=20, R=4):
    """Set the prior-widening factor and the oversampling factor. Returns the module globals."""
    global W_FACTOR, R_FACTOR, N_WIDE, t_wide, PRIOR_LOW_W, PRIOR_HIGH_W, DF_HAT
    W_FACTOR, R_FACTOR, N_WIDE = W, R, R * N
    # Every R-th wide sample lands exactly on Part 1's grid, which is what `condition` returns.
    t_wide = torch.arange(N_WIDE, device=device) * (DT / R)
    PRIOR_LOW_W = gb.PRIOR_LOW.clone()
    PRIOR_HIGH_W = gb.PRIOR_HIGH.clone()
    PRIOR_LOW_W[1], PRIOR_HIGH_W[1] = -W * DF_P1, W * DF_P1
    # 2W tiles of half Part 1's prior width, exactly tiling the widened prior.  Spacing the
    # proxies by half the trained prior width keeps the residual delta f inside the middle
    # half of what the network has seen, which is what makes tile boundaries harmless.
    DF_HAT = -W * DF_P1 + (torch.arange(2 * W, device=device) + 0.5) * DF_P1
    return W_FACTOR, R_FACTOR, N_WIDE


configure()


def sample_prior_wide(n):
    """Draw n parameter vectors from the widened prior. -> (n, 5)"""
    return PRIOR_LOW_W + (PRIOR_HIGH_W - PRIOR_LOW_W) * torch.rand(n, 5, device=device)


def signal_wide(theta):
    """Noise-free wide-grid features. (B, 5) -> (B, 2 N_WIDE) real, [Re, Im] concatenated."""
    y = envelope(theta, t_wide) / np.sqrt(R_FACTOR)
    return torch.cat([y.real, y.imag], dim=-1)


def simulate_wide(theta):
    """(B, 5) -> (B, 2 N_WIDE) real features, with unit-variance whitened noise."""
    d = signal_wide(theta)
    return d + gb.NOISE_STD * torch.randn_like(d)


def log_likelihood_wide(theta, d):
    """Exact Gaussian log likelihood on the wide data (up to a constant)."""
    return -0.5 * ((d - signal_wide(theta)) ** 2).sum(-1) / gb.NOISE_STD**2


def condition(d, df_hat):
    """Demodulate by df_hat, keep the central N frequency bins, rescale.

    A brick-wall band selection, not a block average: block-averaging R samples is a sinc filter
    whose first sidelobe is only 13 dB down, so out-of-band sources leak back in at the per-cent
    level, whereas truncating the spectrum removes them up to the leakage of a non-periodic
    record.  The 1/sqrt(R) restores unit noise variance -- only N of the N_WIDE noise bins survive.

    (..., 2 N_WIDE) -> (..., 2 N)
    """
    z = torch.complex(d[..., :N_WIDE], d[..., N_WIDE:])
    z = z * torch.exp(-2j * np.pi * df_hat * t_wide)
    spectrum = torch.fft.fftshift(torch.fft.fft(z), dim=-1)
    lo = (N_WIDE - N) // 2
    z = torch.fft.ifft(torch.fft.ifftshift(spectrum[..., lo:lo + N], dim=-1)) / np.sqrt(R_FACTOR)
    return torch.cat([z.real, z.imag], dim=-1)


def log_prior_tile(theta, df_hat):
    """Log density of the prior restricted to one tile: uniform, zero outside. (B,)"""
    low, high = PRIOR_LOW_W.clone(), PRIOR_HIGH_W.clone()
    low[1], high[1] = df_hat - TILE_HALF, df_hat + TILE_HALF
    inside = ((theta > low) & (theta < high)).all(-1)
    return torch.where(inside, -torch.log(high - low).sum(),
                       torch.tensor(-torch.inf, device=theta.device))


def reference_posterior(log_post, theta0, mean, sd, n_chains=256, n_steps=3000, thin=4, seed=0,
                        n_pilot=3, pilot_steps=300, init=1e-4):
    """Exact posterior by random-walk Metropolis, preconditioned by a pilot covariance.

    Runs in standardized coordinates u = (theta - mean) / sd, where the five parameters are all
    O(1) -- in physical units they span seventeen orders of magnitude and a single proposal
    covariance cannot serve them.  The ensemble starts tightly clustered on ``theta0`` and the
    proposal covariance is re-estimated ``n_pilot`` times, growing to fit the posterior; check the
    returned acceptance rates, which should settle around 0.2-0.4.  Seeded at ``theta0``: this is
    a reference computation, not a search.

    Returns (samples, acceptance rate per stage).
    """
    torch.manual_seed(seed)
    u = (theta0 - mean) / sd + init * torch.randn(n_chains, 5, device=device)
    lp = log_post(u * sd + mean)
    L = init * torch.eye(5, device=device)
    accs, keep = [], []
    stages = [pilot_steps] * n_pilot + [n_steps]
    for stage, ns in enumerate(stages):
        n_acc = 0.0
        for i in range(ns):
            v = u + torch.randn(n_chains, 5, device=device) @ L.T
            lq = log_post(v * sd + mean)
            a = torch.rand(n_chains, device=device).log() < (lq - lp)
            u = torch.where(a[:, None], v, u)
            lp = torch.where(a, lq, lp)
            n_acc += a.float().mean().item()
            if stage == n_pilot and i % thin == 0:
                keep.append(u.clone())
        accs.append(n_acc / ns)
        if stage < n_pilot:
            cov = torch.cov(u.T)
            L = torch.linalg.cholesky(cov + 1e-8 * torch.diag(cov.diag())) * 2.38 / np.sqrt(5)
    return torch.cat(keep) * sd + mean, accs


def _ks_pvalue(q):
    """Asymptotic Kolmogorov-Smirnov p-value against U[0, 1]. (scipy is not a dependency.)"""
    n = len(q)
    x = np.sort(q)
    k = np.arange(1, n + 1)
    d = max(np.max(k / n - x), np.max(x - (k - 1) / n))
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d
    j = np.arange(1, 101)
    return float(np.clip(2 * np.sum((-1) ** (j - 1) * np.exp(-2 * j**2 * lam**2)), 0, 1))


def pp_plot(samples_fn, injections, labels=None, ax=None, confidence=(0.68, 0.95, 0.997),
            n_band=4000, seed=0):
    """Probability-probability plot over a set of injections.

    ``samples_fn(theta)`` takes one injection ``(1, D)``, simulates an observation of it and
    returns posterior samples ``(n, D)``.  For every injection and every parameter we record the
    quantile of the truth in the corresponding one-dimensional marginal.  If the posteriors are
    calibrated those quantiles are uniform, so each empirical CDF follows the diagonal; the grey
    bands are the exact null distribution of the order statistics, obtained by simulation.
    Legend entries give the Kolmogorov-Smirnov p-value against uniformity.

    Returns the ``(n_injections, D)`` array of quantiles.
    """
    import matplotlib.pyplot as plt

    q = np.stack([
        (samples_fn(theta[None]) < theta[None]).float().mean(0).cpu().numpy()
        for theta in injections
    ])
    n, dim = q.shape
    if ax is None:
        _, ax = plt.subplots(figsize=(4.8, 4.8))
    y = np.linspace(0, 1, n)
    null = np.sort(np.random.default_rng(seed).random((n_band, n)), axis=1)
    for c in confidence:
        lo, hi = np.quantile(null, [(1 - c) / 2, (1 + c) / 2], axis=0)
        ax.fill_betweenx(y, lo, hi, color="k", alpha=0.08, lw=0)
    for j in range(dim):
        label = labels[j] if labels is not None else rf"$\theta_{j}$"
        ax.plot(np.sort(q[:, j]), y, lw=1.4, label=f"{label}  ($p$={_ks_pvalue(q[:, j]):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("credible level"); ax.set_ylabel("fraction of injections")
    ax.legend(fontsize=8, loc="upper left")
    return q


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.manual_seed(0)
    configure(20, 4)

    # 1. the wide envelope on the Part 1 grid is Part 1's envelope, exactly
    th = gb.sample_prior(64)
    assert (envelope(th, gb.t) - gb.gb_envelope(th)).abs().max() == 0
    assert gb.t.shape == (N,)          # the grid swap left no trace

    # 2. total SNR is preserved by the widening
    snr_p1 = gb.signal(th).pow(2).sum(-1).sqrt()
    snr_w = signal_wide(th).pow(2).sum(-1).sqrt()
    print(f"SNR ratio wide/Part 1: {(snr_w / snr_p1).min():.4f} .. {(snr_w / snr_p1).max():.4f}")

    # 3. conditioning at the true frequency reproduces the Part 1 signal
    th = gb.sample_prior(64)
    th[:, 1] = (torch.rand(64, device=device) - 0.5) * 2 * TILE_HALF     # residuals inside a tile
    th_w = th.clone()
    th_w[:, 1] = DF_HAT[13] + th[:, 1]
    resid = condition(signal_wide(th_w), DF_HAT[13])
    err = (resid - gb.signal(th)).pow(2).sum(-1).sqrt() / gb.signal(th).pow(2).sum(-1).sqrt()
    print(f"conditioning relative error: max {err.max():.4f}, median {err.median():.4f}")
    assert err.max() < 0.06

    # 3b. a source well outside the retained band is removed, far better than a block average
    th_far = gb.sample_prior(1)
    th_far[:, 1] = DF_HAT[13] - 3.3e-6
    rho = signal_wide(th_far).pow(2).sum().sqrt()
    supp = (condition(signal_wide(th_far), DF_HAT[13]).pow(2).sum().sqrt() / rho).item()
    x = -3.3e-6 * DT
    box = abs(np.sin(np.pi * x)) / (R_FACTOR * abs(np.sin(np.pi * x / R_FACTOR)))
    print(f"out-of-band suppression: {supp:.4f} (brick wall) vs {box:.4f} (block average)")
    assert supp < box / 5

    # 4. conditioning maps unit-variance noise to unit-variance noise
    noise = torch.randn(2000, 2 * N_WIDE, device=device)
    print(f"conditioned noise sd: {condition(noise, DF_HAT[7]).std():.4f} (expect 1)")
    assert abs(condition(noise, DF_HAT[7]).std().item() - 1) < 0.02

    # 5. reference_posterior recovers a known Gaussian
    mu = torch.tensor([[1.0, 2.0, -1.0, 0.5, 0.0]], device=device)
    sd_true = torch.tensor([0.3, 0.1, 1.0, 0.2, 0.5], device=device)
    ref, accs = reference_posterior(lambda x: -0.5 * (((x - mu) / sd_true) ** 2).sum(-1),
                                    mu, mu[0], sd_true, n_steps=1500)
    print(f"reference_posterior on a Gaussian: sd ratio "
          f"{(ref.std(0) / sd_true).min():.3f} .. {(ref.std(0) / sd_true).max():.3f}, "
          f"accept {accs[-1]:.2f}")
    assert (ref.std(0) / sd_true - 1).abs().max() < 0.05

    # 6. pp_plot is flat for calibrated posteriors.  A calibrated toy: the posterior is
    #    N(theta_true + sigma * eta, sigma^2) with eta ~ N(0, 1) the "noise" of the observation,
    #    so the quantile of the truth is Phi(-eta), which is uniform.
    def toy(theta, shrink=1.0, sigma=0.1, n=400):
        centre = theta + sigma * torch.randn(1, 5, device=device)
        return centre + (sigma / shrink) * torch.randn(n, 5, device=device)

    injections = torch.randn(400, 5, device=device)
    q = pp_plot(toy, injections, labels=gb.LABELS)
    plt.savefig("/tmp/gb_wide_pp.png", dpi=80)
    print(f"pp_plot calibrated:    quantile mean {q.mean():.3f} (expect 0.5), "
          f"KS p {[round(_ks_pvalue(q[:, j]), 2) for j in range(5)]}")
    assert abs(q.mean() - 0.5) < 0.06
    assert min(_ks_pvalue(q[:, j]) for j in range(5)) > 0.005

    # ... and rejects posteriors that are 1.6x too narrow
    plt.figure()
    q_bad = pp_plot(lambda th: toy(th, shrink=1.6), injections, labels=gb.LABELS)
    plt.savefig("/tmp/gb_wide_pp_bad.png", dpi=80)
    print(f"pp_plot overconfident: KS p {[round(_ks_pvalue(q_bad[:, j]), 4) for j in range(5)]}")
    assert max(_ks_pvalue(q_bad[:, j]) for j in range(5)) < 0.01

    print("\nall self-tests passed")
