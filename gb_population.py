"""A galactic-binary population prior, and NPE training data for its tiles (Part 2).

Part 1 injected a hand-picked list of sources into white noise. Here one population prior
generates *everything*: a Poisson process in frequency with a power-law amplitude
distribution, whose loud tail is the catalog and whose faint bulk is the confusion
background. The inference target of the Part 2 network is, per tile, the parameters of the
loudest source whose frequency lies in that tile ("loudest" by noise-free SNR, a
deterministic function of the source parameters, so the target is well defined whatever the
noise does). Everything else in the band is marginalized over.

The training loop never needs full records. Restricting a Poisson population to a band is
again a Poisson population, so training examples are generated band-locally: draw the
sources within +/- WINDOW of the tile center, sum their conditioned waveforms, add
instrument noise. Band-local waveforms are produced on a small oversampled grid (the
W = 20 grid of Stephen Green's Part 2a) and brick-wall conditioned to the 256-sample
network format, via a precomputed bank: amplitude and initial phase act on a stored
waveform exactly (a real scale and a global complex rotation), so only the bank itself
needs waveform evaluations.

    import gb_population as gbp
    pop = gbp.PopulationSettings()
    bank = gbp.build_bank(pop)
    d, theta, snr = gbp.sample_tiles(bank, 512, pop)      # training batch
    theta_all = gbp.sample_population(pop, w=1000)        # a full record's worth
    d_rec = gbp.make_record(theta_all, pop)

Run ``python gb_population.py`` for a self-test.
"""

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

import gb_simulator as gb
import gb_wide as gbw
from gb_simulator import N, device
from gb_wide import DF_P1, DT, TILE_HALF

# Band-local generation grid: Part 2a's W = 20 grid. Nyquist 4 uHz covers the retained
# band (+/- 1.01 uHz) plus the margin sources whose leakage still reaches it.
R_LOC = 4
N_LOC = R_LOC * N
t_loc = torch.arange(N_LOC, device=device) * (DT / R_LOC)
WINDOW = 1.25e-6             # half-width of the band-local source window [Hz]


@dataclass
class PopulationSettings:
    """The population prior. One object generates records, training data, and validation.

    Attributes
    ----------
    rho : float
        Source density per uHz of band.
    alpha : float
        Amplitude power-law slope, dN/dA ~ A^-alpha.
    a_min, a_max : float
        Amplitude range. a_min sets the faint bulk (the confusion floor), a_max the
        loudest binaries.
    noise_std : float
        Instrument noise per whitened sample; 0 makes the record pure binaries.
    seed : int
        Base seed for population draws.
    """

    rho: float = 150.0
    alpha: float = 3.3
    a_min: float = 0.1
    a_max: float = 7.0
    noise_std: float = 1.0
    seed: int = 0


def _sample_amplitude(n, s, g):
    """Power-law amplitudes via inverse CDF. -> (n,)"""
    p = 1 - s.alpha
    u = torch.rand(n, generator=g, device=device)
    return (s.a_min**p + u * (s.a_max**p - s.a_min**p)) ** (1 / p)


def _sample_source(n, s, g, f_half):
    """n population sources with delta f uniform in +/- f_half. -> (n, 5)"""
    theta = torch.empty(n, 5, device=device)
    theta[:, 0] = _sample_amplitude(n, s, g)
    theta[:, 1] = (2 * torch.rand(n, generator=g, device=device) - 1) * f_half
    for i in (2, 3, 4):
        theta[:, i] = gb.PRIOR_LOW[i] + (gb.PRIOR_HIGH[i] - gb.PRIOR_LOW[i]) \
            * torch.rand(n, generator=g, device=device)
    return theta


def sample_population(s, w, seed=None):
    """All sources of one record: a Poisson draw over the +/- w * 1e-7 Hz band. -> (n, 5)"""
    g = torch.Generator(device=device)
    g.manual_seed(s.seed if seed is None else seed)
    f_half = w * DF_P1
    n = int(torch.poisson(torch.tensor(2 * f_half * 1e6 * s.rho, device=device),
                          generator=g).item())
    return _sample_source(n, s, g, f_half)


def make_record(theta, s, seed=1, chunk=500):
    """Sum every source onto the wide grid, add instrument noise. -> (2 N_WIDE,)

    Requires gb_wide to be configured for the record's W (e.g. via gb_catalog.configure).
    """
    d = torch.zeros(2 * gbw.N_WIDE, device=device)
    for i in range(0, len(theta), chunk):
        d += gbw.signal_wide(theta[i:i + chunk]).sum(0)
    torch.manual_seed(seed)
    return d + s.noise_std * torch.randn_like(d)


# --------------------------------------------------------------------------
# Band-local generation
# --------------------------------------------------------------------------

def local_signal(theta):
    """Conditioned band waveform of sources near one tile center. (B, 5) -> (B, N) complex.

    theta[:, 1] is the frequency *relative to the tile center*, |delta f| < WINDOW. The
    waveform is evaluated on the small oversampled grid and brick-wall conditioned to the
    network's 256-sample format, in Part 1 units (the same double 1/sqrt(R) convention as
    signal_wide followed by condition).
    """
    z = gbw.envelope(theta, t_loc) / np.sqrt(R_LOC)
    spectrum = torch.fft.fftshift(torch.fft.fft(z), dim=-1)
    lo = (N_LOC - N) // 2
    return torch.fft.ifft(torch.fft.ifftshift(spectrum[..., lo:lo + N], dim=-1)) \
        / np.sqrt(R_LOC)


def build_bank(s, n_bank=65536, seed=100):
    """Precompute unit-amplitude, zero-phase conditioned waveforms. -> dict

    Bank entries sample (delta f, fdot, cos iota) from the population restricted to the
    local window; amplitude and phi0 are applied at draw time (scale and rotation, exact).
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    theta = _sample_source(n_bank, s, g, WINDOW)
    theta[:, 0] = 1.0
    theta[:, 3] = 0.0
    wave = torch.empty(n_bank, N, dtype=torch.complex64, device=device)
    for i in range(0, n_bank, 2048):
        wave[i:i + 2048] = local_signal(theta[i:i + 2048]).to(torch.complex64)
    return {"wave": wave, "theta": theta,
            "norm": wave.abs().pow(2).sum(-1).sqrt(),
            "in_tile": theta[:, 1].abs() <= TILE_HALF,
            "n_bank": n_bank, "seed": seed, "settings": asdict(s)}


def sample_tiles(bank, n, s, seed=None, k_chunk=128):
    """A batch of training tiles from the population prior.

    Parameters
    ----------
    bank : dict
        From `build_bank`.
    n : int
        Batch size.
    s : PopulationSettings
        Population and noise settings (must match the bank's).
    seed : int or None, optional
        Seed for this batch; None leaves torch's global generator state in charge.
    k_chunk : int, optional
        Sources summed per memory chunk.

    Returns
    -------
    d : torch.Tensor
        Network inputs [Re, Im], shape (n, 2 N).
    theta_label : torch.Tensor
        The loudest in-tile source of each example, shape (n, 5); delta f is relative to
        the tile center.
    snr_label : torch.Tensor
        Its noise-free SNR, shape (n,).
    """
    if seed is not None:
        torch.manual_seed(seed)
    d, theta_label, snr_label = _sample_tiles_once(bank, n, s, k_chunk)
    # A tile without any in-tile source is a Poisson(0.1 rho) zero: ~e^-15 with the
    # default population, but over tens of millions of training draws it does occur.
    # The target is undefined there, so those rows are redrawn (rejection sampling
    # against an essentially-probability-one condition).
    bad = snr_label <= 0
    while bad.any():
        d_r, th_r, s_r = _sample_tiles_once(bank, int(bad.sum()), s, k_chunk)
        d[bad], theta_label[bad], snr_label[bad] = d_r, th_r, s_r
        bad[bad.clone()] = s_r <= 0
    return d, theta_label, snr_label


def _sample_tiles_once(bank, n, s, k_chunk):
    """One unconditioned batch; rows with an empty tile carry snr_label = 0."""
    lam = 2 * WINDOW * 1e6 * s.rho
    k_max = int(lam + 5 * np.sqrt(lam))
    count = torch.poisson(torch.full((n,), lam, device=device))
    active = torch.arange(k_max, device=device)[None, :] < count[:, None]
    idx = torch.randint(bank["n_bank"], (n, k_max), device=device)
    # amplitudes and phases are redrawn per draw even when a bank entry repeats
    p = 1 - s.alpha
    u = torch.rand(n, k_max, device=device)
    amp = (s.a_min**p + u * (s.a_max**p - s.a_min**p)) ** (1 / p)
    phi = 2 * np.pi * torch.rand(n, k_max, device=device)

    snr = amp * bank["norm"][idx]
    snr_in = torch.where(active & bank["in_tile"][idx], snr,
                         torch.zeros_like(snr))
    best = snr_in.argmax(-1)

    rows = torch.arange(n, device=device)
    theta_label = bank["theta"][idx[rows, best]].clone()
    theta_label[:, 0] = amp[rows, best]
    theta_label[:, 3] = phi[rows, best]
    snr_label = torch.where(snr_in[rows, best] > 0, snr[rows, best],
                            torch.zeros_like(best, dtype=snr.dtype))

    z = torch.zeros(n, N, dtype=torch.complex64, device=device)
    rot = (amp * torch.polar(torch.ones_like(phi), phi)).to(torch.complex64)
    rot = torch.where(active, rot, torch.zeros_like(rot))
    for k in range(0, k_max, k_chunk):
        z += (rot[:, k:k + k_chunk, None] * bank["wave"][idx[:, k:k + k_chunk]]).sum(1)
    d = torch.cat([z.real, z.imag], dim=-1)
    return d + s.noise_std * torch.randn_like(d), theta_label, snr_label


# --------------------------------------------------------------------------
# Label transform: the flow trains on standardized (log A, delta f, fdot, phi0, cos i)
# --------------------------------------------------------------------------

def label_stats(bank, s, n=100_000):
    """Mean and std of the transformed label under the population. -> (2, 5)"""
    _, theta, _ = sample_tiles(bank, n, s, seed=12345)
    t = theta.clone()
    t[:, 0] = torch.log(t[:, 0])
    return torch.stack([t.mean(0), t.std(0)])


def to_std(theta, stats):
    """Physical label -> standardized flow space. (B, 5) -> (B, 5)"""
    t = theta.clone()
    t[..., 0] = torch.log(t[..., 0])
    return (t - stats[0]) / stats[1]


def from_std(z, stats):
    """Standardized flow space -> physical label. (B, 5) -> (B, 5)"""
    t = z * stats[1] + stats[0]
    t = t.clone()
    t[..., 0] = torch.exp(t[..., 0])
    return t


# --------------------------------------------------------------------------
# The Part 2 network: neural spline flow (same scaffolding as Part 1's MAF)
# --------------------------------------------------------------------------

def create_flow(arch="nsf", param_dim=5, context_dim=2 * N, num_flow_steps=5,
                hidden_dim=128, num_transform_blocks=2, num_bins=8, tail_bound=5.0):
    """Part 1's flow construction with the affine step swappable for a spline step.

    The spline gives the capacity that per-tile posteriors need here (railed-faint
    amplitudes, and bimodal near-ties between two comparable sources); training on fresh
    simulations every batch removes the overfitting risk that made the spline fail
    Stephen's fixed-budget challenge.
    """
    from torch import nn
    from glasflow.nflows import distributions, flows, transforms

    def base():
        if arch == "maf":
            return transforms.MaskedAffineAutoregressiveTransform(
                param_dim, hidden_features=hidden_dim, context_features=context_dim,
                num_blocks=num_transform_blocks, activation=nn.ELU(), use_batch_norm=False)
        return transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=param_dim, hidden_features=hidden_dim, context_features=context_dim,
            num_blocks=num_transform_blocks, num_bins=num_bins, tails="linear",
            tail_bound=tail_bound, activation=nn.ELU(), use_batch_norm=False)

    def linear():
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=param_dim),
            transforms.LULinear(param_dim, identity_init=True)])

    transform = transforms.CompositeTransform(
        [transforms.CompositeTransform([linear(), base()]) for _ in range(num_flow_steps)]
        + [linear()])
    return flows.Flow(transform, distributions.StandardNormal(shape=[param_dim]))


def load_population_flow(path=None):
    """Load a checkpoint from train_confusion_flow.py. -> (flow, stats, PopulationSettings)"""
    if path is None:
        path = Path(__file__).parent / "checkpoints" / "part2_population_flow.pt"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    flow = create_flow(**ckpt["arch"]).to(device)
    flow.load_state_dict(ckpt["state_dict"])
    flow.eval()
    return flow, ckpt["stats"].to(device), PopulationSettings(**ckpt["population"])


# --------------------------------------------------------------------------
# Inference on records
# --------------------------------------------------------------------------

def tile_posterior_draws(flow, stats, d_cond, n_draws=256, tile_chunk=64):
    """Posterior draws for every tile. (K, 2N) -> theta (K, n, 5), snr (K, n)

    delta f in the returned theta is relative to each tile's proxy. The SNR per draw is
    the noise-free SNR of the drawn source, the quantity detection thresholds on.
    """
    K = len(d_cond)
    theta = torch.empty(K, n_draws, 5, device=device)
    snr = torch.empty(K, n_draws, device=device)
    for k in range(0, K, tile_chunk):
        sl = slice(k, min(k + tile_chunk, K))
        with torch.no_grad():
            z, _ = flow.sample_and_log_prob(n_draws, d_cond[sl])
        th = from_std(z, stats)
        theta[sl] = th
        snr[sl] = gb.signal(th.reshape(-1, 5)).pow(2).sum(-1).sqrt().reshape(th.shape[:2])
    return theta, snr


def loudest_per_tile(theta_pop, w):
    """Ground truth for a record: the loudest in-tile source of every tile.

    Returns (theta (K, 5) with absolute delta f, snr (K,)); tiles are gb_wide.DF_HAT.
    """
    snr_all = gbw.signal_wide(theta_pop).pow(2).sum(-1).sqrt() if len(theta_pop) < 5000 \
        else torch.cat([gbw.signal_wide(theta_pop[i:i + 2000]).pow(2).sum(-1).sqrt()
                        for i in range(0, len(theta_pop), 2000)])
    K = len(gbw.DF_HAT)
    k_idx = torch.bucketize(theta_pop[:, 1].contiguous(),
                            (gbw.DF_HAT + TILE_HALF).contiguous())
    theta_best = torch.zeros(K, 5, device=device)
    snr_best = torch.zeros(K, device=device)
    for i in range(len(theta_pop)):
        k = int(k_idx[i])
        if 0 <= k < K and snr_all[i] > snr_best[k]:
            snr_best[k] = snr_all[i]
            theta_best[k] = theta_pop[i]
    return theta_best, snr_best


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import gb_catalog as gbc

    pop = PopulationSettings()
    torch.manual_seed(0)

    # 1. local_signal matches gb.signal for in-tile sources (double-conditioning units)
    th = gb.sample_prior(64)
    th[:, 1] = (torch.rand(64, device=device) - 0.5) * 2 * TILE_HALF
    z = local_signal(th)
    s_ref = gb.signal(th)
    err = ((torch.cat([z.real, z.imag], -1) - s_ref).pow(2).sum(-1).sqrt()
           / s_ref.pow(2).sum(-1).sqrt())
    print(f"local_signal vs Part 1 signal: max rel err {err.max():.4f}")
    assert err.max() < 0.06

    # 2. bank + batch generation: shapes, label sanity, occupancy
    bank = build_bank(pop, n_bank=16384)
    d, theta_lab, snr_lab = sample_tiles(bank, 2048, pop, seed=7)
    print(f"batch: d {tuple(d.shape)}, label SNR median {snr_lab.median():.2f}, "
          f"q90 {snr_lab.quantile(0.9):.1f}, max {snr_lab.max():.1f}")
    assert d.shape == (2048, 2 * N) and (theta_lab[:, 1].abs() <= TILE_HALF).all()

    # 3. the band generator agrees with conditioned full records: compare the per-feature
    #    variance of the clean confusion signal (population noise) between both paths
    K_conf = gbc.configure(gbc.CatalogSettings(w=1000))
    quiet = PopulationSettings(noise_std=0.0)
    theta_pop = sample_population(quiet, w=1000, seed=3)
    d_rec = make_record(theta_pop, quiet, seed=3)
    d_cond = gbc.condition_all(d_rec)
    var_record = d_cond.var().item()
    d_band, _, _ = sample_tiles(bank, 2048, quiet, seed=8)
    var_band = d_band.var().item()
    print(f"clean per-feature variance: record tiles {var_record:.3f}, "
          f"band generator {var_band:.3f}")
    assert abs(var_record - var_band) / var_record < 0.15

    # 4. label transform round trip
    stats = label_stats(bank, pop, n=20_000)
    back = from_std(to_std(theta_lab, stats), stats)
    assert (back - theta_lab).abs().max() < 1e-4
    print(f"label stats (log A, df, fdot, phi0, cos i): mean {stats[0].tolist()}")

    # 5. flow constructs, samples, and evaluates in both architectures
    for arch in ("nsf", "maf"):
        flow = create_flow(arch=arch).to(device)
        with torch.no_grad():
            zz, lp = flow.sample_and_log_prob(8, d[:4])
        n_par = sum(p.numel() for p in flow.parameters())
        print(f"{arch}: {n_par:,} parameters, sample {tuple(zz.shape)} ok")

    # 6. loudest_per_tile agrees with the label convention on a record
    theta_best, snr_best = loudest_per_tile(theta_pop, w=1000)
    frac = (snr_best > 0).float().mean()
    print(f"record ground truth: {frac:.3f} of tiles occupied, "
          f"SNR median {snr_best[snr_best > 0].median():.2f}, max {snr_best.max():.0f}")
    assert frac > 0.999

    print("\nall self-tests passed")
