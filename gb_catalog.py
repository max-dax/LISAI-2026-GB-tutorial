"""Multi-source galactic-binary catalogs with a frozen single-source network (Part 1).

Builds directly on Stephen Green's ESTEC tutorial (https://github.com/stephengreen/
LISAAI-Hackathon-ESTEC, MIT); ``gb_simulator.py`` and ``gb_wide.py`` are vendored
verbatim from that repository, and everything here composes their pieces.  Part 2a of
that tutorial showed that one frozen narrow-prior network plus a frequency scan solves
a W-times-wider single-source problem.  This module turns the same scan into a
*catalog* pipeline for data containing many overlapping sources:

- ``CatalogSettings`` / ``sample_population`` / ``observe``: a configurable population
  of galactic binaries (number, amplitude range, frequency range) summed into one
  data stream with whitened noise;
- ``scan``: the frozen network run over every tile, scored by the per-tile log Bayes
  factor ln B_k against the no-source hypothesis.  The likelihood is *band-limited*,
  evaluated on the 2N conditioned features rather than the 2 N_WIDE raw ones, so the
  cost per tile does not grow with W -- the search stage Part 2a section 6 sketches;
- ``detect`` / ``match_candidates``: threshold ln B_k, group contiguous tiles into
  candidates, and match candidates to an injected truth catalog;
- ``tile_posterior`` / ``stitched_posterior``: Part 2a's importance-sampled tile
  posterior, against either the exact wide-record likelihood (default, for the final
  catalog) or the band-limited one (for cross-checks).

The evidence never appears alone: all weights are computed relative to the noise-only
likelihood, log L(d | theta) - log L(d | 0) = <d, s> - |s|^2 / 2, so the large
constant -|d|^2 / 2 cancels analytically and per-tile results are directly the
log Bayes factors we threshold on.

Run ``python gb_catalog.py`` for a self-test (uses ``checkpoints/part1_flow_v2.pt``).

    import gb_catalog as gbc
    settings = gbc.CatalogSettings(w=1000, n_sources=40)
    gbc.configure(settings)
    theta_true = gbc.sample_population(settings)
    d = gbc.observe(theta_true, seed=1)
    flow = gbc.load_part1_flow()
    result = gbc.scan(flow, gbc.condition_all(d))
    candidates = gbc.detect(result["log_B"], threshold)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import gb_simulator as gb
import gb_wide as gbw
from gb_simulator import N, device
from gb_wide import DF_P1, TILE_HALF

# Part 1's standardization, needed to talk to the Part 1 network (Part 2a, section 1).
P1_MEAN = (gb.PRIOR_LOW + gb.PRIOR_HIGH) / 2
P1_STD = (gb.PRIOR_HIGH - gb.PRIOR_LOW) / np.sqrt(12)


@dataclass
class CatalogSettings:
    """Knobs of the injected population and of the scan.

    Attributes
    ----------
    w : int
        Prior-widening factor: delta f spans +/- w * 1e-7 Hz (Part 2a used w = 20).
    r : int or None
        Oversampling factor, N_WIDE = r * N.  None selects w // 5, which keeps the
        wide grid's Nyquist frequency at twice the prior edge, Part 2a's headroom.
    n_sources : int
        Number of injected galactic binaries.
    amp_low, amp_high : float
        Uniform amplitude range.  The network's prior floor is 1.4 (SNR of about 6);
        drawing below it injects sources the model cannot represent -- the
        below-threshold population.
    seed : int
        Seed for the population draw (the observation takes its own).
    """

    w: int = 1000
    r: int = None
    n_sources: int = 40
    amp_low: float = 0.7
    amp_high: float = 7.0
    seed: int = 42


def configure(settings):
    """Configure gb_wide's grid and tiling for this catalog. Returns the tile count."""
    r = settings.r if settings.r is not None else settings.w // 5
    gbw.configure(settings.w, r)
    return len(gbw.DF_HAT)


def sample_population(settings):
    """Draw the injected catalog, sorted by frequency. -> (n_sources, 5)

    delta f is uniform over the widened prior less one tile at each edge, so that
    every source's tile lies fully inside the scan; the other four parameters follow
    Part 1's prior, except the amplitude which follows the settings' range.
    """
    torch.manual_seed(settings.seed)
    theta = gb.sample_prior(settings.n_sources)
    theta[:, 0] = settings.amp_low + (settings.amp_high - settings.amp_low) \
        * torch.rand(settings.n_sources, device=device)
    f_edge = (settings.w - 1) * DF_P1
    theta[:, 1] = (2 * torch.rand(settings.n_sources, device=device) - 1) * f_edge
    return theta[theta[:, 1].argsort()]


def observe(theta, seed):
    """One data stream containing every source in `theta`, plus whitened noise. -> (2 N_WIDE,)"""
    torch.manual_seed(seed)
    d = gbw.signal_wide(theta).sum(0)
    return d + gb.NOISE_STD * torch.randn_like(d)


def snr(theta):
    """Optimal SNR of each source alone in the wide record. (B, 5) -> (B,)"""
    return gbw.signal_wide(theta).pow(2).sum(-1).sqrt()


# --------------------------------------------------------------------------
# The frozen Part 1 network (architecture copied from the Part 1 notebook)
# --------------------------------------------------------------------------

def create_flow(num_flow_steps=5, param_dim=5, context_dim=2 * N, hidden_dim=128,
                num_transform_blocks=2):
    """Part 1's conditional flow, needed to load its checkpoint."""
    from torch import nn
    from glasflow.nflows import distributions, flows, transforms

    def base(param_dim, context_dim):
        return transforms.MaskedAffineAutoregressiveTransform(
            param_dim, hidden_features=hidden_dim, context_features=context_dim,
            num_blocks=num_transform_blocks, activation=nn.ELU(), use_batch_norm=False)

    def linear(param_dim):
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=param_dim),
            transforms.LULinear(param_dim, identity_init=True)])

    transform = transforms.CompositeTransform(
        [transforms.CompositeTransform([linear(param_dim), base(param_dim, context_dim)])
         for _ in range(num_flow_steps)] + [linear(param_dim)])
    return flows.Flow(transform, distributions.StandardNormal(shape=[param_dim]))


def load_part1_flow(path=None):
    """Load the frozen Part 1 checkpoint in eval mode."""
    if path is None:
        path = Path(__file__).parent / "checkpoints" / "part1_flow_v2.pt"
    flow = create_flow().to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    flow.load_state_dict(ckpt["state_dict"])
    flow.eval()
    return flow


# --------------------------------------------------------------------------
# The scan
# --------------------------------------------------------------------------

def condition_all(d, tile_chunk=128):
    """Condition the observation on every tile's proxy at once. (2 N_WIDE,) -> (K, 2 N)

    gb_wide.condition broadcasts over a batch of proxies; chunking bounds the
    (tile_chunk, N_WIDE) complex intermediate.
    """
    out = []
    for k in range(0, len(gbw.DF_HAT), tile_chunk):
        out.append(gbw.condition(d, gbw.DF_HAT[k:k + tile_chunk, None]))
    return torch.cat(out)


def _log_B_and_eps(log_w):
    """Mean-weight log evidence ratio and sample efficiency, batched. (..., n) -> 2 x (...,)

    Weights are relative to the noise-only likelihood, so the "evidence" is directly
    the log Bayes factor.  Tiles where no draw landed inside get -inf and eps = 0.
    """
    n = log_w.shape[-1]
    lse = torch.logsumexp(log_w, dim=-1)
    log_B = lse - np.log(n)
    log_eps = 2 * lse - np.log(n) - torch.logsumexp(2 * log_w, dim=-1)
    eps = torch.where(torch.isfinite(lse), log_eps.exp(), torch.zeros_like(lse))
    return log_B, eps


def _flow_proposal(flow, context, n_draws):
    """Draws and physical-unit log density from the frozen network. -> (B, n, 5), (B, n)"""
    with torch.no_grad():
        theta, log_q = flow.sample_and_log_prob(n_draws, context)
    return theta * P1_STD + P1_MEAN, log_q - torch.log(P1_STD).sum()


def _band_log_w(d_cond, theta_res, df_hat, log_q):
    """Band-limited importance weights relative to noise. (B, 2N), (B, n, 5) -> (B, n)

    The template is Part 1's signal at the residual parameters; the likelihood ratio
    <d, s> - |s|^2 / 2 uses only the retained band, so its cost is independent of W.
    Draws outside the tile-restricted prior get -inf.
    """
    B, n, _ = theta_res.shape
    s = gb.signal(theta_res.reshape(-1, 5)).reshape(B, n, 2 * N)
    log_l = (d_cond[:, None, :] * s).sum(-1) - 0.5 * s.pow(2).sum(-1)
    inside = ((theta_res[..., 0] > gb.PRIOR_LOW[0]) & (theta_res[..., 0] < gb.PRIOR_HIGH[0])
              & (theta_res[..., 1].abs() < TILE_HALF)
              & (theta_res[..., 2:] > gb.PRIOR_LOW[2:]).all(-1)
              & (theta_res[..., 2:] < gb.PRIOR_HIGH[2:]).all(-1))
    log_prior = -torch.log(gbw.PRIOR_HIGH_W - gbw.PRIOR_LOW_W).sum() \
        + np.log((gbw.PRIOR_HIGH_W[1] - gbw.PRIOR_LOW_W[1]).item() / (2 * TILE_HALF))
    return torch.where(inside, log_l + log_prior - log_q,
                       torch.full_like(log_q, -torch.inf))


def scan(flow, d_cond, n_draws=512, tile_chunk=32, seed=0):
    """Run the frozen network over every tile and score each against no-source.

    Parameters
    ----------
    flow : flows.Flow
        The frozen Part 1 network.
    d_cond : torch.Tensor
        Conditioned data for every tile, from `condition_all`, shape (K, 2 N).
    n_draws : int, optional
        Proposal draws per tile in this (search) pass.
    tile_chunk : int, optional
        Tiles processed per batch through the flow.
    seed : int, optional
        Seed for the proposal draws, for reproducibility.

    Returns
    -------
    dict
        "log_B" (K,): per-tile log Bayes factor, one source in this tile vs none;
        "eps" (K,): per-tile importance-sampling efficiency of the proposal.
    """
    torch.manual_seed(seed)
    K = len(gbw.DF_HAT)
    log_B = torch.full((K,), -torch.inf, device=device)
    eps = torch.zeros(K, device=device)
    for k in range(0, K, tile_chunk):
        sl = slice(k, min(k + tile_chunk, K))
        theta_res, log_q = _flow_proposal(flow, d_cond[sl], n_draws)
        log_w = _band_log_w(d_cond[sl], theta_res, gbw.DF_HAT[sl], log_q)
        log_B[sl], eps[sl] = _log_B_and_eps(log_w)
    return {"log_B": log_B, "eps": eps}


# --------------------------------------------------------------------------
# Detection, grouping, matching
# --------------------------------------------------------------------------

def detect(log_B, threshold, max_gap=1):
    """Group above-threshold tiles into candidates, one per (contiguous) run.

    A loud source can push a neighboring tile over threshold through the tails of its
    Doppler sidebands; runs separated by at most `max_gap` below-threshold tiles are
    merged so such leakage does not spawn spurious candidates.

    Returns a list of dicts sorted by frequency, each with "tiles" (list of tile
    indices), "k_peak" (the tile with the largest log B) and "log_B" (its value).
    """
    above = torch.nonzero(log_B > threshold).flatten().tolist()
    groups = []
    for k in above:
        if groups and k - groups[-1][-1] <= max_gap + 1:
            groups[-1].append(k)
        else:
            groups.append([k])
    out = []
    for tiles in groups:
        k_peak = max(tiles, key=lambda k: log_B[k].item())
        out.append({"tiles": tiles, "k_peak": k_peak, "log_B": log_B[k_peak].item()})
    return out


def match_candidates(candidates, theta_true, pad=1.0):
    """Greedily match candidates to injected sources by frequency.

    A source can match a candidate if its delta f lies within the candidate's tile
    span padded by `pad` tiles.  Candidates are matched loudest-first, each to the
    nearest still-unmatched source.  Adds "matched" (source index or None) to each
    candidate and returns a boolean per-source detected mask.
    """
    f_true = theta_true[:, 1]
    taken = torch.zeros(len(theta_true), dtype=torch.bool, device=theta_true.device)
    for c in sorted(candidates, key=lambda c: -c["log_B"]):
        lo = gbw.DF_HAT[c["tiles"][0]] - (0.5 + pad) * DF_P1
        hi = gbw.DF_HAT[c["tiles"][-1]] + (0.5 + pad) * DF_P1
        dist = (f_true - gbw.DF_HAT[c["k_peak"]]).abs()
        dist[(f_true < lo) | (f_true > hi) | taken] = torch.inf
        i = int(dist.argmin())
        c["matched"] = i if torch.isfinite(dist[i]) else None
        if c["matched"] is not None:
            taken[i] = True
    return taken


# --------------------------------------------------------------------------
# Per-candidate posteriors (Part 2a's tile_posterior, weights relative to noise)
# --------------------------------------------------------------------------

def tile_posterior(flow, d, k, n=8000, likelihood="full", draw_chunk=512, seed=0):
    """Frozen network on tile k, importance-weighted -- Part 2a section 4.

    Parameters
    ----------
    flow : flows.Flow
        The frozen Part 1 network.
    d : torch.Tensor
        The wide-grid observation, shape (2 N_WIDE,).
    k : int
        Tile index into gb_wide.DF_HAT.
    n : int, optional
        Number of proposal draws.
    likelihood : {"full", "band"}, optional
        "full" evaluates the exact wide-record likelihood ratio (the default for the
        final catalog); "band" reuses the scan's cheap band-limited one.
    draw_chunk : int, optional
        Draws per batch through the wide-record likelihood.
    seed : int, optional
        Seed for the proposal draws.

    Returns
    -------
    dict
        "theta" (n, 5): draws in absolute parameters; "log_w" (n,): log importance
        weights relative to the noise-only likelihood; "log_B", "eps": tile log
        Bayes factor and sampling efficiency.
    """
    torch.manual_seed(seed)
    df_hat = gbw.DF_HAT[k]
    d_cond = gbw.condition(d, df_hat)
    theta_res, log_q = _flow_proposal(flow, d_cond[None], n)
    if likelihood == "band":
        log_w = _band_log_w(d_cond[None], theta_res, df_hat[None], log_q)[0]
    else:
        theta_abs = theta_res[0].clone()
        theta_abs[:, 1] = theta_abs[:, 1] + df_hat
        log_l = torch.empty(n, device=device)
        for i in range(0, n, draw_chunk):
            s = gbw.signal_wide(theta_abs[i:i + draw_chunk])
            log_l[i:i + draw_chunk] = (d * s).sum(-1) - 0.5 * s.pow(2).sum(-1)
        log_w = log_l + gbw.log_prior_tile(theta_abs, df_hat) - log_q[0]
    theta = theta_res[0].clone()
    theta[:, 1] = theta[:, 1] + df_hat
    log_B, eps = _log_B_and_eps(log_w)
    return {"theta": theta, "log_w": log_w, "log_B": log_B.item(), "eps": eps.item()}


def stitched_posterior(flow, d, candidate, n_per_tile=8000, n_keep=8000, **kwargs):
    """Posterior samples for one candidate, stitched over its tiles -- Part 2a section 4.

    Each tile in the candidate is importance-sampled separately; tiles are then drawn
    in proportion to their Bayes factors and samples within each tile by weight.
    Returns (samples (n_keep, 5), info dict of the peak tile).
    """
    posts = {k: tile_posterior(flow, d, k, n_per_tile, **kwargs) for k in candidate["tiles"]}
    log_B = torch.tensor([posts[k]["log_B"] for k in candidate["tiles"]])
    share = torch.exp(log_B - log_B.max())
    share = share / share.sum()
    parts = []
    for k, w_tile in zip(candidate["tiles"], share):
        n_k = int(round(float(w_tile) * n_keep))
        if n_k == 0:
            continue
        p = posts[k]
        idx = torch.multinomial(torch.exp(p["log_w"] - p["log_w"].max()), n_k, replacement=True)
        parts.append(p["theta"][idx])
    return torch.cat(parts), posts[candidate["k_peak"]]


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    settings = CatalogSettings(w=100, n_sources=3, seed=0)
    K = configure(settings)
    print(f"self-test at W = {settings.w}: {K} tiles, N_WIDE = {gbw.N_WIDE}")

    # 1. conditioning still reproduces the Part 1 problem at this W
    th = gb.sample_prior(64)
    th[:, 1] = (torch.rand(64, device=device) - 0.5) * 2 * TILE_HALF
    th_w = th.clone()
    th_w[:, 1] = gbw.DF_HAT[K // 3] + th[:, 1]
    err = ((gbw.condition(gbw.signal_wide(th_w), gbw.DF_HAT[K // 3]) - gb.signal(th))
           .pow(2).sum(-1).sqrt() / gb.signal(th).pow(2).sum(-1).sqrt())
    noise_sd = gbw.condition(torch.randn(500, 2 * gbw.N_WIDE, device=device),
                             gbw.DF_HAT[K // 3]).std()
    print(f"conditioning: relative error max {err.max():.4f}, noise sd {noise_sd:.4f}")
    assert err.max() < 0.06 and abs(noise_sd - 1) < 0.02

    # 2. three hand-placed sources: loud, medium, and below the network's prior floor
    theta_true = torch.tensor([[6.0, -7.03e-6, 6e-17, 1.0, 0.7],
                               [3.0, 2.31e-6, 5e-17, 4.0, -0.4],
                               [0.9, 8.02e-6, 8e-17, 2.5, 0.1]], device=device)
    print(f"injected SNRs: {[f'{s:.1f}' for s in snr(theta_true).tolist()]}")
    d = observe(theta_true, seed=1)

    flow = load_part1_flow()
    t0 = time.time()
    d_cond = condition_all(d)
    result = scan(flow, d_cond)
    print(f"scan: {K} tiles in {time.time() - t0:.1f} s")

    # 3. threshold from a noise-only scan of the same pipeline
    torch.manual_seed(2)
    noise = gb.NOISE_STD * torch.randn(2 * gbw.N_WIDE, device=device)
    log_B_noise = scan(flow, condition_all(noise))["log_B"]
    threshold = log_B_noise.max().item() + 2.0
    print(f"noise-only log B: max {log_B_noise.max():.1f} -> threshold {threshold:.1f}")

    candidates = detect(result["log_B"], threshold)
    detected = match_candidates(candidates, theta_true)
    k_true = [int((gbw.DF_HAT - f).abs().argmin()) for f in theta_true[:, 1]]
    print(f"candidates: {[(c['k_peak'], round(c['log_B'], 1)) for c in candidates]}; "
          f"true tiles {k_true}; detected mask {detected.tolist()}")
    assert detected[0] and detected[1], "loud and medium sources must be detected"
    assert all(c["matched"] is not None for c in candidates), "no false alarms expected"

    # 4. leakage profile around the loud source: how far off-tile does log B stay up?
    near = [(k - k_true[0], round(result["log_B"][k].item(), 1))
            for k in range(k_true[0] - 4, k_true[0] + 5)]
    print(f"log B around the loud source (offset, log B): {near}")

    # 5. exact-likelihood posterior of the loud source covers the truth
    samples, info = stitched_posterior(flow, d, candidates[0], n_per_tile=8000)
    pull = (samples.median(0).values - theta_true[0]) / samples.std(0)
    print(f"loud source: eps {info['eps']:.3f}, log B {info['log_B']:.0f}, "
          f"pulls {[f'{p:.2f}' for p in pull.tolist()]}")
    assert info["eps"] > 0.005 and pull.abs().max() < 4

    # 6. band-limited and exact weights agree up to a near-constant offset
    p_full = tile_posterior(flow, d, k_true[0], n=4000, likelihood="full")
    p_band = tile_posterior(flow, d, k_true[0], n=4000, likelihood="band")
    both = torch.isfinite(p_full["log_w"]) & torch.isfinite(p_band["log_w"])
    diff = (p_band["log_w"] - p_full["log_w"])[both]
    print(f"band - full log w: mean {diff.mean():.2f}, sd {diff.std():.3f} "
          f"({int(both.sum())} finite draws)")
    assert diff.std() < 1.0

    print("\nall self-tests passed")
