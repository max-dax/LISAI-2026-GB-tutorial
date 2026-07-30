"""Simplified LISA galactic-binary simulator (shared by Part 1 and Part 2).

The source of truth for this module is the simulator cell of
``part1_npe_SOLUTIONS.ipynb``.  Everything below the BEGIN marker is a verbatim
copy of that cell; only the imports and the device selection above it (which the
notebook does in its setup cell) are added here, so that the file can be used on
its own::

    from gb_simulator import sample_prior, simulate, log_likelihood

If you change the physics, change the notebook cell and re-copy.

The response model is deliberately schematic -- a quasi-monochromatic carrier
with Doppler phase modulation and a slowly rotating antenna pattern.  It is not
a TDI response and is not intended to be physically accurate; it exists to give
the inference problem realistic structure at negligible cost.

Interface (matches Max Dax's Part 2): sample_prior(n), simulate(theta),
log_likelihood(theta, d).  `signal` and `gb_envelope` are additional helpers.
"""

import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------
# BEGIN verbatim copy of the notebook simulator cell
# --------------------------------------------------------------------------

YEAR = 3.15581498e7          # sidereal-ish year [s]
T_OBS = 4 * YEAR             # observation time (LISA nominal mission)
N = 256                      # samples of the complex envelope
F_REF = 3e-3                 # heterodyne reference frequency [Hz]
R_AU_C = 499.0               # astronomical unit in light seconds
BETA, LAM = 0.5, 1.0         # ecliptic latitude / longitude (fixed & known here)
NOISE_STD = 1.0              # whitened noise, per real sample

t = torch.linspace(0, T_OBS, N, device=device)

PRIOR_LOW = torch.tensor([1.4, -1e-7, 3e-17, 0.0, -1.0], device=device)
PRIOR_HIGH = torch.tensor([7.0, 1e-7, 1e-16, 2 * np.pi, 1.0], device=device)
LABELS = [r"$A$", r"$\delta f\ [10^{-7}\,{\rm Hz}]$", r"$\dot f\ [10^{-17}\,{\rm Hz/s}]$",
          r"$\phi_0$", r"$\cos\iota$"]
PLOT_SCALE = torch.tensor([1.0, 1e7, 1e17, 1.0, 1.0], device=device)  # readable axes


def sample_prior(n):
    """Draw n parameter vectors from the (uniform, box) prior. -> (n, 5)"""
    return PRIOR_LOW + (PRIOR_HIGH - PRIOR_LOW) * torch.rand(n, 5, device=device)


def gb_envelope(theta):
    """Complex envelope of the heterodyned galactic-binary signal. (B, 5) -> (B, N) complex."""
    A, df, fdot, phi0, cosi = (theta[:, i:i + 1] for i in range(5))
    # intrinsic phase evolution relative to F_REF
    phase = 2 * np.pi * (df * t + 0.5 * fdot * t**2) + phi0
    # Doppler modulation from the orbit around the Sun
    phase = phase + 2 * np.pi * F_REF * R_AU_C * np.cos(BETA) * torch.cos(2 * np.pi * t / YEAR - LAM)
    # inclination-dependent polarization amplitudes
    a_plus, a_cross = (1 + cosi**2) / 2, cosi
    # schematic LISA-like antenna modulation (2 cycles/yr), not a TDI response
    f_plus = 0.5 * (1 + 0.6 * torch.cos(4 * np.pi * t / YEAR - 2 * LAM))
    f_cross = 0.5 * (1 + 0.6 * torch.sin(4 * np.pi * t / YEAR - 2 * LAM))
    return A * (f_plus * a_plus + 1j * f_cross * a_cross) * torch.exp(1j * phase)


def signal(theta):
    """Noise-free network features. (B, 5) -> (B, 2N) real, [Re, Im] concatenated."""
    y = gb_envelope(theta)
    return torch.cat([y.real, y.imag], dim=-1)


def simulate(theta):
    """(B, 5) -> (B, 2N) real network features [Re, Im], with whitened noise."""
    d = signal(theta)
    return d + NOISE_STD * torch.randn_like(d)


def log_likelihood(theta, d):
    """Exact Gaussian log likelihood (up to a constant), used for validation in Part 2."""
    return -0.5 * ((d - signal(theta)) ** 2).sum(-1) / NOISE_STD**2
