"""
Cheap, non-generative uncertainty signals -- alternatives to the flow's
sample-spread variance.

  - epistemic_uncertainty (Stretch Goal 1): distance to k-NN reference
    latents. Cells far from any reference cell are extrapolations.
  - integration_uncertainty (Stretch Goal 3): weight-perturbation Monte
    Carlo through the encoder.

Note on SG3: the original plan was MC-dropout, but scMODAL's encoder
(scmodal.networks.encoder) contains no nn.Dropout layers -- confirmed by
checking model.E_A.modules(). MC-dropout would silently produce zero
variation. The working alternative is small Gaussian perturbations
injected directly into the encoder's linear-layer weights (W_1, b_1, W_2,
b_2), run n_samples times, with weights restored after each pass. See
decisions.md for the noise-scale selection (0.20, chosen via a
signal-to-latent-magnitude ratio sweep over [0.01, 0.05, 0.1, 0.2, 0.3]).
"""

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_epistemic_uncertainty(z_query: np.ndarray, z_reference: np.ndarray, k: int = 10) -> np.ndarray:
    """Mean distance from each query latent to its k nearest reference latents."""
    nn_model = NearestNeighbors(n_neighbors=k)
    nn_model.fit(z_reference)
    distances, _ = nn_model.kneighbors(z_query)
    return distances.mean(axis=1)


def mc_noise_forward(encoder_module, x: torch.Tensor, n_samples: int = 20, noise_scale: float = 0.01) -> torch.Tensor:
    """Runs a scMODAL encoder forward n_samples times with small Gaussian
    noise added to its linear-layer weights (W_1, b_1, W_2, b_2), restoring
    the original weights after each pass.

    Returns stacked latents of shape (n_samples, n_cells, latent_dim).

    Default noise_scale=0.01 matches the original notebook functions' own
    default (cells 159, 174) -- every actual call site in this project
    explicitly overrides it to 0.20 (the value chosen via the sweep below),
    so this default is never actually used for real results, but is kept
    faithful to the original rather than silently changed (found during
    the reproducibility audit; see decisions.md).

    noise_scale=0.20 was chosen by sweeping [0.01, 0.05, 0.1, 0.2, 0.3] and
    picking the value giving a reasonable MC-std-to-latent-magnitude ratio
    -- too small and the signal is dominated by float noise, too large and
    it swamps the actual latent structure.
    """
    param_names = ["W_1", "b_1", "W_2", "b_2"]
    originals = {name: getattr(encoder_module, name).clone() for name in param_names}

    outputs = []
    with torch.no_grad():
        for _ in range(n_samples):
            for name in param_names:
                orig = originals[name]
                noise = torch.randn_like(orig) * noise_scale * orig.std()
                getattr(encoder_module, name).copy_(orig + noise)
            outputs.append(encoder_module(x).unsqueeze(0))
        # restore originals
        for name in param_names:
            getattr(encoder_module, name).copy_(originals[name])

    return torch.cat(outputs, dim=0)


def integration_uncertainty_from_samples(z_samples: torch.Tensor) -> torch.Tensor:
    """Per-cell integration-confidence score: average std across latent
    dims, over the MC sample axis (dim 0)."""
    return z_samples.std(dim=0).mean(dim=1)
