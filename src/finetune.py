"""
Stretch Goal 4 -- down-weighting uncertain cross-modal anchor links during
fine-tuning.

This is the corrected, baseline-matched version. Fine-tuning itself (even
with floor=1.0, i.e. no down-weighting) shifts accuracy by about -0.66%
relative to the un-fine-tuned model_v2 checkpoint -- so every treatment
delta below is measured against a *seed-matched floor=1.0 control*, not
against the raw checkpoint. An earlier, uncorrected version of this
function (measuring deltas against the raw checkpoint directly) is not
included here; see decisions.md for why it was replaced.

Result (validated null): down-weighting does not improve overall accuracy
and in fact regresses the high-uncertainty tertile by +2.68% on average
across a 5-seed sweep (std 0.38%; values [3.25, 2.17, 2.38, 2.69, 2.93] --
see cell 190's printed output). This is confirmed as a real, dose-dependent
effect via a 4-point floor-strength sweep, and confirmed to be driven by
the down-weighting mechanism itself (not the choice of uncertainty signal)
via a robustness check using the SG3 integration-uncertainty signal
(r=0.12, weaker correlation with error than SG1) in place of the SG1
epistemic signal.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def run_sg4_finetune(
    model_v2,
    gt_protein_v2: np.ndarray,
    seed: int,
    unc: np.ndarray,
    floor: float = 0.2,
    steps: int = 2000,
    baseline_err: np.ndarray | None = None,
    ckpt_path: str = "CITE-seq_PBMC_v2/ckpt.pth",
):
    """Fine-tunes fresh encoder/generator copies from the model_v2
    checkpoint, with the cross-modal anchor (MNN) loss reweighted per-cell
    by `weight = 1.0 - (1.0 - floor) * normalized_uncertainty`.

    If `baseline_err` is provided, tertile deltas are computed against that
    per-cell error array (e.g. a seed-matched floor=1.0 control) instead of
    the raw, un-fine-tuned model_v2 predictions. Passing floor=1.0 with no
    baseline_err produces that control run itself.

    Returns (summary_dict, per_cell_err) -- per_cell_err is always returned
    so a floor=1.0 run's error array can be reused as `baseline_err` for
    matched treatment runs at the same seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Lazy import: scmodal is only importable once the submodule is on sys.path
    # (added by the calling script) -- keeping this inside the function, not at
    # module level, means `import src.finetune` itself never fails just because
    # scMODAL isn't set up yet. Matches the lazy-import pattern already used in
    # src/teaseq_prep.py. Found and fixed during the reproducibility audit --
    # a module-level import here would have broken on any fresh clone.
    from scmodal.utils import acquire_pairs
    from scmodal.networks import encoder, generator, discriminator

    device = model_v2.device
    N_LATENT = 20

    E_A_s = encoder(model_v2.emb_A.shape[1], N_LATENT).to(device)
    E_B_s = encoder(model_v2.emb_B.shape[1], N_LATENT).to(device)
    G_A_s = generator(model_v2.emb_A.shape[1], N_LATENT).to(device)
    G_B_s = generator(model_v2.emb_B.shape[1], N_LATENT).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    E_A_s.load_state_dict(ckpt["E_A"]); E_B_s.load_state_dict(ckpt["E_B"])
    G_A_s.load_state_dict(ckpt["G_A"]); G_B_s.load_state_dict(ckpt["G_B"])
    D_Z_s = discriminator(N_LATENT).to(device)

    unc_norm = (unc - unc.min()) / (unc.max() - unc.min())
    weight_A_full = 1.0 - (1.0 - floor) * unc_norm

    batch_size = model_v2.batch_size
    n_KNN = model_v2.n_KNN
    shared_gene_num = model_v2.shared_gene_num
    lambdaAE, lambdaLA, lambdaMNN, lambdaGeo, lambdaGAN = (
        model_v2.lambdaAE, model_v2.lambdaLA, model_v2.lambdaMNN, model_v2.lambdaGeo, model_v2.lambdaGAN
    )
    params_G = list(E_A_s.parameters()) + list(E_B_s.parameters()) + list(G_A_s.parameters()) + list(G_B_s.parameters())
    optimizer_G = optim.Adam(params_G, lr=0.0001, weight_decay=0.001)
    optimizer_D = optim.Adam(list(D_Z_s.parameters()), lr=0.0001, weight_decay=0.001)
    E_A_s.train(); E_B_s.train(); G_A_s.train(); G_B_s.train(); D_Z_s.train()

    N_A = model_v2.emb_A.shape[0]
    N_B = model_v2.emb_B.shape[0]
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    for step in range(steps):
        index_A = np.random.choice(np.arange(N_A), size=batch_size)
        index_B = np.random.choice(np.arange(N_B), size=batch_size)
        x_A = torch.from_numpy(model_v2.emb_A[index_A, :]).float().to(device)
        x_B = torch.from_numpy(model_v2.emb_B[index_B, :]).float().to(device)
        z_A = E_A_s(x_A); z_B = E_B_s(x_B)
        x_AtoB = G_B_s(z_A); x_BtoA = G_A_s(z_B)
        x_Arecon = G_A_s(z_A); x_Brecon = G_B_s(z_B)
        z_AtoB = E_B_s(x_AtoB); z_BtoA = E_A_s(x_BtoA)
        K_A = torch.mean((x_A.view(batch_size, 1, -1) - x_A.view(1, batch_size, -1)) ** 2, dim=2); K_A = torch.exp(-K_A / 2)
        K_B_z = torch.mean((z_B.view(batch_size, 1, -1) - z_B.view(1, batch_size, -1)) ** 2, dim=2); K_B_z = torch.exp(-K_B_z / 2)
        K_B = torch.mean((x_B.view(batch_size, 1, -1) - x_B.view(1, batch_size, -1)) ** 2, dim=2); K_B = torch.exp(-K_B / 2)
        K_A_z = torch.mean((z_A.view(batch_size, 1, -1) - z_A.view(1, batch_size, -1)) ** 2, dim=2); K_A_z = torch.exp(-K_A_z / 2)

        for _ in range(5):
            optimizer_D.zero_grad()
            loss_D = (torch.log(1 + torch.exp(-D_Z_s(z_A))) + torch.log(1 + torch.exp(D_Z_s(z_B)))).mean()
            loss_D.backward(retain_graph=True)
            optimizer_D.step()

        loss_AE = torch.mean((x_Arecon - x_A) ** 2) + torch.mean((x_Brecon - x_B) ** 2)
        loss_LA = torch.mean((z_A - z_AtoB) ** 2) + torch.mean((z_B - z_BtoA) ** 2)
        loss_G_GAN = -(torch.log(1 + torch.exp(-D_Z_s(z_A))) + torch.log(1 + torch.exp(D_Z_s(z_B)))).mean()
        loss_Geo = -(torch.clamp(cos(K_A, K_A_z), max=0.975).mean() + torch.clamp(cos(K_B, K_B_z), max=0.975).mean())

        Sim = acquire_pairs(model_v2.emb_A[index_A, :shared_gene_num], model_v2.emb_B[index_B, :shared_gene_num], k=n_KNN)
        Sim = torch.from_numpy(Sim).float().to(device)
        batch_weight_A = torch.from_numpy(weight_A_full[index_A]).float().to(device)
        Sim_weighted = Sim * batch_weight_A.view(batch_size, 1)
        z_dist = torch.mean((z_A.view(batch_size, 1, -1) - z_B.view(1, batch_size, -1)) ** 2, dim=2)
        loss_MNN = torch.sum(Sim_weighted * z_dist) / torch.sum(Sim_weighted)

        optimizer_G.zero_grad()
        loss_G = lambdaGAN * loss_G_GAN + lambdaAE * loss_AE + lambdaLA * loss_LA + lambdaMNN * loss_MNN + lambdaGeo * loss_Geo
        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(params_G, 5.0)
        optimizer_G.step()

    E_A_s.eval(); G_B_s.eval()
    with torch.no_grad():
        x_A_full = torch.from_numpy(model_v2.emb_A).float().to(device)
        pred = G_B_s(E_A_s(x_A_full)).cpu().numpy()

    per_cell_err = np.mean((pred - gt_protein_v2) ** 2, axis=1)
    overall_corr = np.nanmean([np.corrcoef(pred[:, i], gt_protein_v2[:, i])[0, 1] for i in range(pred.shape[1])])

    tertiles = np.quantile(unc, [1 / 3, 2 / 3])
    low_m = unc <= tertiles[0]
    mid_m = (unc > tertiles[0]) & (unc <= tertiles[1])
    high_m = unc > tertiles[1]
    ref = baseline_err if baseline_err is not None else np.mean(
        (model_v2.G_B(model_v2.E_A(torch.from_numpy(model_v2.emb_A).float().to(device))).detach().cpu().numpy() - gt_protein_v2) ** 2,
        axis=1,
    )

    summary = {}
    for name, m in [("low", low_m), ("mid", mid_m), ("high", high_m)]:
        summary[name] = 100 * (per_cell_err[m].mean() - ref[m].mean()) / ref[m].mean()
    summary["overall_corr"] = overall_corr
    return summary, per_cell_err
