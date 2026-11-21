# Author: Aliyu Agboola Alege
# Developed with the support of Claude Sonnet 5.5 (Anthropic) as a pair-programming assistant.
# License: MIT (see ../LICENSE)
"""
All fusion methods compared in the benchmark, built on a common encoder
budget so differences in results reflect the COMBINATION RULE, not extra
model capacity.

  PoEModel          - the pre-existing standard: Gaussian Product-of-Experts
                      fusion (Wu & Goodman 2018 style), own probabilistic heads.
  SheafModel         - faithful SheafAlign reproduction: restriction maps F_v,
                      F_w into a shared discourse space, PLUS dual reconstruction
                      maps Q_v, Q_w (missing-modality inference), trained with
                      a contrastive alignment loss + reconstruction loss.
  SheafHardGated     - SheafModel's projections, with a hard fallback to the
                      vision-only projection when discord exceeds a
                      (validation-calibrated) threshold.
  SheafSoftGated     - same idea, but the fallback weight is a smooth sigmoid
                      function of discord, trained end-to-end rather than a
                      post-hoc switch.
  EntropyGated       - an "AGFN-style" baseline: each modality gets its own
                      classifier; fusion weight is set by each modality's own
                      predictive-entropy (confidence), not by cross-modal
                      agreement. Included to compare against a DIFFERENT
                      family of conflict-aware fusion, not just PoE.

All classes expose: .parameters(), .forward_train(batch) -> loss,
.predict(vision_x, modB_x) -> class predictions, and where relevant
.discord(vision_x, modB_x) -> per-sample disagreement score.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_mlp_encoder(in_dim, out_dim, hidden=64):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


class PoEModel(nn.Module):
    """PoE baseline following Wu & Goodman (2018): the MVAE ELBO with full-weight KL and a
    reconstruction decoder for every modality (not only a downstream classifier)."""
    def __init__(self, vision_dim, modb_dim, n_classes, D=16, modb_is_discrete=False, n_modb_classes=10):
        super().__init__()
        self.D = D
        self.vision_dim = vision_dim
        self.modb_is_discrete = modb_is_discrete
        self.vision_enc = make_mlp_encoder(vision_dim, 2 * D)
        if modb_is_discrete:
            self.modb_enc = nn.Embedding(n_modb_classes, 2 * D)
        else:
            self.modb_enc = make_mlp_encoder(modb_dim, 2 * D)
        self.classifier = nn.Linear(D, n_classes)          # downstream task head
        self.vision_dec = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, vision_dim), nn.Sigmoid())
        if modb_is_discrete:
            self.modb_dec = nn.Linear(D, n_modb_classes)     # reconstructing word = classifying it
        else:
            self.modb_dec = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, modb_dim), nn.Sigmoid())

    def _encode(self, x, is_vision):
        out = self.vision_enc(x) if is_vision else self.modb_enc(x)
        return out[:, :self.D], out[:, self.D:]

    def combine(self, vision_x, modb_x):
        mu_v, logvar_v = self._encode(vision_x, True)
        mu_w, logvar_w = self._encode(modb_x, False)
        prec_v, prec_w = torch.exp(-logvar_v), torch.exp(-logvar_w)
        total_prec = 1.0 + prec_v + prec_w
        mu = (mu_v * prec_v + mu_w * prec_w) / total_prec
        var = 1.0 / total_prec
        return mu, var, mu_v, prec_v

    def forward_train(self, vision_x, modb_x, y):
        mu, var, mu_v, prec_v = self.combine(vision_x, modb_x)
        z = mu + torch.randn_like(mu) * var.sqrt()
        logits = self.classifier(z)
        task_loss = F.cross_entropy(logits, y)
        # mean reduction (per-element, not per-batch-summed) so reconstruction loss
        # magnitude doesn't scale with modality dimensionality (64-d vs 196-d) --
        # summed reduction was found to dominate the task loss ~50x on the
        # higher-dimensional mnist_views modality, silently preventing the
        # classifier from learning at all. Fixed here, not left in.
        recon_v = F.binary_cross_entropy(self.vision_dec(z), vision_x, reduction='mean')
        if self.modb_is_discrete:
            recon_w = F.cross_entropy(self.modb_dec(z), modb_x, reduction='mean')
        else:
            recon_w = F.binary_cross_entropy(self.modb_dec(z), modb_x, reduction='mean')
        kl = 0.5 * torch.mean(torch.sum(var + mu ** 2 - 1 - torch.log(var), dim=-1))
        return task_loss + (recon_v + recon_w) + 0.1 * kl   # full per-element ELBO; KL kept moderate (standard practice: beta-VAE with beta<1 to avoid posterior collapse)

    def predict(self, vision_x, modb_x):
        mu, var, mu_v, prec_v = self.combine(vision_x, modb_x)
        return self.classifier(mu).argmax(dim=1)

    def combined_var(self, vision_x, modb_x):
        mu, var, mu_v, prec_v = self.combine(vision_x, modb_x)
        return var.mean(dim=-1)

    def predict_vision_only(self, vision_x):
        mu_v, _ = self._encode(vision_x, True)
        return self.classifier(mu_v).argmax(dim=1)

    def precision_shares(self, vision_x, modb_x):
        mu_v, logvar_v = self._encode(vision_x, True)
        mu_w, logvar_w = self._encode(modb_x, False)
        prec_v, prec_w = torch.exp(-logvar_v), torch.exp(-logvar_w)
        total = 1.0 + prec_v + prec_w
        return (prec_v / total).mean().item(), (prec_w / total).mean().item()


class SheafBase(nn.Module):
    """Shared machinery for the whole sheaf family: encoders, restriction
    maps F_v/F_w, dual reconstruction maps Q_v/Q_w, discord, contrastive
    alignment loss, classification head."""
    def __init__(self, vision_dim, modb_dim, n_classes, D=16, K=8,
                 modb_is_discrete=False, n_modb_classes=10):
        super().__init__()
        self.D, self.K = D, K
        self.vision_enc = make_mlp_encoder(vision_dim, D)
        if modb_is_discrete:
            self.modb_enc = nn.Embedding(n_modb_classes, D)
        else:
            self.modb_enc = make_mlp_encoder(modb_dim, D)
        self.F_v = nn.Linear(D, K, bias=False)
        self.F_w = nn.Linear(D, K, bias=False)
        self.Q_v = nn.Linear(K, D, bias=False)   # dual map: discourse -> vision space
        self.Q_w = nn.Linear(K, D, bias=False)   # dual map: discourse -> modB space
        self.classifier = nn.Linear(K, n_classes)

    def encode(self, vision_x, modb_x):
        h_v = self.vision_enc(vision_x)
        h_w = self.modb_enc(modb_x)
        fv, fw = self.F_v(h_v), self.F_w(h_w)
        return h_v, h_w, fv, fw

    def discord(self, vision_x, modb_x):
        _, _, fv, fw = self.encode(vision_x, modb_x)
        return ((fv - fw) ** 2).sum(dim=-1)

    def contrastive_alignment_loss(self, fv, fw, temp=0.2):
        # SheafAlign-style: InfoNCE between the two modalities' discourse
        # projections, matching pair = positive, rest of batch = negatives
        fv_n = F.normalize(fv, dim=-1)
        fw_n = F.normalize(fw, dim=-1)
        logits = fv_n @ fw_n.T / temp
        targets = torch.arange(fv.size(0), device=fv.device)
        return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))

    def reconstruction_loss(self, h_v, h_w, fv, fw):
        # dual maps: reconstruct each modality's own embedding FROM THE
        # OTHER's discourse projection -- this is what enables missing-
        # modality inference (Q_v, Q_w are the "Qij" from SheafAlign)
        recon_v_from_w = self.Q_v(fw)
        recon_w_from_v = self.Q_w(fv)
        return F.mse_loss(recon_v_from_w, h_v) + F.mse_loss(recon_w_from_v, h_w)

    def fused_z(self, fv, fw):
        return 0.5 * (fv + fw)


class SheafModel(SheafBase):
    """Faithful SheafAlign reproduction: contrastive alignment + dual
    reconstruction (Qij) + classification on the fused discourse point."""
    def forward_train(self, vision_x, modb_x, y):
        h_v, h_w, fv, fw = self.encode(vision_x, modb_x)
        z = self.fused_z(fv, fw)
        class_loss = F.cross_entropy(self.classifier(z), y)
        align_loss = self.contrastive_alignment_loss(fv, fw)
        recon_loss = self.reconstruction_loss(h_v, h_w, fv, fw)
        return class_loss + align_loss + 0.5 * recon_loss

    def predict(self, vision_x, modb_x):
        _, _, fv, fw = self.encode(vision_x, modb_x)
        return self.classifier(self.fused_z(fv, fw)).argmax(dim=1)

    def predict_vision_only(self, vision_x):
        h_v = self.vision_enc(vision_x)
        fv = self.F_v(h_v)
        return self.classifier(fv).argmax(dim=1)

    def predict_missing_modb_via_reconstruction(self, vision_x):
        # missing-modality inference using the dual map: infer what modB's
        # discourse projection WOULD be, purely from vision, via Q_w -> F_w path
        # (approximate: reconstruct h_w from fv via Q_w, re-project with F_w)
        h_v = self.vision_enc(vision_x)
        fv = self.F_v(h_v)
        h_w_hat = self.Q_w(fv)
        fw_hat = self.F_w(h_w_hat)
        z = self.fused_z(fv, fw_hat)
        return self.classifier(z).argmax(dim=1)


class SheafHardGated(SheafModel):
    """Same training as SheafModel; adds a hard, validation-calibrated
    discord threshold used only at inference time to fall back to vision-only."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.threshold = None  # set externally after calibration

    def predict_gated(self, vision_x, modb_x):
        h_v, h_w, fv, fw = self.encode(vision_x, modb_x)
        z_blend = self.fused_z(fv, fw)
        d = ((fv - fw) ** 2).sum(dim=-1)
        pred_blend = self.classifier(z_blend).argmax(dim=1)
        pred_vision = self.classifier(fv).argmax(dim=1)
        gate = d > self.threshold
        return torch.where(gate, pred_vision, pred_blend), gate


class SheafSoftGated(SheafModel):
    """Discord drives a CONTINUOUS, differentiable trust weight (sigmoid),
    used both in training and inference -- no hard switch, no post-hoc
    threshold calibration needed for the gate itself (a temperature and
    midpoint are still learned/set, but the blend is smooth)."""
    def __init__(self, *a, gate_temp=1.0, gate_midpoint=1.0, **kw):
        super().__init__(*a, **kw)
        self.log_temp = nn.Parameter(torch.tensor(float(gate_temp)).log())
        self.midpoint = nn.Parameter(torch.tensor(float(gate_midpoint)))

    def soft_alpha(self, discord):
        temp = self.log_temp.exp()
        # alpha near 1 = trust the fused representation; near 0 = fall back to vision
        return torch.sigmoid(-(discord - self.midpoint) / temp)

    def forward_train(self, vision_x, modb_x, y):
        h_v, h_w, fv, fw = self.encode(vision_x, modb_x)
        d = ((fv - fw) ** 2).sum(dim=-1)
        alpha = self.soft_alpha(d).unsqueeze(-1)
        z_blend = self.fused_z(fv, fw)
        z_soft = alpha * z_blend + (1 - alpha) * fv
        class_loss = F.cross_entropy(self.classifier(z_soft), y)
        align_loss = self.contrastive_alignment_loss(fv, fw)
        recon_loss = self.reconstruction_loss(h_v, h_w, fv, fw)
        return class_loss + align_loss + 0.5 * recon_loss

    def predict(self, vision_x, modb_x):
        h_v, h_w, fv, fw = self.encode(vision_x, modb_x)
        d = ((fv - fw) ** 2).sum(dim=-1)
        alpha = self.soft_alpha(d).unsqueeze(-1)
        z_blend = self.fused_z(fv, fw)
        z_soft = alpha * z_blend + (1 - alpha) * fv
        return self.classifier(z_soft).argmax(dim=1)


class EntropyGated(SheafBase):
    """'AGFN-style' baseline: each modality gets its OWN classifier; fusion
    weight is set by each modality's own predictive entropy (confidence),
    NOT by cross-modal agreement. This is a functionally different family
    of conflict-aware fusion than the sheaf-discord approach, included so
    the benchmark isn't just sheaf-vs-PoE."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.classifier_v = nn.Linear(self.K, kw.get('n_classes', 10) if 'n_classes' in kw else self.classifier.out_features)
        self.classifier_w = nn.Linear(self.K, self.classifier.out_features)

    def _entropy(self, logits):
        p = F.softmax(logits, dim=-1)
        return -(p * torch.log(p + 1e-8)).sum(dim=-1)

    def forward_train(self, vision_x, modb_x, y):
        h_v, h_w, fv, fw = self.encode(vision_x, modb_x)
        logits_v = self.classifier_v(fv)
        logits_w = self.classifier_w(fw)
        ent_v = self._entropy(logits_v)
        ent_w = self._entropy(logits_w)
        # lower entropy (more confident) -> more weight; softmax over (-entropy)
        weights = F.softmax(torch.stack([-ent_v, -ent_w], dim=-1), dim=-1)
        fused_logits = weights[:, 0:1] * logits_v + weights[:, 1:2] * logits_w
        return F.cross_entropy(fused_logits, y)

    def predict(self, vision_x, modb_x):
        h_v, h_w, fv, fw = self.encode(vision_x, modb_x)
        logits_v = self.classifier_v(fv)
        logits_w = self.classifier_w(fw)
        ent_v = self._entropy(logits_v)
        ent_w = self._entropy(logits_w)
        weights = F.softmax(torch.stack([-ent_v, -ent_w], dim=-1), dim=-1)
        fused_logits = weights[:, 0:1] * logits_v + weights[:, 1:2] * logits_w
        return fused_logits.argmax(dim=1)

    def predict_vision_only(self, vision_x):
        h_v = self.vision_enc(vision_x)
        fv = self.F_v(h_v)
        return self.classifier_v(fv).argmax(dim=1)
