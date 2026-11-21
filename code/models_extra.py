# Author: Aliyu Agboola Alege
# Developed with the support of Claude Sonnet 5.5 (Anthropic) as a pair-programming assistant.
# License: MIT (see ../LICENSE)
"""Additional baselines (published-style defences).

PDFLateFusion : port of the late-fusion branch of Predictive Dynamic Fusion (Cao et al., ICML 2024),
                official code https://github.com/Yinan-Xia/PDF (src/models/latefusion_pdf.py, train_pdf.py):
                confidence nets predicting true-class probability, holo-confidence, relative calibration.
                Only the encoders are replaced by this repo's MLPs.
KimGated      : gated information fusion in the spirit of Kim et al. (ACCV 2018): per-feature sigmoid
                gates computed from both modalities weight each modality's features before classification.
                Simplified re-implementation (no official code located).
OOOFusion     : simplified odd-one-out fusion in the spirit of Yang et al. (CVPR 2021): a network predicts
                {clean, v corrupted, w corrupted} from the two embeddings and routes the prediction to the
                uncorrupted source. Supervised with corruption labels. Simplified re-implementation (no
                official code located); adversarial training of the gate is NOT replicated.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from models import make_mlp_encoder


def _encs(vision_dim, modb_dim, D, discrete, n_modb):
    ev = make_mlp_encoder(vision_dim, D)
    ew = nn.Embedding(n_modb, D) if discrete else make_mlp_encoder(modb_dim, D)
    return ev, ew


class PDFLateFusion(nn.Module):
    def __init__(self, vision_dim, modb_dim, n_classes, D=64, modb_is_discrete=False, n_modb_classes=10):
        super().__init__()
        self.C = n_classes
        self.enc_v, self.enc_w = _encs(vision_dim, modb_dim, D, modb_is_discrete, n_modb_classes)
        self.clf_v, self.clf_w = nn.Linear(D, n_classes), nn.Linear(D, n_classes)
        mk = lambda: nn.Sequential(nn.Linear(D, 2 * D), nn.Linear(2 * D, D), nn.Linear(D, 1), nn.Sigmoid())
        self.conf_v, self.conf_w = mk(), mk()

    def parts(self, x, w):
        hv, hw = self.enc_v(x), self.enc_w(w)
        return hv, hw, self.clf_v(hv), self.clf_w(hw)

    @staticmethod
    def _holo(tv, tw):
        tv, tw = tv.clamp(1e-6, 1), tw.clamp(1e-6, 1)
        den = torch.log(tv * tw) - 1e-8
        return torch.log(tw) / den, torch.log(tv) / den, tv, tw

    def loss(self, x, w, y):
        hv, hw, lv, lw = self.parts(x, w)
        tv, tw = self.conf_v(hv.detach()), self.conf_w(hw.detach())
        hov, how, tv_, tw_ = self._holo(tv, tw)
        cb = torch.stack((tv_.detach() + hov.detach(), tw_.detach() + how.detach()), 1)
        wt = F.softmax(cb, 1)
        out = wt[:, 0] * lv + wt[:, 1] * lw
        oh = F.one_hot(y, self.C)
        tcp_v = (F.softmax(lv, 1) * oh).max(1, keepdim=True).values.detach()
        tcp_w = (F.softmax(lw, 1) * oh).max(1, keepdim=True).values.detach()
        tcp = F.l1_loss(tv, tcp_v) + F.l1_loss(tw, tcp_w)
        return F.cross_entropy(lv, y) + F.cross_entropy(lw, y) + F.cross_entropy(out, y) + tcp

    def logits(self, x, w):                      # 'pdf_test' branch of the official code
        hv, hw, lv, lw = self.parts(x, w)
        hov, how, tv, tw = self._holo(self.conf_v(hv), self.conf_w(hw))
        cbv, cbw = tv + hov, tw + how
        du_v = (F.softmax(lv, 1) - 1 / self.C).abs().mean(1, keepdim=True)
        du_w = (F.softmax(lw, 1) - 1 / self.C).abs().mean(1, keepdim=True)
        cond = du_v > du_w
        rcv = torch.where(cond, torch.ones_like(du_v), du_v / (du_w + 1e-12))
        rcw = torch.where(cond, du_w / (du_v + 1e-12), torch.ones_like(du_w))
        wt = F.softmax(torch.stack((cbv * rcv, cbw * rcw), 1), 1)
        return wt[:, 0] * lv + wt[:, 1] * lw

    def predict(self, x, w):
        return self.logits(x, w).argmax(1)


class KimGated(nn.Module):
    def __init__(self, vision_dim, modb_dim, n_classes, D=64, modb_is_discrete=False, n_modb_classes=10):
        super().__init__()
        self.enc_v, self.enc_w = _encs(vision_dim, modb_dim, D, modb_is_discrete, n_modb_classes)
        self.g_v, self.g_w = nn.Linear(2 * D, D), nn.Linear(2 * D, D)
        self.clf = nn.Linear(D, n_classes)

    def logits(self, x, w):
        hv, hw = self.enc_v(x), self.enc_w(w)
        c = torch.cat([hv, hw], -1)
        return self.clf(torch.sigmoid(self.g_v(c)) * hv + torch.sigmoid(self.g_w(c)) * hw)

    def loss(self, x, w, y):
        return F.cross_entropy(self.logits(x, w), y)

    def predict(self, x, w):
        return self.logits(x, w).argmax(1)


class OOOFusion(nn.Module):
    """labels: 0 clean, 1 v corrupted, 2 w corrupted"""
    def __init__(self, vision_dim, modb_dim, n_classes, D=64, modb_is_discrete=False, n_modb_classes=10):
        super().__init__()
        self.enc_v, self.enc_w = _encs(vision_dim, modb_dim, D, modb_is_discrete, n_modb_classes)
        self.head_f = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU(), nn.Linear(64, n_classes))
        self.head_v, self.head_w = nn.Linear(D, n_classes), nn.Linear(D, n_classes)
        self.ooo = nn.Sequential(nn.Linear(2 * D, 64), nn.ReLU(), nn.Linear(64, 3))

    def parts(self, x, w):
        hv, hw = self.enc_v(x), self.enc_w(w)
        c = torch.cat([hv, hw], -1)
        return self.head_f(c), self.head_v(hv), self.head_w(hw), self.ooo(c)

    def loss(self, x, w, y, lab):
        lf, lv, lw, lo = self.parts(x, w)
        l = F.cross_entropy(lo, lab)
        for logits, keep in ((lf, lab == 0), (lv, lab != 1), (lw, lab != 2)):
            if keep.any():
                l = l + F.cross_entropy(logits[keep], y[keep])
        return l

    def logits(self, x, w):                       # differentiable surrogate of the routed output
        lf, lv, lw, lo = self.parts(x, w)
        p = F.softmax(lo, 1)
        return p[:, 0:1] * lf + p[:, 1:2] * lw + p[:, 2:3] * lv

    def predict(self, x, w, return_route=False):
        lf, lv, lw, lo = self.parts(x, w)
        r = lo.argmax(1)
        out = torch.where((r == 0).unsqueeze(1), lf, torch.where((r == 1).unsqueeze(1), lw, lv)).argmax(1)
        return (out, r) if return_route else out

    def score(self, x, w):                        # suspicion score: P(some source corrupted)
        return 1 - F.softmax(self.parts(x, w)[3], 1)[:, 0]
